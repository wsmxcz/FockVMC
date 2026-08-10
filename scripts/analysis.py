"""Read a VMC checkpoint and print posterior wavefunction diagnostics."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from detnqs import Hamiltonian, MCState
from detnqs.model import PBackflow
from detnqs.model.base import to_ratio
from detnqs.operator import S2
from detnqs.sampler import MCSampler
from detnqs.utils import batch, checkpoint, precision, tree


def support(weight: np.ndarray) -> dict[str, float]:
    """Return participation and concentration diagnostics."""
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight = weight[np.isfinite(weight) & (weight > 0.0)]
    weight /= np.sum(weight)
    ordered = np.sort(weight)[::-1]

    return {
        "n2": float(1.0 / np.sum(weight * weight)),
        "entropy": float(-np.sum(weight * np.log(weight))),
        "top1": float(np.sum(ordered[:1])),
        "top10": float(np.sum(ordered[:10])),
        "top100": float(np.sum(ordered[:100])),
    }


def tail(
    weight: np.ndarray,
    eloc: np.ndarray,
    energy: float,
) -> dict[str, float]:
    """Return weighted local-energy residual quantiles."""
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    residual = np.abs(np.asarray(eloc).reshape(-1) - float(energy))
    order = np.argsort(residual)
    residual = residual[order]
    weight = weight[order] / np.sum(weight)
    cdf = np.cumsum(weight)

    return {
        label: float(
            residual[min(np.searchsorted(cdf, q, side="left"), residual.size - 1)]
        )
        for label, q in (("q90", 0.9), ("q99", 0.99), ("q999", 0.999))
    }


def spin_correlation(
    state: MCState,
    x: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """Return the active-orbital matrix <S_p . S_q>."""
    x = state.hamiltonian.sector.asarray(x)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight /= np.sum(weight)

    norb = state.hamiltonian.sector.norb
    orbital = np.arange(norb, dtype=np.int64)
    word = orbital >> 6
    bit = np.left_shift(np.uint64(1), (orbital & 63).astype(np.uint64))
    alpha = (x[:, 0, word] & bit) != 0
    beta = (x[:, 1, word] & bit) != 0
    single = alpha ^ beta
    sz = 0.5 * (alpha.astype(np.float64) - beta.astype(np.float64))

    corr = np.einsum("n,np,nq->pq", weight, sz, sz, optimize=True)
    np.fill_diagonal(corr, 0.75 * np.einsum("n,np->p", weight, single))

    _, ptr, bra, value = S2(state.hamiltonian.sector).local_conn(x)
    count = np.diff(ptr)
    if value.size:
        ket_index = np.repeat(np.arange(x.shape[0], dtype=np.int64), count)
        p = np.empty(value.size, dtype=np.int64)
        q = np.empty(value.size, dtype=np.int64)

        for i, (lo, hi) in enumerate(zip(ptr[:-1], ptr[1:], strict=True)):
            if lo == hi:
                continue
            alpha_only = np.flatnonzero(alpha[i] & ~beta[i])
            beta_only = np.flatnonzero(beta[i] & ~alpha[i])
            p[lo:hi] = np.repeat(alpha_only, beta_only.size)
            q[lo:hi] = np.tile(beta_only, alpha_only.size)

        ket_logpsi = batch.apply(state.model.logpsi, state.params, x)
        bra_logpsi = batch.apply(state.model.logpsi, state.params, bra)
        jax.block_until_ready((ket_logpsi, bra_logpsi))
        ket_logpsi = tree.host(ket_logpsi)
        bra_logpsi = tree.host(bra_logpsi)
        ratio = np.asarray(
            to_ratio(
                bra_logpsi,
                jax.tree.map(lambda a: a[ket_index], ket_logpsi),
            )
        ).reshape(-1)

        transverse = 0.5 * weight[ket_index] * value * ratio
        np.add.at(corr, (p, q), np.real(transverse))
        np.add.at(corr, (q, p), np.real(transverse))

    return corr


def main() -> None:
    npz = Path("H2O_ccpvdz_1.0re_03000.npz")
    fcidump = (
        Path(__file__).with_name("FCIDUMP")
        / "H2O_ccpvdz"
        / "H2O_ccpvdz_1.0re.FCIDUMP"
    )
    n_samples = 819_200

    batch.configure(
        forward_chunk=32768,
        backward_chunk=4096,
        param_chunk=None,
        bucket_min=4096,
    )
    precision.configure("double")

    saved_state = checkpoint.load(npz, key="state")
    saved_sampler = saved_state["sampler_state"]
    params = saved_state["params"]

    hamiltonian = Hamiltonian.load(fcidump)
    sector = hamiltonian.sector
    hidden_names = sorted(
        (key for key in params if key.startswith("hidden_")),
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    hidden = tuple(int(np.asarray(params[key]["bias"]).size) for key in hidden_names)
    model = PBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=hidden,
    )

    chains = np.ascontiguousarray(saved_sampler["x"], dtype=np.uint64)
    sampler = MCSampler(
        n_samples=n_samples,
        n_chains=chains.shape[0],
        thermal_steps=4096,
        discard_steps=256,
        proposal="single",
        blur=0.0,
        alpha=2.0,
    )
    sampler_state = sampler.init(
        params,
        hamiltonian,
        model,
        key=jax.device_put(saved_sampler["key"]),
        eps1=0.0,
        chains=chains,
    )
    state = MCState(
        model=model,
        params=params,
        hamiltonian=hamiltonian,
        sampler=sampler,
        sampler_state=sampler_state,
        chains=chains,
        eps1=0.0,
        eps2=0.0,
        eloc_sample=0,
    )

    state, stats, data = state.expect(data=True)
    sup = support(data["weight"])
    residual = tail(data["weight"], data["eloc"], stats["energy"])
    corr = spin_correlation(state, data["x"], data["weight"])

    np.set_printoptions(precision=6, suppress=True, linewidth=160)
    print(f"checkpoint : {npz}")
    print(f"FCIDUMP    : {fcidump}")
    print(f"active     : ({sector.nelec}e, {sector.norb}o)")
    print(f"samples    : {n_samples}")
    print(f"energy     : {stats['energy']:.12f}")
    print(f"variance   : {stats['eloc_var']:.6e}")
    print(f"S^2        : {np.sum(corr):.8f}")
    print(f"particip.  : {sup['n2']:.3f}")
    print(f"entropy    : {sup['entropy']:.6f}")
    print(f"top 1      : {sup['top1']:.3%}")
    print(f"top 10     : {sup['top10']:.3%}")
    print(f"top 100    : {sup['top100']:.3%}")
    print(f"tail q90   : {residual['q90']:.6e}")
    print(f"tail q99   : {residual['q99']:.6e}")
    print(f"tail q999  : {residual['q999']:.6e}")
    print("spin correlation <S_p . S_q>")
    print(corr)


if __name__ == "__main__":
    main()
