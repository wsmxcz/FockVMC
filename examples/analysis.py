from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np

from detnqs import operator, utils
from detnqs.model import GBackflow
from detnqs.sampler import Chains
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


def main() -> None:
    # Runtime.
    utils.batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    H = operator.Hamiltonian.load("N2_ham.npz")
    sector = H.sector

    model = GBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(64,),
    )
    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=sector.reference(sampler.n_chains),
        key=jax.random.key(1),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    checkpoint = utils.checkpoint.load("N2_00500.npz")
    saved = checkpoint["state"]
    sampler_state = saved["sampler_state"]

    state = state.replace(
        params=saved["params"],
        sampler_state=Chains(
            key=jax.device_put(sampler_state["key"]),
            x=np.ascontiguousarray(sampler_state["x"], dtype=np.uint64),
            logabs=utils.precision.cast(
                sampler_state["logabs"],
                "calc",
                "real",
                host=True,
            ),
            alpha=float(np.asarray(sampler_state["alpha"])),
        ),
        chains=np.ascontiguousarray(saved["chains"], dtype=np.uint64),
    )

    obs = {"s2": operator.S2(sector)}
    post_state = state.replace(
        sampler=replace(
            state.sampler,
            n_samples=819200,
            alpha=float(state.sampler_state.alpha),
        ),
    )

    post = utils.analysis.estimate(post_state, n_blocks=8, obs=obs, profile=True)
    _, snap, data = post_state.expect(obs=obs, profile=True, data=True)

    sup = utils.analysis.support(data["w"])
    exc = utils.analysis.excitation(data["x"], data["w"], sector.reference(1))
    tail = utils.analysis.tail(data["w"], data["eloc"], snap["energy"])

    _, _, _, _, geometry = post_state.expect_and_grad(
        geometry=True,
        obs=obs,
        profile=True,
    )
    sr = utils.analysis.sr(geometry, shift=1.0e-3)

    print("Posterior")
    print(f"Energy     : {post['energy']:.12f}")
    print(f"Energy SE  : {post['energy_se']:.3e}")
    print(f"Eloc var   : {post['eloc_var']:.6e}")
    print(f"ESS frac   : {post['ess_frac']:.3%}")
    print(f"ESSu frac  : {post['essu_frac']:.3%}")
    print(f"W max      : {post['w_max']:.3e}")
    print(f"S^2        : {post['s2']:.8f}")
    print(f"S^2 var    : {post['s2_var']:.3e}")

    print("\nSupport")
    print(f"N2         : {sup['n2']:.3f}")
    print(f"Entropy    : {sup['entropy']:.6f}")
    print(f"Top 1      : {sup['top1']:.3%}")
    print(f"Top 5      : {sup['top5']:.3%}")
    print(f"Top 10     : {sup['top10']:.3%}")
    print(f"Top 100    : {sup['top100']:.3%}")

    print("\nExcitation")
    for key in sorted(exc, key=lambda s: int(s[3:])):
        print(f"{key:<10s} : {exc[key]:.3%}")

    print("\nTail")
    print(f"R q90      : {tail['r_q90']:.6e}")
    print(f"R q99      : {tail['r_q99']:.6e}")
    print(f"R q999     : {tail['r_q999']:.6e}")
    print(f"Var max    : {tail['var_max']:.6e}")

    print("\nSR")
    print(f"Tr QGT     : {sr['tr_qgt']:.6e}")
    print(f"Rank eff   : {sr['rank_eff']:.3f}")
    print(f"SR force   : {sr['sr_force']:.6e}")
    print(f"SR damp    : {sr['sr_damp']:.6e}")


if __name__ == "__main__":
    main()
