"""Read a VMC checkpoint and analyze its wavefunction structure."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from detnqs import Hamiltonian, MCState
from detnqs.model import PBackflow
from detnqs.operator import density_correlation, rdm1, spin_correlation
from detnqs.sampler import MCSampler
from detnqs.utils import batch, checkpoint, precision


def main() -> None:
    checkpoint_name = "H2O_ccpvdz_1.0re_05000.npz"
    fcidump_name = "H2O_ccpvdz_1.0re.FCIDUMP"
    npz = next(Path.cwd().rglob(checkpoint_name))
    fcidump = next(Path.cwd().rglob(fcidump_name))

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
    hidden = tuple(
        int(np.asarray(params[key]["bias"]).size)
        for key in hidden_names
    )

    model = PBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=hidden,
    )

    chains = np.ascontiguousarray(saved_sampler["x"], dtype=np.uint64)
    sampler = MCSampler(
        n_samples=4096,
        n_chains=chains.shape[0],
        burn_in=0,
        discard=0,
        proposal="ham",
        beta=0.5,
        alpha=float(np.asarray(saved_sampler["alpha"])),
    )
    sampler_state = sampler.init(
        params,
        hamiltonian,
        model,
        key=jax.device_put(saved_sampler["key"]),
        eps1=1e-3,
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
        n_eloc=0,
    )

    state, stats, data = state.expect(data=True)

    weight = data["weight"]

    one_rdm = rdm1(state, data["x"], weight)
    gamma = np.sum(one_rdm, axis=0)
    gamma = 0.5 * (gamma + np.conjugate(gamma.T))
    occupation = np.linalg.eigvalsh(gamma)[::-1]

    density_corr = density_correlation(sector, data["x"], weight,)

    spin = spin_correlation(state, data["x"], weight)
    spin = np.real_if_close(spin)
    s2 = np.real_if_close(np.sum(spin)).item()

    orbital = np.arange(1, occupation.size + 1)
    log = npz.with_suffix(".log")

    with log.open("w", encoding="utf-8") as file:
        file.write(f"checkpoint = {npz}\n")
        file.write(f"fcidump = {fcidump}\n")
        file.write(f"active = {sector.nelec}e, {sector.norb}o\n")
        file.write(f"samples = {sampler.n_samples}\n\n")

        file.write("[energy]\n")
        file.write(f"value = {stats['energy']:.6f}\n")
        file.write(f"variance = {stats['eloc_var']:.6e}\n\n")

        file.write("\n[natural_occupations]\n")
        file.write("orbital occupation\n")
        np.savetxt(
            file,
            np.column_stack((orbital, occupation)),
            fmt=("%d", "%.6f"),
        )

        file.write("\n[density_correlation]\n")
        np.savetxt(file, density_corr, fmt="%.6f")

        file.write("\n[spin_correlation]\n")
        np.savetxt(file, spin, fmt="%.6f")

        file.write("\n[s2]\n")
        file.write(f"{s2:.6f}\n")


if __name__ == "__main__":
    main()
