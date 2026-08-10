"""Read a VMC checkpoint and analyze its wavefunction structure."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from detnqs import Hamiltonian, MCState
from detnqs.model import PBackflow
from detnqs.operator import rdm1, rdm2, spin_correlation
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
        n_samples=409600,
        n_chains=chains.shape[0],
        thermal_steps=4096,
        discard_steps=0,
        proposal="ham",
        blur=0.0,
        alpha=2.0,
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
        eloc_sample=0,
    )

    state, _, data = state.expect(data=True)

    probability = data["weight"]
    participation_ratio = 1.0 / np.sum(probability**2)
    sorted_probability = np.sort(probability)[::-1]

    one_rdm = rdm1(
        state,
        data["x"],
        probability,
    )
    gamma = np.sum(one_rdm, axis=0)
    gamma = 0.5 * (gamma + np.conjugate(gamma.T))
    occupation = np.linalg.eigvalsh(gamma)[::-1]

    two_rdm = rdm2(
        state,
        data["x"],
        probability,
    )
    density = np.einsum("spp->p", one_rdm)
    density_product = np.einsum("abijij->ij", two_rdm)
    density_product[np.diag_indices(sector.norb)] += density
    density_correlation = density_product - np.outer(density, density)
    density_correlation = 0.5 * (
        density_correlation + np.conjugate(density_correlation.T)
    )
    density_correlation = np.real_if_close(density_correlation)

    spin = spin_correlation(state, data["x"], probability)
    spin = np.real_if_close(spin)

    rank = np.arange(1, sorted_probability.size + 1)
    orbital = np.arange(1, occupation.size + 1)
    log = npz.with_suffix(".log")

    with log.open("w", encoding="utf-8") as file:
        file.write(f"checkpoint = {npz}\n")
        file.write(f"fcidump = {fcidump}\n")
        file.write(f"active = {sector.nelec}e, {sector.norb}o\n")
        file.write(f"samples = {sampler.n_samples}\n\n")

        file.write("[participation_ratio]\n")
        file.write(f"{participation_ratio:.12e}\n\n")

        file.write("[determinant_probability]\n")
        file.write("rank probability\n")
        np.savetxt(
            file,
            np.column_stack((rank, sorted_probability)),
            fmt=("%d", "%.12e"),
        )

        file.write("\n[natural_occupations]\n")
        file.write("orbital occupation\n")
        np.savetxt(
            file,
            np.column_stack((orbital, occupation)),
            fmt=("%d", "%.12e"),
        )

        file.write("\n[density_correlation]\n")
        np.savetxt(file, density_correlation, fmt="%.12e")

        file.write("\n[spin_correlation]\n")
        np.savetxt(file, spin, fmt="%.12e")


if __name__ == "__main__":
    main()
