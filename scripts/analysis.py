"""Read a VMC checkpoint and analyze its wavefunction structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fvmc import Hamiltonian, IRState
from fvmc.model import PBackflow
from fvmc.operator import density_correlation, rdm1, spin_correlation
from fvmc.sampler import ChainState, HamSampler, MCSampler
from fvmc.utils import batch, checkpoint, precision, stats


def main() -> None:
    npz = next(Path.cwd().rglob("H2O_ccpvdz_1.0re_05000.npz"))
    fcidump = next(Path.cwd().rglob("H2O_ccpvdz_1.0re.FCIDUMP"))

    precision.configure("double")
    batch.configure(
        forward_chunk=64,
        backward_chunk=4096,
        param_chunk=None,
        bucket_min=4096,
    )

    # Load the frozen wave function and its final chains.
    saved_state = checkpoint.load(npz, key="state")
    saved_chain = saved_state["sampler_state"]
    params = saved_state["params"]

    hamiltonian = Hamiltonian.load(fcidump)
    sector = hamiltonian.sector

    hidden = tuple(
        params[key]["bias"].size
        for key in sorted(
            (key for key in params if key.startswith("hidden_")),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
    )

    model = PBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=hidden,
    )

    # Run continuous Born chains from the saved configurations.
    mc_sampler = MCSampler(
        n_samples=64,
        n_chains=64,
        thermal_steps=4096,
        discard_steps=1,
        rank={1: 0.5, 2: 0.5},
    )
    mc_chain = mc_sampler.init(
        params,
        model,
        sector,
        chains=saved_chain["x"][: mc_sampler.n_chains],
        key=saved_chain["key"],
        alpha=2.0,
    )
    log_trace = np.empty((16384, mc_sampler.n_chains), dtype=np.float64)
    accepted = 0
    for step in range(log_trace.shape[0]):
        old = mc_chain.x
        mc_chain, _ = mc_sampler.step(
            params,
            model,
            sector,
            mc_chain,
            alpha=2.0,
        )
        log_trace[step] = 2.0 * mc_chain.logabs
        accepted += np.any(mc_chain.x != old, axis=(1, 2)).sum()

    # Compute standard diagnostics from the log-probability chains.
    log_trace = np.ascontiguousarray(log_trace.T)
    tau = stats.int_time(stats.autocorr(log_trace))
    mcmc_ess = log_trace.size / tau
    mcmc_frac = 1.0 / tau
    acceptance_rate = accepted / log_trace.size
    rhat = stats.rhat(log_trace)

    # Evaluate the wave-function observables.
    batch.configure(
        forward_chunk=32768,
        backward_chunk=4096,
        param_chunk=None,
        bucket_min=4096,
    )
    chains = saved_chain["x"]
    sampler = HamSampler(
        n_samples=4096,
        n_chains=chains.shape[0],
        thermal_steps=0,
        eps1=1.0e-3,
    )
    state = IRState(
        model=model,
        params=params,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chain=ChainState(
            key=saved_chain["key"],
            x=chains,
            logabs=saved_chain["logabs"],
        ),
        alpha=None,
        alpha_value=saved_chain["alpha"].item(),
        beta=0.5,
        eps1=0.0,
        eps2=0.0,
        n_eloc=0,
    )

    state, rec, data = state.expect(data=True)
    weight = data["weight"]

    gamma = rdm1(state, data["x"], weight).sum(axis=0)
    gamma = 0.5 * (gamma + gamma.T.conj())
    occupation = np.linalg.eigvalsh(gamma)[::-1]

    density_corr = density_correlation(sector, data["x"], weight)

    spin = spin_correlation(state, data["x"], weight)
    spin = np.real_if_close(spin)
    s2 = np.real_if_close(spin.sum()).item()

    # Write all results to one log file.
    with npz.with_suffix(".log").open("w", encoding="utf-8") as file:
        file.write(f"checkpoint = {npz}\n")
        file.write(f"fcidump = {fcidump}\n")
        file.write(f"active = {sector.nelec}e, {sector.norb}o\n")
        file.write(f"samples = {sampler.n_samples}\n\n")

        file.write("[energy]\n")
        file.write(f"value = {rec['energy']:.6f}\n")
        file.write(f"variance = {rec['eloc_var']:.6e}\n\n")

        file.write("[mcmc]\n")
        file.write(f"tau = {tau:.6f}\n")
        file.write(f"mcmc_ess = {mcmc_ess:.6f}\n")
        file.write(f"mcmc_frac = {mcmc_frac:.6e}\n")
        file.write(f"acceptance_rate = {acceptance_rate:.6f}\n")
        file.write(f"rhat = {rhat:.6f}\n")

        file.write("\n[natural_occupations]\n")
        file.write("orbital occupation\n")
        np.savetxt(
            file,
            np.column_stack((np.arange(1, occupation.size + 1), occupation)),
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
