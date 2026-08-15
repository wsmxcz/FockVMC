from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax

from fvmc import Hamiltonian, IRState, VMC
from fvmc.model import PBackflow, slater_reference
from fvmc.operator import S2
from fvmc.optimizer import psr
from fvmc.sampler import HamSampler, sample_slater
from fvmc.utils import Logger, batch, precision


def main() -> None:
    precision.configure("double")
    batch.configure(
        forward_chunk=32768,
        backward_chunk=4096,
        param_chunk=None,
        bucket_min=4096,
    )

    name = "fe2s2"
    seed = 0
    fcidump = next(Path.cwd().rglob(f"{name}.FCIDUMP"))
    hamiltonian = Hamiltonian.load(fcidump)
    sector = hamiltonian.sector

    fe1 = np.arange(2, 7)
    fe2 = np.arange(13, 18)
    closed = np.setdiff1d(np.arange(sector.norb), np.concatenate((fe1, fe2)))

    occ_a = np.sort(np.concatenate((closed, fe1)))
    occ_b = np.sort(np.concatenate((closed, fe2)))
    assert len(occ_a) == sector.n_alpha and len(occ_b) == sector.n_beta
    orbitals = np.eye(sector.norb)
    ref_mat = slater_reference(orbitals[:, occ_a], orbitals[:, occ_b])

    model = PBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(256, 256),
        ref_mat=ref_mat,
        init_scale=1.0e-3,
    )

    sampler = HamSampler(
        n_samples=4096,
        n_chains=4096,
        thermal_steps=4096,
        discard_steps=16,
        eps1=1.0e-3,
    )
    chains = sample_slater(
        sector,
        ref_mat,
        n=sampler.n_chains,
        seed=seed,
    )

    state = IRState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(seed),
        alpha=None,
        beta=0.5,
        eps2=1.0e-12,
        n_eloc=1024,
    )
    optimizer = optax.chain(
        psr(shift=1.0e-3, mu=0.95),
        optax.scale_by_learning_rate(5.0e-2),
    )
    vmc = VMC.init(state, optimizer)

    print(f"active      : ({sector.nelec}e, {sector.norb}o)")
    print("reference   : Fe1 up, Fe2 down")

    vmc.run(
        5000,
        obs={"s2": S2(sector)},
        log=Logger(file=f"{name}.jsonl", every=100),
        profile=True,
        checkpoint=f"{name}_{{step:05d}}.npz",
        checkpoint_every=5000,
    )


if __name__ == "__main__":
    main()
