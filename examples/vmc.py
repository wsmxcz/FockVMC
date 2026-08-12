from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax

from detnqs import Hamiltonian, MCState, VMC
from detnqs.model import GBackflow, slater_reference
from detnqs.operator import S2
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler, sample_slater
from detnqs.utils import Logger, batch, precision


def main() -> None:
    batch.configure(
        forward_chunk=1024,
        backward_chunk=128,
        param_chunk=None,
        bucket_min=128,
    )
    precision.configure("single")

    name = "H2"
    seed = 0
    fcidump = next(Path.cwd().parent.rglob(f"{name}.FCIDUMP"), None)
    hamiltonian = Hamiltonian.load(fcidump)
    sector = hamiltonian.sector

    orbitals = np.eye(sector.norb)
    ref_mat = slater_reference(
        orbitals[:, :sector.n_alpha],
        orbitals[:, :sector.n_beta],
    )
    model = GBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(16,),
        ref_mat=ref_mat,
        init_scale=1e-3,
    )

    sampler = MCSampler(
        n_samples=128,
        n_chains=128,
        burnin_steps=0,
        discard_steps=0,
        proposal="ham",
        beta=0.5,
        alpha=None,
    )

    chains = sample_slater(
        sector,
        ref_mat,
        n=sampler.n_chains,
        seed=seed,
    )

    state = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(seed),
        eps1=0.0,
        eps2=0.0,
        eloc_sample=0,
    )

    optimizer = optax.chain(
        psr(shift=1.0e-3, mu=0.95),
        optax.scale_by_learning_rate(5.0e-2),
    )
    vmc = VMC.init(state, optimizer)
    obs = {"s2": S2(sector)}
    log = Logger(file=f"{name}.jsonl", every=10)

    vmc.run(
        5000,
        obs=obs,
        logger=log,
        profile=True,
        checkpoint=f"{name}_{{step:05d}}.npz",
        checkpoint_every=500,
    )


if __name__ == "__main__":
    main()
