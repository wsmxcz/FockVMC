from __future__ import annotations

from pathlib import Path

import jax
import optax

from detnqs import Hamiltonian, MCState, VMC
from detnqs.model import PBackflow, slater_reference
from detnqs.operator import S2
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler, init_chains
from detnqs.utils import Logger, batch, precision


def main() -> None:
    batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    precision.configure("single")
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_log_compiles", False)

    name = "N2"
    seed = 0
    path = Path(__file__).parents[1] / "scripts" / "FCIDUMP" / f"{name}.FCIDUMP"
    hamiltonian = Hamiltonian.load(path)
    sector = hamiltonian.sector

    ref_mat = slater_reference(sector, hamiltonian.integrals, seed=seed)
    model = PBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(64,),
        ref_mat=ref_mat,
        init_scale=1e-3,
    )

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=256,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    chains = init_chains(
        sector,
        ref_mat,
        n_chains=sampler.n_chains,
        seed=seed,
    )

    state = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(seed),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
    )

    optimizer = optax.chain(
        psr(shift=1.0e-3, mu=0.95),
        optax.scale_by_learning_rate(5.0e-2),
    )
    vmc = VMC.init(state, optimizer)
    obs = {"s2": S2(sector)}
    log = Logger(file=f"{name}.jsonl", every=10)

    vmc.run(
        1000,
        obs=obs,
        logger=log,
        profile=True,
        checkpoint=f"{name}_{{step:05d}}.npz",
        checkpoint_every=100,
    )


if __name__ == "__main__":
    main()
