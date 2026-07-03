from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import SBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


NAME = "H2O_ccpvdz_1.0re"
FCIDUMP = f"FCIDUMP/H2O_ccpvdz/{NAME}.FCIDUMP"
LOG = f"{NAME}.jsonl"
CKPT = f"{NAME}_{{step:05d}}.npz"

STEPS = 1000
CHECKPOINT_EVERY = 100


def configure() -> None:
    # Configure runtime.
    utils.batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)


def build(*, seed: int = 0) -> tuple[VMC, dict[str, object]]:
    # Load the FCIDUMP Hamiltonian.
    H = operator.Hamiltonian.load(FCIDUMP)
    sector = H.sector

    # Build a mean-field reference and sampled chains.
    ref_mat = utils.ref_init(sector, H.integrals, seed=seed)

    model = SBackflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(256,),
        ref_mat=jnp.asarray(ref_mat),
    )

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    chains = utils.chain_init(
        sector,
        ref_mat,
        n_chains=sampler.n_chains,
        seed=seed,
    )

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(seed),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    scale = optax.linear_schedule(
        init_value=0.0,
        end_value=-5.0e-2,
        transition_steps=100,
    )
    optimizer = psr(shift=1.0e-3, mu=0.95, scale=scale)

    vmc = VMC.init(state, optimizer)
    obs = {"s2": operator.S2(sector)}
    return vmc, obs


def main() -> None:
    configure()

    vmc, obs = build(seed=0)
    log = utils.Logger(file=LOG, every=10)

    vmc.run(
        STEPS,
        obs=obs,
        logger=log,
        profile=True,
        checkpoint=CKPT,
        checkpoint_every=CHECKPOINT_EVERY,
    )


if __name__ == "__main__":
    main()
