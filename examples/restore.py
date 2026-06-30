from __future__ import annotations

import jax

from detnqs import operator, utils
from detnqs.driver import VMC
from detnqs.model import GBackflow
from detnqs.optimizer import psr
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
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    target_steps = 500
    checkpoint_every = 100
    optimizer = psr(shift=1.0e-3, mu=0.95, scale=-5.0e-2)

    vmc = VMC.init(state, optimizer).load("N2_00100.npz")
    log = utils.Logger(file="N2.jsonl", every=10, append=True)
    obs = {"s2": operator.S2(sector)}

    while vmc.step_count < target_steps:
        rec = vmc.step(obs=obs, profile=True)
        log.add(rec)

        step = int(rec["step"])
        if step % checkpoint_every == 0 or step == target_steps:
            vmc.save(f"N2_{step:05d}.npz")


if __name__ == "__main__":
    main()
