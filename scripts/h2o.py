from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax

from detnqs import Hamiltonian, MCState, VMC
from detnqs.model import PBackflow, slater_reference
from detnqs.operator import S2
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler, sample_slater
from detnqs.utils import Logger, batch, precision


def main() -> None:
    batch.configure(
        forward_chunk=32768,
        backward_chunk=4096,
        param_chunk=None,
        bucket_min=4096,
    )
    precision.configure("double")

    root = Path(__file__).with_name("FCIDUMP") / "H2O_ccpvdz"
    points = ("1.0re", "1.5re", "2.0re", "2.5re", "3.0re")

    for point in points:
        name = f"H2O_ccpvdz_{point}"
        hamiltonian = Hamiltonian.load(root / f"{name}.FCIDUMP")
        sector = hamiltonian.sector

        orbitals = np.eye(sector.norb)
        ref_mat = slater_reference(
            orbitals[:, : sector.n_alpha],
            orbitals[:, : sector.n_beta],
        )
        model = PBackflow(
            norb=sector.norb,
            n_alpha=sector.n_alpha,
            n_beta=sector.n_beta,
            hidden=(256, 256),
            ref_mat=ref_mat,
            init_scale=1.0e-3,
        )

        sampler = MCSampler(
            n_samples=4096,
            n_chains=4096,
            thermal_steps=4096,
            discard_steps=16,
            proposal="ham",
            blur=0.5,
            alpha=None,
        )
        chains = sample_slater(
            sector,
            ref_mat,
            n=sampler.n_chains,
            seed=0,
        )
        state = MCState.init(
            model=model,
            hamiltonian=hamiltonian,
            sampler=sampler,
            chains=chains,
            key=jax.random.key(0),
            eps1=1.0e-3,
            eps2=1.0e-12,
            eloc_sample=32768,
        )
        optimizer = optax.chain(
            psr(shift=1.0e-3, mu=0.95),
            optax.scale_by_learning_rate(5.0e-2),
        )
        vmc = VMC.init(state, optimizer)

        print(f"scan point  : {point}")
        print(f"active      : ({sector.nelec}e, {sector.norb}o)")

        vmc.run(
            5000,
            obs={"s2": S2(sector)},
            logger=Logger(file=f"{name}.jsonl", every=10),
            profile=True,
            checkpoint=f"{name}_{{step:05d}}.npz",
            checkpoint_every=1000,
        )


if __name__ == "__main__":
    main()
