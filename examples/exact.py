from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax

from detnqs import ExactState, Hamiltonian, VMC
from detnqs.model import Backflow, slater_reference
from detnqs.optimizer import psr
from detnqs.utils import Logger, batch, precision


def main() -> None:
    batch.configure(
        forward_chunk=128,
        backward_chunk=128,
        param_chunk=None,
        bucket_min=128,
    )
    precision.configure("single")

    path = Path(__file__).parents[1] / "scripts" / "FCIDUMP" / "H2.FCIDUMP"
    hamiltonian = Hamiltonian.load(path)
    sector = hamiltonian.sector

    orbitals = np.eye(sector.norb)
    ref_mat = slater_reference(
        orbitals[:, :sector.n_alpha],
        orbitals[:, :sector.n_beta],
    )
    model = Backflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(64,),
        ref_mat=ref_mat,
    )

    state = ExactState.init(
        model=model,
        hamiltonian=hamiltonian,
        key=jax.random.key(0),
    )
    optimizer = optax.chain(
        psr(shift=1.0e-3, mu=0.95),
        optax.scale_by_learning_rate(5.0e-2),
    )
    vmc = VMC.init(state, optimizer)

    log = Logger(every=10, verbose=2)
    vmc.run(100, logger=log)


if __name__ == "__main__":
    main()
