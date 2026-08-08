from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import optax

from detnqs import Hamiltonian, SelectedState, VMC
from detnqs.model import Backflow, slater_reference
from detnqs.utils import Logger, batch, precision
from detnqs.vstate import topk_selector


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

    state = SelectedState.init(
        model=model,
        hamiltonian=hamiltonian,
        basis=sector.reference(1),
        key=jax.random.key(0),
    )

    vmc = VMC.init(state, optax.adamw(1.0e-3), geometry=False)
    log = Logger(every=1, keys=("step", "outer", "energy", "eloc_var", "n_basis"))

    for outer in range(2):
        vmc.state = vmc.state.evolve(topk_selector(k=128), eps=1.0e-6)
        rec = vmc.run(50)
        rec["outer"] = float(outer + 1)
        rec["n_basis"] = float(vmc.state.n_basis)
        log.add(rec)


if __name__ == "__main__":
    main()
