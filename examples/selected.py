from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from detnqs import operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.vstate import SelectedState, topk_selector


FCIDUMP = "H2O.FCIDUMP"
OUTER_STEPS = 5
INNER_STEPS = 100


def configure() -> None:
    # Configure runtime.
    utils.batch.configure(
        forward_chunk=32768,
        backward_chunk=32768,
        param_chunk=32768,
        bucket_min=1024,
    )
    utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)


def main() -> None:
    configure()

    # Load a small FCIDUMP Hamiltonian.
    H = operator.Hamiltonian.load(FCIDUMP)
    sector = H.sector

    ref_mat = utils.ref_init(sector, H.integrals)
    model = Backflow(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(64,),
        ref_mat=jnp.asarray(ref_mat),
    )

    state = SelectedState.init(
        model=model,
        H=H,
        basis=sector.reference(1),
        key=jax.random.key(0),
    )

    vmc = VMC.init(state, optax.adamw(1.0e-3), geometry=False)
    log = utils.Logger(every=1, keys=("step", "outer", "energy", "eloc_var", "n_basis"))

    # Alternate basis growth and projected optimization.
    for outer in range(OUTER_STEPS):
        vmc.state = vmc.state.evolve(topk_selector(k=128), eps=1.0e-6)
        rec = vmc.run(INNER_STEPS)
        rec["outer"] = float(outer + 1)
        rec["n_basis"] = float(vmc.state.n_basis)
        log.add(rec)


if __name__ == "__main__":
    main()
