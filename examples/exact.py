from __future__ import annotations

import jax
import jax.numpy as jnp

from detnqs import operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.vstate import ExactState


FCIDUMP = "H2O.FCIDUMP"
STEPS = 500


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

    state = ExactState.init(model=model, H=H, key=jax.random.key(0))
    optimizer = psr(shift=1.0e-3, mu=0.95, scale=-5.0e-2)
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(every=10, verbose=2)
    vmc.run(STEPS, logger=log)


if __name__ == "__main__":
    main()
