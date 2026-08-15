from pathlib import Path

import jax
import optax

from fvmc import Hamiltonian, MCState, VMC
from fvmc.model import RBM
from fvmc.sampler import MCSampler
from fvmc.optimizer import sr
from fvmc.utils import Logger


def main() -> None:
    hamiltonian = Hamiltonian.load(next(Path.cwd().parent.rglob("H2.FCIDUMP")))
    sector = hamiltonian.sector
    model = RBM(norb=sector.norb, alpha=1)
    sampler = MCSampler(n_samples=128, n_chains=128, rank=None)
    chains = sector.random(sampler.n_chains, seed=0)

    state = MCState.init(
        model,
        hamiltonian,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
    )
    vmc = VMC.init(state, optax.chain(sr(), optax.scale_by_learning_rate(5.0e-2)))
    vmc.run(200, log=Logger(every=10))


if __name__ == "__main__":
    main()
