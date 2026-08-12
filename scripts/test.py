"""Quick end-to-end check for DetNQS."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np
import optax

from detnqs import ExactState, Hamiltonian, MCState, VMC, psr
from detnqs.model import RBM
from detnqs.sampler import MCSampler
from detnqs.utils import batch, precision


def main() -> None:
    precision.configure("single")
    batch.configure(
        forward_chunk=128,
        backward_chunk=128,
        param_chunk=None,
        bucket_min=128,
    )
    fcidump = next(Path.cwd().rglob("H2.FCIDUMP"))
    hamiltonian = Hamiltonian.load(fcidump)
    sector = hamiltonian.sector
    model = RBM(norb=sector.norb, alpha=1)

    sampler = MCSampler(
        n_samples=128,
        n_chains=128,
        burnin_steps=4,
        proposal="ham",
        beta=0.5,
        alpha=None,
    )
    state = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=sector.random(sampler.n_chains, seed=0),
        key=jax.random.key(0),
        eps1=0.0,
        eps2=0.0,
        eloc_sample=0,
    )

    exact = ExactState.init(
        model=model,
        hamiltonian=hamiltonian,
        key=jax.random.key(1),
    ).replace(params=state.params)
    _, exact_stats = exact.expect()

    state, mc_stats = state.expect()
    assert np.isfinite(mc_stats["energy"])
    assert abs(mc_stats["energy"] - exact_stats["energy"]) < 0.25

    optimizer = optax.chain(
        psr(shift=1.0e-3, mu=0.0),
        optax.scale_by_learning_rate(5.0e-2),
    )
    vmc = VMC.init(state, optimizer)
    record = vmc.step()
    assert np.isfinite(record["energy"])

    with TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "vmc.npz"
        vmc.save(checkpoint)
        step = vmc.step_count
        vmc.load(checkpoint)
        assert vmc.step_count == step

    exact_ground = float(
        np.linalg.eigvalsh(hamiltonian.matrix(sector.enumerate()).toarray())[0]
    )
    print(f"exact ground : {exact_ground:.6f}")
    print(f"exact model  : {exact_stats['energy']:.6f}")
    print(f"MC model     : {mc_stats['energy']:.6f}")
    print("PSR step     : ok")
    print("checkpoint   : ok")


if __name__ == "__main__":
    main()
