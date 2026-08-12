import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fvmc.hilbert import DetSector
from fvmc.model import (
    Backflow,
    GBackflow,
    PBackflow,
    RBM,
    SBackflow,
    slater_reference,
)


def test_rbm() -> None:
    x = DetSector(norb=2, nelec=2, spin=0).enumerate()
    model = RBM(norb=2, alpha=1)
    params = model.init(jax.random.key(0), x[:1])["params"]
    value = model.logpsi(params, x)
    grad = jax.grad(lambda p: jnp.mean(model.logabs(p, x)))(params)

    assert value.shape == (len(x),)
    assert np.isfinite(value).all()
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(grad))


@pytest.mark.parametrize("kind", (Backflow, GBackflow, SBackflow, PBackflow))
def test_backflow(kind) -> None:
    sector = DetSector(norb=3, nelec=4, spin=0)
    orbitals = np.eye(sector.norb)
    ref = slater_reference(
        orbitals[:, : sector.n_alpha],
        orbitals[:, : sector.n_beta],
    )
    model = kind(
        norb=sector.norb,
        n_alpha=sector.n_alpha,
        n_beta=sector.n_beta,
        hidden=(),
        ref_mat=ref,
        init_scale=0.0,
    )
    x = sector.reference(1)
    params = model.init(jax.random.key(1), x)["params"]
    sign, logabs = model.apply(params, x)

    np.testing.assert_allclose(np.abs(sign), 1.0)
    np.testing.assert_allclose(logabs, 0.0, atol=1.0e-12)
