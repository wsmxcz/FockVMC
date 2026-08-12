import numpy as np

from fvmc.hilbert import DetSector
from fvmc.operator import number


def test_packed() -> None:
    sector = DetSector(norb=65, nelec=6, spin=0)
    x = sector.random(16, seed=4)

    assert x.shape == (16, 2, 2)
    assert x.dtype == np.uint64
    np.testing.assert_array_equal(number(x, spin=0), 3)
    np.testing.assert_array_equal(number(x, spin=1), 3)

    full = DetSector(norb=65, nelec=130, spin=0).reference(1)
    np.testing.assert_array_equal(full[0, :, 1], 1)


def test_unique() -> None:
    sector = DetSector(norb=3, nelec=2, spin=0)
    basis = sector.enumerate()
    x = np.concatenate((basis[:3], basis[1:3]))
    unique, first, inverse = sector.unique(x)

    assert basis.shape == (9, 2, 1)
    np.testing.assert_array_equal(unique, basis[:3])
    np.testing.assert_array_equal(first, [0, 1, 2])
    np.testing.assert_array_equal(unique[inverse], x)
