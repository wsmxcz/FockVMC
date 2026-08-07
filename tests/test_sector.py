import numpy as np

from detnqs.hilbert import DetSector
from detnqs.operator import number


def test_config() -> None:
    sector = DetSector(norb=5, nelec=4, spin=0)
    reference = sector.reference(3)
    random = sector.random(8, seed=4)

    assert reference.shape == (3, 2, 1)
    assert random.shape == (8, 2, 1)
    assert reference.dtype == np.uint64
    np.testing.assert_array_equal(number(random, spin=0), 2.0)
    np.testing.assert_array_equal(number(random, spin=1), 2.0)


def test_unique() -> None:
    sector = DetSector(norb=3, nelec=2, spin=0)
    basis = sector.enumerate()
    assert basis.shape == (9, 2, 1)

    x = np.concatenate((basis[:3], basis[1:3]), axis=0)
    unique, first, inverse = sector.unique(x)

    np.testing.assert_array_equal(unique, basis[:3])
    np.testing.assert_array_equal(first, [0, 1, 2])
    np.testing.assert_array_equal(unique[inverse], x)
