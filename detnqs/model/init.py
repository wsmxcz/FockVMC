from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def slater_reference(alpha: ArrayLike, beta: ArrayLike) -> np.ndarray:
    """Assemble occupied alpha/beta orbitals into a Slater reference.

    The orbitals are expressed in the Hamiltonian basis with shapes
    ``(norb, n_alpha)`` and ``(norb, n_beta)``. The result has shape
    ``(n_alpha + n_beta, 2 * norb)``.
    """
    alpha = np.asarray(alpha, dtype=np.float64, order="C")
    beta = np.asarray(beta, dtype=np.float64, order="C")

    if alpha.ndim != 2 or beta.ndim != 2 or alpha.shape[0] != beta.shape[0]:
        raise ValueError("alpha and beta must have shape (norb, n_elec)")

    norb = alpha.shape[0]
    n_alpha = alpha.shape[1]
    n_beta = beta.shape[1]
    ref_mat = np.zeros((n_alpha + n_beta, 2 * norb), dtype=np.float64)
    ref_mat[:n_alpha, :norb] = alpha.T
    ref_mat[n_alpha:, norb:] = beta.T
    return ref_mat
