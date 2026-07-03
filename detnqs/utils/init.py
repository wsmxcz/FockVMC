from __future__ import annotations

from typing import Any

import numpy as np
from pyscf import ao2mo


def ref_init(sector: Any, source: Any = None, *, seed: int = 0) -> np.ndarray:
    """Build an electron Slater reference in the active orbital basis.

    Returned shape is always:
        (n_alpha + n_beta, 2 * norb)

    Row convention:
        first  n_alpha rows: alpha occupied orbitals
        last   n_beta  rows: beta  occupied orbitals

    Supported source conventions:
        None:
            canonical spatial reference

        (h1, eri):
            active-space mean-field reference

        (alpha_coeff, beta_coeff):
            spin-block coefficient reference

        array(n_alpha + n_beta, 2 * norb):
            direct electron reference

        array(2 * norb, m):
            generalized spin-orbital coefficients

        array(norb, m):
            spatial coefficients shared by alpha and beta
    """
    norb = int(sector.norb)
    n_alpha = int(sector.n_alpha)
    n_beta = int(sector.n_beta)

    n_elec = n_alpha + n_beta
    n_sorb = 2 * norb

    # Canonical spatial reference.
    if source is None:
        coeff = np.eye(norb, dtype=np.float64)
        ref = np.zeros((n_elec, n_sorb), dtype=np.float64)
        ref[:n_alpha, :norb] = coeff[:, :n_alpha].T
        ref[n_alpha:, norb:] = coeff[:, :n_beta].T
        return np.ascontiguousarray(ref)

    # Tuple input selects mean-field or spin-block coefficients.
    if isinstance(source, tuple):
        a = np.asarray(source[0], dtype=np.float64)
        b = np.asarray(source[1], dtype=np.float64)

        # Mean-field reference from active integrals.
        if b.ndim != 2:
            h1 = a
            eri = ao2mo.restore(1, b, norb) if b.ndim != 4 else b

            rng = np.random.default_rng(seed)
            z = rng.normal(size=(norb, norb))

            f = h1 + 1.0e-10 * (z + z.T)
            f = 0.5 * (f + f.T)

            _, coeff = np.linalg.eigh(f)
            dm = None

            for _ in range(64):
                ca = coeff[:, :n_alpha]
                cb = coeff[:, :n_beta]

                dm_new = ca @ ca.T + cb @ cb.T
                dm = dm_new if dm is None else 0.65 * dm + 0.35 * dm_new

                j = np.einsum("pqrs,rs->pq", eri, dm, optimize=True)
                k = np.einsum("prqs,rs->pq", eri, dm, optimize=True)

                f = h1 + j - 0.5 * k
                f = 0.5 * (f + f.T)

                _, coeff = np.linalg.eigh(f)

            ref = np.zeros((n_elec, n_sorb), dtype=np.float64)
            ref[:n_alpha, :norb] = coeff[:, :n_alpha].T
            ref[n_alpha:, norb:] = coeff[:, :n_beta].T
            return np.ascontiguousarray(ref)

        # Spin-block coefficient reference.
        qa, _ = np.linalg.qr(a, mode="reduced")
        qb, _ = np.linalg.qr(b, mode="reduced")

        ref = np.zeros((n_elec, n_sorb), dtype=np.float64)
        ref[:n_alpha, :norb] = qa[:, :n_alpha].T
        ref[n_alpha:, norb:] = qb[:, :n_beta].T
        return np.ascontiguousarray(ref)

    arr = np.asarray(source, dtype=np.float64)

    # Direct electron reference.
    if arr.shape == (n_elec, n_sorb):
        return np.ascontiguousarray(arr)

    # General spin-orbital coefficients.
    if arr.ndim == 2 and arr.shape[0] == n_sorb:
        coeff, _ = np.linalg.qr(arr, mode="reduced")
        return np.ascontiguousarray(coeff[:, :n_elec].T)

    # Shared spatial coefficients.
    coeff, _ = np.linalg.qr(arr, mode="reduced")

    ref = np.zeros((n_elec, n_sorb), dtype=np.float64)
    ref[:n_alpha, :norb] = coeff[:, :n_alpha].T
    ref[n_alpha:, norb:] = coeff[:, :n_beta].T
    return np.ascontiguousarray(ref)


def chain_init(
    sector: Any,
    ref_mat: Any = None,
    *,
    n_chains: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """Sample determinant chains from a spin-block Slater reference."""
    norb = int(sector.norb)
    n_alpha = int(sector.n_alpha)
    n_beta = int(sector.n_beta)
    n_chains = int(n_chains)

    ref = ref_init(sector, ref_mat, seed=seed)
    chains = sector.zeros(n_chains)

    rng = np.random.default_rng(seed)
    batch = np.arange(n_chains, dtype=np.int64)

    alpha = ref[:n_alpha, :norb].T
    beta = ref[n_alpha:, norb:].T

    for spin, n_elec, coeff in ((0, n_alpha, alpha), (1, n_beta, beta)):
        if n_elec == 0:
            continue

        # Initialize the residual Slater subspace.
        basis = np.linalg.qr(coeff[:, :n_elec], mode="reduced")[0]
        basis = np.broadcast_to(basis, (n_chains, norb, n_elec)).copy()
        occ = np.empty((n_chains, n_elec), dtype=np.int64)

        for k in range(n_elec):
            # Sample from the residual projection kernel.
            prob = np.sum(basis * basis, axis=-1)

            if k > 0:
                prob[batch[:, None], occ[:, :k]] = 0.0

            prob /= prob.sum(axis=1, keepdims=True)
            u = rng.random(n_chains)
            occ[:, k] = (np.cumsum(prob, axis=1) < u[:, None]).sum(axis=1)

            if k + 1 == n_elec:
                break

            # Condition on the selected orbital.
            row = basis[batch, occ[:, k]].copy()
            piv = np.argmax(np.abs(row), axis=1)
            piv_val = row[batch, piv]
            col = basis[batch, :, piv].copy()

            basis -= (col / piv_val[:, None])[:, :, None] * row[:, None, :]
            basis[batch, :, piv] = basis[:, :, -1]
            basis = np.linalg.qr(basis[:, :, :-1], mode="reduced")[0]

        # Pack the sampled occupations.
        word = occ >> 6
        bit = (occ & 63).astype(np.uint64)

        np.bitwise_or.at(
            chains,
            (batch[:, None], np.full_like(word, spin), word),
            np.uint64(1) << bit,
        )

    return np.ascontiguousarray(chains)
