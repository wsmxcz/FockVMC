from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from pyscf import ao2mo, gto, scf

from fvmc import Hamiltonian
from fvmc.hilbert import DetSector
from fvmc.utils import Logger


@dataclass(slots=True)
class HCIState:
    """Selected-CI wavefunction data used by variational and PT2 stages."""

    basis: np.ndarray
    coeff: np.ndarray
    energy: float


def ground(
    hamiltonian: Hamiltonian,
    basis: np.ndarray,
    initial: np.ndarray | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    max_space: int = 64,
    mode: str = "sparse",
) -> tuple[float, np.ndarray, float, float, float]:
    """Solve the variational problem in a fixed determinant basis."""
    start = time.perf_counter()

    diag = hamiltonian.diag(basis).reshape(-1)
    n = diag.size

    if n == 1:
        return (
            float(diag[0]),
            np.array([1.0], dtype=np.float64),
            0.0,
            0.0,
            time.perf_counter() - start,
        )

    if initial is None:
        vector = np.zeros(n, dtype=np.float64)
        vector[0] = 1.0
    else:
        vector = np.asarray(initial, dtype=np.float64).reshape(-1).copy()
        vector /= np.linalg.norm(vector)

    conn_time = time.perf_counter()

    if mode == "sparse":
        matrix = hamiltonian.matrix(basis)

    elif mode == "matvec":

        def matvec(x: np.ndarray) -> np.ndarray:
            return hamiltonian.matvec(
                basis,
                np.asarray(x, dtype=np.float64).reshape(-1),
                kets=basis,
            )

        def matmat(x: np.ndarray) -> np.ndarray:
            return hamiltonian.matvec(
                basis,
                np.asarray(x, dtype=np.float64),
                kets=basis,
            )

        matrix = LinearOperator(
            (n, n),
            matvec=matvec,
            matmat=matmat,
            dtype=np.float64,
        )

    else:
        raise ValueError("mode must be 'sparse' or 'matvec'")

    conn_time = time.perf_counter() - conn_time

    solve_time = time.perf_counter()
    if n == 2:
        values, vectors = np.linalg.eigh(hamiltonian.matrix(basis).toarray())
        values = values[:1]
        vectors = vectors[:, :1]
    else:
        values, vectors = eigsh(
            matrix,
            k=1,
            which="SA",
            v0=vector,
            tol=tol,
            maxiter=max_iter,
            ncv=min(n, max(3, min(max_space, 24))),
        )
    solve_time = time.perf_counter() - solve_time

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float64)

    coeff = vectors[:, 0].copy()
    if coeff[np.argmax(np.abs(coeff))] < 0.0:
        coeff = -coeff
    coeff /= np.linalg.norm(coeff)
    energy = float(values[0])

    other_time = time.perf_counter() - start - conn_time - solve_time
    return energy, coeff, conn_time, solve_time, other_time


def select(
    hamiltonian: Hamiltonian,
    *,
    eps: float = 1e-4,
    max_cycle: int = 10,
    mode: str = "sparse",
) -> HCIState:
    """Grow a heat-bath selected determinant basis and diagonalize in it."""
    start = time.perf_counter()

    basis = hamiltonian.sector.reference(1)
    coeff = np.array([1.0], dtype=np.float64)
    energy = float(hamiltonian.diag(basis).reshape(-1)[0])

    log = Logger(
        every=1,
        keys=[
            "step",
            "n_det",
            "n_new",
            "energy",
            "time_screen",
            "time_conns",
            "time_solve",
            "time_other",
        ],
    )

    for it in range(max_cycle):
        screen_time = time.perf_counter()
        candidates = hamiltonian.expand(basis, eps, scale=coeff, exclude=basis)
        screen_time = time.perf_counter() - screen_time

        if candidates.shape[0] == 0:
            print(f"Converged at cycle {it}: no new configurations.")
            break

        old_size = len(basis)
        basis = np.ascontiguousarray(
            np.concatenate([basis, candidates], axis=0),
            dtype=np.uint64,
        )

        initial = np.zeros(len(basis), dtype=np.float64)
        initial[:old_size] = coeff

        energy, coeff, conn_time, solve_time, other_time = ground(
            hamiltonian,
            basis,
            initial,
            mode=mode,
        )

        log(
            {
                "step": it + 1,
                "n_det": len(basis),
                "n_new": candidates.shape[0],
                "energy": energy,
                "time_screen": screen_time,
                "time_conns": conn_time,
                "time_solve": solve_time,
                "time_other": other_time,
            },
        )

    print(f"Total time: {time.perf_counter() - start:.3f}s")

    return HCIState(
        basis=basis,
        coeff=coeff,
        energy=energy,
    )


def pt2(
    hamiltonian: Hamiltonian,
    state: HCIState,
    *,
    eps1: float = 1e-6,
    eps2: float = 1e-12,
    counts: int = 16,
    n_rep: int = 4,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    basis = state.basis
    coeff = state.coeff
    energy = state.energy

    # Deterministic contribution above the upper cutoff.
    projection = hamiltonian.project(
        None,
        basis,
        coeff,
        eps=eps1,
        exclude=basis,
    )

    hpsi = np.asarray(projection.hpsi)
    diag = np.asarray(projection.diag)
    det_pt2 = float(np.sum((hpsi * hpsi) / (energy - diag)))

    # Stochastic contribution inside the remaining cutoff window.
    rng = np.random.default_rng(seed)
    corr = np.zeros(n_rep, dtype=np.float64)
    sample_count = np.full((2, len(basis)), counts, dtype=np.int64)

    for r in range(n_rep):
        weak = hamiltonian.sample_project(
            basis,
            coeff,
            sample_count,
            eps1=eps1,
            eps2=eps2,
            exclude=basis,
            seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
        )

        sample_bra = np.asarray(weak.bra)
        weak_hpsi = np.asarray(weak.hpsi)
        if sample_bra.shape[0] == 0:
            continue

        strong = hamiltonian.project(sample_bra, basis, coeff, eps=eps1)
        strong_hpsi = np.asarray(strong.hpsi)
        denom = energy - np.asarray(strong.diag)
        weak_a, weak_b = weak_hpsi

        corr[r] = np.sum(
            (strong_hpsi * (weak_a + weak_b) + weak_a * weak_b) / denom
        )

    sample_pt2 = float(np.mean(corr))
    err = 0.0 if n_rep == 1 else float(np.std(corr, ddof=1) / np.sqrt(n_rep))

    return det_pt2 + sample_pt2, det_pt2, sample_pt2, err



mol = gto.M(
    atom="""
    O   0.00000000,  0.00000000,  0.00000000
    H   0.75700000,  0.00000000,  0.58590000
    H  -0.75700000,  0.00000000,  0.58590000
    """,
    basis="6-31g",
    unit="Angstrom",
    spin=0,
    verbose=0,
)

mf = scf.ROHF(mol).run()
norb = mf.mo_coeff.shape[1]
n_alpha, n_beta = mol.nelec
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb)

sector = DetSector(norb, nelec=n_alpha + n_beta, spin=mol.spin)
hamiltonian = Hamiltonian(sector, h1, eri, ecore=mol.energy_nuc())

state = select(hamiltonian)
correction, deterministic, stochastic, error = pt2(hamiltonian, state)

print()
print(f"SCF          : {mf.e_tot:.6f}")
print(f"HCI var      : {state.energy:.6f}  Ndet: {len(state.basis)}")
print(f"PT2 det      : {deterministic:.6f}")
print(f"PT2 stoch    : {stochastic:.6f} +/- {error:.6e}")
print(f"PT2 total    : {correction:.6f} +/- {error:.6e}")
print(f"HCI + PT2    : {state.energy + correction:.6f} +/- {error:.6e}")
