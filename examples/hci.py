from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import primme
from scipy.sparse.linalg import LinearOperator

import libdet
from pyscf import ao2mo, gto, scf


@dataclass(slots=True)
class State:
    dets: np.ndarray
    coeffs: np.ndarray
    energy: float
    diags: np.ndarray
    eps: float | None = None


def hf_det(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    nword = (norb + 63) // 64
    det = np.zeros((2, nword), dtype=np.uint64)

    for spin, nocc in enumerate(nelec):
        for p in range(nocc):
            det[spin, p // 64] |= np.uint64(1) << np.uint64(p % 64)

    return det


def davidson_primme(
    ham,
    dets: np.ndarray,
    guess: np.ndarray | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    max_space: int = 64,
    mode: str = "sparse",
):
    t0 = time.perf_counter()

    dets = libdet.to_dets(dets)
    hdiag = np.asarray(ham.diags(dets), dtype=np.float64).reshape(-1)
    n = hdiag.size

    if n == 1:
        return (
            float(hdiag[0]),
            np.array([1.0], dtype=np.float64),
            hdiag,
            0.0,
            0.0,
            time.perf_counter() - t0,
        )

    if guess is None:
        v0 = np.zeros(n, dtype=np.float64)
        v0[0] = 1.0
    else:
        v0 = np.asarray(guess, dtype=np.float64).reshape(-1).copy()
        v0 /= np.linalg.norm(v0)

    t_conns = time.perf_counter()

    if mode == "sparse":
        A = ham.matrix(dets)

    elif mode == "matvec":

        def matvec(x):
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            return np.asarray(ham.matvec(dets, x, kets=dets), dtype=np.float64)

        def matmat(X):
            X = np.asarray(X, dtype=np.float64)
            return np.asarray(ham.matvec(dets, X, kets=dets), dtype=np.float64)

        A = LinearOperator((n, n), matvec=matvec, matmat=matmat, dtype=np.float64)

    else:
        raise ValueError("mode must be 'sparse' or 'matvec'")

    t_conns = time.perf_counter() - t_conns

    # Diagonal inverse preconditioner.
    def precond(x):
        X = np.asarray(x, dtype=np.float64)
        is_vec = X.ndim == 1

        if is_vec:
            X = X[:, None]

        shifts = np.asarray(primme.get_eigsh_param("ShiftsForPreconditioner"))
        if shifts.size == 0:
            shifts = np.zeros(X.shape[1], dtype=np.float64)
        elif shifts.size == 1 and X.shape[1] > 1:
            shifts = np.full(X.shape[1], shifts[0], dtype=np.float64)

        Y = np.empty_like(X)
        for j in range(X.shape[1]):
            denom = hdiag - shifts[j]
            denom = np.where(np.abs(denom) < 1e-8, np.copysign(1e-8, denom), denom)
            Y[:, j] = X[:, j] / denom

        return Y[:, 0] if is_vec else Y

    OPinv = LinearOperator((n, n), matvec=precond, matmat=precond, dtype=np.float64)

    t_solve = time.perf_counter()
    w, v = primme.eigsh(
        A,
        k=1,
        which="SA",
        v0=v0[:, None],
        OPinv=OPinv,
        tol=tol,
        maxiter=max_iter,
        ncv=min(n, max(8, min(max_space, 24))),
        maxBlockSize=1,
        raise_for_unconverged=False,
    )
    t_solve = time.perf_counter() - t_solve

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64)

    if w.size == 0:
        coeffs = v0.copy()
        energy = float(np.dot(coeffs, A @ coeffs))
    else:
        coeffs = v[:, 0].copy()
        if coeffs[np.argmax(np.abs(coeffs))] < 0.0:
            coeffs = -coeffs
        coeffs /= np.linalg.norm(coeffs)
        energy = float(w[0])

    t_other = time.perf_counter() - t0 - t_conns - t_solve
    return energy, coeffs, hdiag, t_conns, t_solve, t_other


def hci_solve(
    ham,
    nelec: tuple[int, int],
    *,
    eps: float = 1e-4,
    max_cycle: int = 10,
    davidson_mode: str = "sparse",
) -> State:
    t_total = time.perf_counter()

    dets = hf_det(int(ham.norb), nelec)[None]
    coeffs = np.array([1.0], dtype=np.float64)
    diags = np.asarray(ham.diags(dets), dtype=np.float64).reshape(-1)
    energy = float(diags[0])

    header = (
        f"{'Iter':>4} | {'Ndet':>8} | {'Screen':>10} | "
        f"{'Conns':>11} | {'Solve':>10} | {'Other':>10} | {'Energy':>16}"
    )
    print(header)
    print("-" * len(header))

    for it in range(max_cycle):
        t_screen = time.perf_counter()
        cand = ham.expand(dets, eps, coeffs=coeffs, exclude=dets)
        t_screen = time.perf_counter() - t_screen

        if cand.shape[0] == 0:
            print(f"Converged at cycle {it}: no new determinants.")
            break

        n_old = len(dets)
        dets = np.ascontiguousarray(np.concatenate([dets, cand], axis=0), dtype=np.uint64)

        guess = np.zeros(len(dets), dtype=np.float64)
        guess[:n_old] = coeffs

        energy, coeffs, diags, t_conns, t_solve, t_other = davidson_primme(
            ham,
            dets,
            guess,
            mode=davidson_mode,
        )

        print(
            f"{it + 1:4d} | {len(dets):8d} | {t_screen:10.4f} | "
            f"{t_conns:11.4f} | {t_solve:10.4f} | {t_other:10.4f} | "
            f"{energy:16.10f}"
        )

    print("-" * len(header))
    print(f"Total: {time.perf_counter() - t_total:.4f} s")

    return State(dets=dets, coeffs=coeffs, energy=energy, diags=diags, eps=float(eps))


def semi_pt2(
    ham,
    state: State,
    *,
    eps1: float = 1e-4,
    eps2: float = 1e-6,
    counts: int = 16,
    n_rep: int = 4,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    dets = state.dets
    coeffs = state.coeffs
    energy = state.energy

    # Strong external amplitudes.
    prj = ham.project(
        None,
        dets,
        coeffs,
        eps=eps1,
        exclude=dets,
    )

    hpsi = np.asarray(prj.hpsi, dtype=np.float64)
    diags = np.asarray(prj.diags, dtype=np.float64)
    e2_det = float(np.sum((hpsi * hpsi) / (energy - diags)))

    # Sample the weak window eps2 <= |H_ai c_i| < eps1.
    sample = ham.sample_project(
        dets,
        coeffs,
        eps1,
        eps2,
        counts,
        exclude=dets,
        n_rep=n_rep,
        seed=seed,
    )

    rep_ptr = np.asarray(sample.rep_ptr, dtype=np.int64)
    diags = np.asarray(sample.diags, dtype=np.float64)
    hpsi_strong = np.asarray(sample.hpsi_strong, dtype=np.float64)
    hpsi_a = np.asarray(sample.hpsi_a, dtype=np.float64)
    hpsi_b = np.asarray(sample.hpsi_b, dtype=np.float64)

    corr = np.zeros(n_rep, dtype=np.float64)

    for r in range(n_rep):
        lo = int(rep_ptr[r])
        hi = int(rep_ptr[r + 1])

        if hi == lo:
            continue

        denom = energy - diags[lo:hi]
        s = hpsi_strong[lo:hi]
        wa = hpsi_a[lo:hi]
        wb = hpsi_b[lo:hi]

        corr[r] = np.sum((s * (wa + wb) + wa * wb) / denom)

    e2_stoch = float(np.mean(corr))
    err = 0.0 if n_rep == 1 else float(np.std(corr, ddof=1) / np.sqrt(n_rep))

    return e2_det + e2_stoch, e2_det, e2_stoch, err


mol = gto.M(
    atom="""
    O   0.00000000,  0.00000000,  0.00000000
    H   0.75700000,  0.00000000,  0.58590000
    H  -0.75700000,  0.00000000,  0.58590000
    """,
    # atom="""
    # N    0.53920000,  0.00000000,  0.00000000
    # N   -0.53920000,  0.00000000,  0.00000000
    # """,
    basis="cc-pvdz",
    unit="Angstrom",
    verbose=0,
)

mf = scf.RHF(mol).run()
norb = mf.mo_coeff.shape[1]

# Canonical MO integrals.
h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb), dtype=np.float64)

ham = libdet.Hamiltonian.rhf(h1e, eri, ecore=mol.energy_nuc())

state = hci_solve(
    ham,
    mol.nelec,
    eps=1e-4,
    max_cycle=10,
    davidson_mode="matvec",
)

e2_total, e2_det, e2_stoch, err = semi_pt2(
    ham,
    state,
    eps1=1e-6,
    eps2=1e-12,
    counts=16,
    n_rep=4,
    seed=0,
)

print()
print(f"SCF          : {mf.e_tot:16.12f}")
print(f"HCI var      : {state.energy:16.12f}  Ndet: {len(state.dets)}")
print(f"PT2 det      : {e2_det:16.12f}")
print(f"PT2 stoch    : {e2_stoch:16.12f} +/- {err:.12f}")
print(f"PT2 total    : {e2_total:16.12f} +/- {err:.12f}")
print(f"HCI + PT2    : {state.energy + e2_total:16.12f} +/- {err:.12f}")
