from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import primme
from scipy.sparse.linalg import LinearOperator
from pyscf import ao2mo, gto, scf

from detnqs import hilbert, operator, utils


@dataclass(slots=True)
class HCIState:
    """Selected-CI wavefunction data used by variational and PT2 stages."""

    dets: np.ndarray
    coeffs: np.ndarray
    energy: float
    diags: np.ndarray
    eps: float | None = None


def davidson_primme(
    H: operator.Hamiltonian,
    dets: np.ndarray,
    guess: np.ndarray | None = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    max_space: int = 64,
    mode: str = "sparse",
) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """Solve the variational problem in a fixed determinant basis."""
    t0 = time.perf_counter()

    hdiag = H.diag(dets).reshape(-1)
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
        A = H.matrix(dets)

    elif mode == "matvec":

        def matvec(x: np.ndarray) -> np.ndarray:
            return H.matvec(
                dets,
                np.asarray(x, dtype=np.float64).reshape(-1),
                kets=dets,
            )

        def matmat(X: np.ndarray) -> np.ndarray:
            return H.matvec(dets, np.asarray(X, dtype=np.float64), kets=dets)

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
    H: operator.Hamiltonian,
    *,
    eps: float = 1e-4,
    max_cycle: int = 10,
    davidson_mode: str = "sparse",
) -> HCIState:
    """Grow a heat-bath selected determinant basis and diagonalize in it."""
    t_total = time.perf_counter()

    dets = H.sector.reference(1)
    coeffs = np.array([1.0], dtype=np.float64)
    diags = H.diag(dets).reshape(-1)
    energy = float(diags[0])

    log = utils.Logger(
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
        # Select new external configurations above the cutoff.
        t_screen = time.perf_counter()
        cand = H.expand(dets, eps, scale=coeffs, exclude=dets)
        t_screen = time.perf_counter() - t_screen

        if cand.shape[0] == 0:
            print(f"Converged at cycle {it}: no new determinants.")
            break

        n_old = len(dets)
        dets = np.ascontiguousarray(
            np.concatenate([dets, cand], axis=0),
            dtype=np.uint64,
        )

        guess = np.zeros(len(dets), dtype=np.float64)
        guess[:n_old] = coeffs

        energy, coeffs, diags, t_conns, t_solve, t_other = davidson_primme(
            H,
            dets,
            guess,
            mode=davidson_mode,
        )

        log.add(
            it + 1,
            {
                "n_det": len(dets),
                "n_new": cand.shape[0],
                "energy": energy,
                "time_screen": t_screen,
                "time_conns": t_conns,
                "time_solve": t_solve,
                "time_other": t_other,
            },
        )

    print(f"Total time: {time.perf_counter() - t_total:.2f}s")

    return HCIState(
        dets=dets,
        coeffs=coeffs,
        energy=energy,
        diags=diags,
        eps=float(eps),
    )


def semi_pt2(
    H: operator.Hamiltonian,
    state: HCIState,
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

    # Deterministic contribution above the upper cutoff.
    prj = H.project(
        None,
        dets,
        coeffs,
        eps=eps1,
        exclude=dets,
    )

    hpsi = np.asarray(prj.hpsi)
    diags = np.asarray(prj.diag)
    e2_det = float(np.sum((hpsi * hpsi) / (energy - diags)))
    del prj, hpsi, diags

    # Stochastic contribution inside the remaining cutoff window.
    rng = np.random.default_rng(seed)
    corr = np.zeros(n_rep, dtype=np.float64)
    counts_arr = np.full((2, len(dets)), counts, dtype=np.int64)

    for r in range(n_rep):
        weak = H.sample_project(
            dets,
            coeffs,
            counts_arr,
            eps1=eps1,
            eps2=eps2,
            exclude=dets,
            seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
        )

        sample_bra = np.asarray(weak.bra)
        weak_hpsi = np.asarray(weak.hpsi)
        if sample_bra.shape[0] == 0:
            continue

        strong = H.project(sample_bra, dets, coeffs, eps=eps1)
        strong_pair = np.asarray(strong.hpsi)
        denom = energy - np.asarray(strong.diag)
        weak_a, weak_b = weak_hpsi

        corr[r] = np.sum(
            (strong_pair * (weak_a + weak_b) + weak_a * weak_b) / denom
        )

    e2_stoch = float(np.mean(corr))
    err = 0.0 if n_rep == 1 else float(np.std(corr, ddof=1) / np.sqrt(n_rep))

    return e2_det + e2_stoch, e2_det, e2_stoch, err


def main():
    # Build molecule.
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

    h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
    eri = np.asarray(
        ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb),
        dtype=np.float64,
    )

    # Build Hamiltonian.
    sector = hilbert.SpinSector(norb, nelec=n_alpha + n_beta, spin=mol.spin)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    # Run variational stage.
    state = hci_solve(
        H,
        eps=1e-4,
        max_cycle=10,
        davidson_mode="sparse",
    )

    # Run correction stage.
    e2_total, e2_det, e2_stoch, err = semi_pt2(
        H,
        state,
        eps1=1e-6,
        eps2=1e-6,
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


if __name__ == "__main__":
    main()
