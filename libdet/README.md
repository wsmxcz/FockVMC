# libdet

`libdet` is a lightweight determinant-driven C++ kernel with Python bindings for quantum chemistry Hamiltonians.

It is designed for selected CI, Fock-space VMC, FCIQMC, deterministic PT2, and semi-stochastic PT2. The library provides fast row-local Hamiltonian primitives and leaves solver policy, wavefunction updates and MPI orchestration to Python or downstream codes.

- Determinants are stored as `(N, 2, nword)` unsigned 64-bit arrays: alpha words, then beta words.
- `dets` means a determinant batch with no left/right role.
- `bras` and `kets` always mean the two axes of `H[bras, kets]`.
- `coeffs` or `x` are always aligned with `kets`.
- C++ kernels are row-local and mergeable: candidate generation, projection, sparse blocks, and samples can be sharded and reduced outside the library.

## Basic use

```python
import numpy as np
import libdet

ham = libdet.Hamiltonian.rhf(h1, eri, ecore=ecore)

# Determinants are always shaped as (N, 2, nword), even when nword == 1.
diag = ham.diags(dets)

# Candidate generation from source kets.
cand = ham.expand(dets, eps=1e-6, coeffs=coeffs, exclude=dets)

# Projection computes H[bras, kets] @ coeffs.
proj = ham.project(cand, dets, coeffs)
score = np.abs(proj.hpsi)

# Exact sparse Hamiltonian block.
H = ham.matrix(dets)

# Exact matrix-vector or matrix-matrix product.
y = ham.matvec(dets, coeffs)
```

## Future direction

The public API should remain unified while internal models specialize by Hamiltonian family. Future paths may include spin-orbital UHF kernels, GUGA spin-adapted CSF kernels, k-point periodic systems, and relativistic spinor Hamiltonians. These should be implemented as specialized internal models behind the same row-local primitive interface, not as user-visible solver frameworks.
