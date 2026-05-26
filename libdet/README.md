# libdet

`libdet` is a lightweight C++ kernel with Python bindings for determinant-space quantum chemistry Hamiltonians.

It provides row-local Hamiltonian primitives for selected CI, Fock-space VMC, FCIQMC-style workflows, deterministic PT2, and semi-stochastic PT2. Solver policy, wavefunction updates, batching, MPI orchestration, and optimization are left to Python or downstream codes.

## Conventions

- Determinants are stored as `(N, 2, nword)` unsigned 64-bit arrays.
  Axis 1 stores alpha words first, then beta words.
- `dets` means a determinant batch with no left/right role.
- `bras` and `kets` are the two axes of `H[bras, kets]`.
- `coeffs` and `x` are always aligned with `kets`.
- Off-diagonal Hamiltonian connections are single and double excitations.
- Kernels are row-local and mergeable: rows may be sharded, computed independently, and reduced outside the library.

## Design

`libdet` uses two complementary kernels.

### Determinant-driven finite-space search

Used when both sides of the matrix block are known:

- `matrix(bras, kets)`
- `matvec(bras, kets, x)`
- `matmat(bras, kets, X)`
- `project(bras, kets, coeffs)`

The ket space is indexed once, then each bra searches only connected kets inside that finite space.

### Excitation-driven external generation

Used when the target space is not known in advance:

- `expand(kets, eps, coeffs, exclude)`
- `edges(dets, eps)`
- `degrees(dets, eps)`
- `sample_edges(dets, counts, eps1, eps2)`
- `sample_shell(...)`

The source determinant is expanded by single and double excitations. Screened double excitations use a heat-bath table ordered by integral magnitude. Pure degree queries do not materialize target determinants.

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
Y = ham.matvec(dets, X)
```

## Row connectivity

```python
# Screened external row graph.
graph = ham.edges(dets, eps=1e-6)

row_ptr = graph.row_ptr
col = graph.col
h = graph.h
col_dets = graph.col_dets

# Lightweight row statistics without materializing connected determinants.
deg = ham.degrees(dets, eps=1e-6)
row_nnz = deg.row_nnz
row_weight = deg.row_weight
```

`edges` returns a sparse row graph plus a determinant pool. `col[p]` indexes `graph.col_dets`; `h[p]` is the corresponding Hamiltonian matrix element.

## Stochastic edge sampling

```python
# Window: eps2 <= |H_ai| < eps1.
sample = ham.sample_edges(
    dets,
    counts=64,
    eps1=1e-3,
    eps2=1e-6,
    seed=123,
)

rows = sample.rows
dets_y = sample.dets
h = sample.h
pgen = sample.pgen
counts = sample.counts
```

`sample_edges` samples from each row with probability

```text
p(y | x) = |H_xy| / sum_z |H_xz|
```

within the requested window. If `counts=None`, only `row_nnz` and `row_weight` are computed.

## Scope

`libdet` is intentionally not a solver framework. It does not own:

- CI selection policy
- optimizer state
- neural-network parameters
- walker populations
- MPI layout
- checkpointing

Those layers should live in Python or downstream applications.

## Future direction

The public API should remain unified while internal Hamiltonian models may specialize. Future internal backends may include spin-orbital UHF kernels, spin-adapted CSF/GUGA kernels, k-point periodic systems, and relativistic spinor Hamiltonians.

These should remain specialized implementations behind the same row-local primitive interface, not user-visible solver frameworks.