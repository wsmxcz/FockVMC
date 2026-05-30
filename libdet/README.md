# libdet

`libdet` is a lightweight C++ kernel with Python bindings for determinant-space quantum chemistry Hamiltonians.

It provides row-local Hamiltonian primitives for selected CI, Fock-space VMC, FCIQMC-style workflows, deterministic PT2, and semi-stochastic PT2. Solver policy, wavefunction updates, batching, MPI orchestration, checkpointing, and optimization are intentionally left to Python or downstream codes.

The library is designed to be small, explicit, and easy to compose.

---

## Scope

`libdet` owns low-level determinant Hamiltonian operations:

- diagonal matrix elements
- individual Hamiltonian elements
- exact finite-space matrix blocks
- exact finite-space matrix-vector and matrix-matrix products
- screened external determinant generation
- projected amplitudes
- per-ket connection statistics
- stochastic sampling of weak connections

`libdet` does not own:

- CI selection policy
- optimizer state
- neural-network parameters
- walker populations
- MPI layout
- checkpointing
- trial wavefunction management
- solver convergence logic

Those layers should live in Python or downstream applications.

---

## Conventions

Determinants are stored as unsigned 64-bit bit strings with shape:

```python
(N, 2, nword)
```

Axis 1 stores alpha words first, then beta words.

Naming follows Dirac notation:

- `dets`: determinant batch with no left/right role
- `kets`: domain determinants, aligned with `coeffs`, `x`, or `X`
- `bras`: codomain determinants
- `H[bras, kets]`: Hamiltonian block
- `coeffs`: wavefunction coefficients aligned with `kets`
- `x`, `X`: linear algebra inputs aligned with `kets`

Use `dets` only when no direction matters. Use `kets` and `bras` whenever the Hamiltonian role matters.

Off-diagonal Hamiltonian connections are single and double excitations. All kernels are row-local and mergeable: rows may be sharded, computed independently, and reduced outside the library.

---

## Basic use

```python
import numpy as np
import libdet

ham = libdet.Hamiltonian.rhf(h1, eri, ecore=ecore)

# Determinants are always shaped as (N, 2, nword),
# even when nword == 1.
dets = libdet.to_dets(dets)

# Diagonal matrix elements.
diag = ham.diags(dets)

# Exact sparse Hamiltonian block H[dets, dets].
H = ham.matrix(dets)

# Exact matrix-vector product H[dets, dets] @ coeffs.
y = ham.matvec(dets, coeffs)

# Exact matrix-matrix product H[dets, dets] @ X.
Y = ham.matvec(dets, X)
```

For a rectangular Hamiltonian block:

```python
H = ham.matrix(bras, kets)
y = ham.matvec(bras, x, kets=kets)
```

`x` and `X` are always aligned with `kets`.

---

## Public interface

### Hamiltonian construction

```python
ham = libdet.Hamiltonian.rhf(h1, eri, ecore=0.0)
```

The current public constructor builds an RHF spin-free Hamiltonian from one- and two-electron integrals.

---

### Matrix elements

```python
h = ham.hij(bra, ket)
diag = ham.diags(dets)
```

`hij` expects one bra determinant and one ket determinant, each represented as a determinant batch of length 1.

---

### Exact finite-space operations

```python
H = ham.matrix(bras, kets=None)
y = ham.matvec(bras, x, kets=None)
```

If `kets` is omitted, it defaults to `bras`.

`matrix` returns a SciPy CSR matrix by default. With `raw=True`, it returns the internal `Matrix` object:

```python
M = ham.matrix(bras, kets, raw=True)

M.shape
M.diags
M.indptr
M.indices
M.data
```

For `Matrix`, rows are `bras` and columns are `kets`. The CSR fields keep their standard names.

---

### External determinant generation

```python
cand = ham.expand(kets, eps=1e-6, coeffs=coeffs, exclude=kets)
```

`expand` generates unique external bras connected to `kets`.

With `coeffs`, screening uses:

```text
|H_ai c_i| >= eps
```

Without `coeffs`, screening uses:

```text
|H_ai| >= eps
```

`exclude` defaults to `kets`.

---

### Projection

Projection computes Hamiltonian action on a specified or generated bra space.

For a known bra space:

```python
proj = ham.project(bras, kets, coeffs)

bras_out = proj.bras
hpsi = proj.hpsi
diags = proj.diags
```

This computes:

```text
hpsi = H[bras, kets] @ coeffs
```

For generated external bras:

```python
proj = ham.project(None, kets, coeffs, eps=1e-6, exclude=kets)
```

This generates connected bras outside `exclude` and accumulates screened contributions directly.

---

### Generated connections

```python
conns = ham.conns(kets, eps=1e-6)
```

`conns` stores screened Hamiltonian connections generated from each ket.

```python
kets = conns.kets
bras = conns.bras

ket_ptr = conns.ket_ptr
bra = conns.bra
h = conns.h

ket_nconn = conns.ket_nconn
ket_weight = conns.ket_weight
diags = conns.diags
```

The connection layout is ket-based:

```python
for iket in range(conns.n_kets):
    p0 = conns.ket_ptr[iket]
    p1 = conns.ket_ptr[iket + 1]

    for p in range(p0, p1):
        ibra = conns.bra[p]
        hij = conns.h[p]
        bra_det = conns.bras[ibra]
```

Here:

- `ket_ptr` partitions connections by source ket
- `bra[p]` indexes `conns.bras`
- `h[p]` is the Hamiltonian element between that bra and ket
- `ket_weight[iket]` is the sum of `abs(h)` for that ket
- `ket_nconn[iket]` is the number of generated bras for that ket

---

### Degree and weight queries

```python
deg = ham.degrees(kets, eps=1e-6)

ket_nconn = deg.ket_nconn
ket_weight = deg.ket_weight
```

`degrees` computes the same per-ket counts and absolute Hamiltonian weights as `conns`, but does not materialize the connected bras.

Use it when only row statistics are needed.

---

### Stochastic connection sampling

```python
sample = ham.sample_conns(
    kets,
    counts=64,
    eps1=1e-3,
    eps2=1e-6,
    seed=123,
)
```

The sampled window is:

```text
eps2 <= |H_ai| < eps1
```

Within each ket, connections are sampled with probability:

```text
p(a | i) = |H_ai| / sum_b |H_bi|
```

The result stores sampled bras and their generation probabilities:

```python
sample.ket_nconn
sample.ket_weight

sample.ket
sample.bras
sample.h
sample.pgen
sample.counts
```

If `counts=None`, only `ket_nconn` and `ket_weight` are computed.

---

### Stochastic projected amplitudes

```python
sample = ham.sample_project(
    kets,
    coeffs,
    eps1=1e-3,
    eps2=1e-6,
    counts=64,
    exclude=kets,
    n_rep=2,
    seed=123,
)
```

The weak window is:

```text
eps2 <= |H_ai c_i| < eps1
```

The result stores sampled projected amplitudes by replica:

```python
sample.rep_ptr
sample.bras
sample.diags
sample.hpsi_strong
sample.hpsi_a
sample.hpsi_b
```

`hpsi_strong` contains the deterministic contribution above `eps1`. `hpsi_a` and `hpsi_b` are two independently sampled weak estimates, useful for semi-stochastic estimators.

---

## High-level design

`libdet` uses two complementary kernels.

### 1. Determinant-driven finite-space search

Used when both sides of the Hamiltonian block are known:

```python
ham.matrix(bras, kets)
ham.matvec(bras, x, kets=kets)
ham.project(bras, kets, coeffs)
```

The ket space is indexed once. Each bra searches only connected kets inside that finite ket space.

This path is used for exact matrix blocks, exact matrix-vector products, exact matrix-matrix products, and projection onto a known bra space.

### 2. Excitation-driven external generation

Used when the target bra space is not known in advance:

```python
ham.expand(kets, eps, coeffs=coeffs, exclude=dets)
ham.conns(kets, eps)
ham.degrees(kets, eps)
ham.sample_conns(kets, counts, eps1, eps2)
ham.sample_project(kets, coeffs, eps1, eps2, counts)
```

Each ket is expanded by single and double excitations. Screened double excitations use a heat-bath table ordered by integral magnitude. Pure degree queries compute counts and weights without materializing connected bras.

---

## Core algorithmic ideas

### Spin-separated determinant indexing

Finite-space search indexes alpha and beta spin strings separately. This allows connected determinants to be found by matching unchanged or partially changed spin sectors.

Same-spin singles, same-spin doubles, and opposite-spin doubles are handled separately, then combined into Hamiltonian elements.

### Residue search for finite spaces

For a fixed finite ket space, spin strings are grouped by residues obtained by removing one or two occupied orbitals from a fingerprint. For a given bra, matching residues identify candidate ket spin strings that may differ by one or two orbitals.

Candidates are verified exactly before Hamiltonian elements are emitted.

### Heat-bath acceleration for screened doubles

Double-excitation integral candidates are precomputed and sorted by absolute value. For screened external generation, the scan can stop early once candidates fall below the requested threshold.

If the requested threshold is below the heat-bath table cutoff, exact double-excitation enumeration is used.

### Row-local execution

All major kernels operate independently over bras or kets. The library does not assume a global solver layout. Outputs are designed to be merged or reduced by downstream code.

---

## Development conventions

### Naming

Use Dirac notation for Hamiltonian roles:

- `dets` for undirected determinant batches
- `kets` for domain determinants
- `bras` for codomain determinants

Use `coeffs`, `x`, and `X` only for arrays aligned with `kets`.

Use graph-like names only when describing the storage layout of generated connections. Prefer `conns`, `ket_ptr`, `bra`, and `h`.

Use standard sparse matrix names only inside matrix/CSR objects:

- `indptr`
- `indices`
- `data`

### Indices

Orbital indices follow quantum chemistry convention:

- `i, j, k, l`: occupied orbitals
- `a, b, c, d`: virtual orbitals
- `p, q, r, s`: general orbitals or integral indices

Determinant and array indices should be explicit:

- `idet`: index into `dets`
- `iket`: index into `kets`
- `ibra`: index into `bras`
- `idx`: generic lookup result
- `pos`: position in a packed buffer or sparse structure

### Thresholds

Use:

- `eps` for a single threshold
- `eps1`, `eps2` for a window
- window convention: `eps2 <= value < eps1`

Avoid introducing alternate names such as `eps_hi`, `eps_lo`, `threshold`, or `cutoff` for user-facing screening parameters.

### Design principles

Keep the library:

- row-local
- deterministic unless sampling is explicitly requested
- solver-agnostic
- easy to shard and merge
- explicit about ownership and array alignment
- small enough to audit

Prefer short names when the concept is local and conventional. Prefer explicit names when the object crosses an API boundary.

---

## Future direction

The public API should remain unified around row-local determinant Hamiltonian primitives.

Future internal Hamiltonian models may include:

- spin-orbital UHF kernels
- spin-adapted CSF or GUGA kernels
- k-point periodic systems
- relativistic spinor Hamiltonians

These should remain specialized implementations behind the same primitive interface. They should not turn `libdet` into a solver framework.

The long-term goal is a compact, high-performance kernel library that downstream methods can compose freely.