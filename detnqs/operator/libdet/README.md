# libdet

`libdet` is the compiled Hamiltonian oracle behind
`detnqs.operator.Hamiltonian`. Its purpose is narrow: given packed electronic
configurations and molecular integrals, it returns matrix elements,
connections, projections, or finite-space actions.

Python owns sectors, wavefunctions, sampling, and estimators. The C++ layer owns
only the performance-critical Hamiltonian action and returns flat arrays that
Python can consume directly.

## Configuration layout

A configuration contains separate packed alpha and beta occupations:

```text
configuration.dtype = uint64
configuration.shape = (2, nword)
batch.shape         = (N, 2, nword)

block 0 = alpha occupations
block 1 = beta occupations
nword  = ceil(norb / 64)
```

Arrays crossing the nanobind boundary are C-contiguous. Input configuration
memory is treated as immutable; returned arrays are owned by their result
objects.

## Hamiltonian convention

Matrix elements use Dirac order:

$$
H_{bk}=\langle b|H|k\rangle.
$$

`ket` is the input configuration. `bra` is the configuration produced by an
excitation. Connection records are grouped by ket.

One-electron integrals are $h_{pq}$. Two-electron integrals use the PySCF
chemist convention

$$
g_{pqrs}=(pq|rs)
$$

and are accepted in PySCF eight-fold packed form. Matrix elements follow the
Slater--Condon rules in the fixed particle and spin sector supplied by Python.

## Core actions

The Python `Hamiltonian` wrapper exposes four groups of operations.

### Elements and connections

```text
hij(bra, ket)                 matrix element H[bra, ket]
diag(x)                       diagonal for a configuration batch
conn(kets, eps)               deterministic off-diagonal connections
sample_conn(kets, counts, ...) sampled connections in a screened window
local_conn(kets, ...)         strong and weak local-energy connections
```

Deterministic connections satisfy

$$
|H_{bk}|\ge\epsilon.
$$

Sampled connections use the window

$$
\epsilon_2\le|H_{bk}|<\epsilon_1.
$$

`local_conn` returns both regions in one result:

```text
strong    |H_bk| >= eps1
weak      eps2 <= |H_bk| < eps1
```

Weak records include sampling degree and multiplicity so the caller can form
an unbiased retained-window sum.

### Connection result layout

For `Conns` with `n_kets = N`:

```text
bra[:N]          input kets
bra[N + p]       bra for off-diagonal record p
ptr[i]:ptr[i+1]  records belonging to ket i
h[p]             H[bra[N + p], ket[i]]
diag[i]          H[ket[i], ket[i]]
degree[i]        sum of retained |H_bk|
```

For multi-stream sampled connections, pointer rows are stream-major and then
ket-major. Repeated sampled bras are valid records.

`LocalConn.bra` is laid out as

```text
[kets, strong bras, weak bras]
```

with separate `strong_ptr/strong_h` and `weak_ptr/weak_h` arrays. The pointer
arrays always refer to records in their corresponding Hamiltonian-value array.

### Finite-space action

```text
matrix(bras, kets=None)       sparse H[bras, kets]
matvec(bras, x, kets=None)    H[bras, kets] x
```

If `kets` is omitted, the bra space is used on both sides. A two-dimensional
`x` applies the same action to multiple vectors. These operations are exact in
the supplied finite spaces.

### Expansion and projection

```text
expand(kets, eps, scale=None, exclude=None)
project(bras, kets, scale, eps=0, exclude=None)
sample_project(kets, scale, counts, eps1, eps2, exclude, seed)
```

`expand` returns unique connected external configurations. `project` evaluates

$$
(Hc)_b=\sum_k H_{bk}c_k
$$

for supplied bras, or generates external bras when `bras=None`. `exclude`
removes a known space during external generation. `sample_project` provides the
corresponding screened stochastic projection.

## Screening contract

Screening is always defined by matrix-element magnitude. A zero cutoff requests
complete direct enumeration; positive cutoffs restrict connection or projection
work to the stated region.

The returned matrix elements, record order, degrees, and multiplicities fully
describe the selected or sampled action. Internal lookup tables and parallel
workspaces are performance details and do not change this data contract.
