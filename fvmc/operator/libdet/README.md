# libdet

`libdet` is the compiled Hamiltonian oracle used by `Hamiltonian`. It evaluates
matrix elements, connections, projections, and finite-space actions. Sectors,
wavefunctions, sampling, and estimators remain in Python.

## Data convention

```text
configuration.dtype = uint64
configuration.shape = (2, nword)
batch.shape         = (N, 2, nword)

block 0 = alpha occupations
block 1 = beta occupations
nword  = ceil(norb / 64)
```

Arrays crossing the nanobind boundary are C-contiguous. Inputs are immutable;
result objects own their returned arrays.

Matrix elements use

$$
H_{bk}=\langle b|H|k\rangle,
$$

where `ket` is the input and `bra` the excited output. Records are grouped by
ket. One-electron integrals are $h_{pq}$; two-electron integrals use the PySCF
chemist convention $g_{pqrs}=(pq|rs)$ in eight-fold packed form. Matrix
elements follow the Slater--Condon rules in the fixed sector.

## Operations

```text
hij(bra, ket)                   matrix element H[bra, ket]
diag(x)                         diagonal elements
conn(kets, eps)                 deterministic connections
sample_conn(kets, counts, ...)  sampled screened connections
local_conn(kets, ...)           strong and weak local-energy records

matrix(bras, kets=None)         sparse H[bras, kets]
matvec(bras, x, kets=None)      H[bras, kets] x

expand(kets, eps, ...)          unique external configurations
project(bras, kets, scale, ...) projected Hamiltonian action
sample_project(...)             sampled screened projection
```

If `kets` is omitted from `matrix` or `matvec`, the bra space is used on both
sides. A two-dimensional vector argument applies the same action to multiple
vectors. These actions are exact in the supplied finite spaces.

For projection coefficients $c_k$,

$$
(Hc)_b=\sum_k H_{bk}c_k.
$$

With `bras=None`, `project` generates external bras. `exclude` removes a known
space from external generation.

## Connection layout

For `Conns` with `N` input kets:

```text
bra[:N]          input kets
bra[N + p]       connected bra for record p
ptr[i]:ptr[i+1]  records belonging to ket i
h[p]             H[bra[N + p], ket[i]]
diag[i]          H[ket[i], ket[i]]
degree[i]        sum of retained |H_bk|
```

Multi-stream sampled pointers are stream-major, then ket-major. Repeated
sampled bras are valid records.

`LocalConn.bra` has layout

```text
[kets, strong bras, weak bras]
```

`strong_ptr/strong_h` and `weak_ptr/weak_coeff` index their own value arrays.

## Screening

All screening uses matrix-element magnitude:

```text
deterministic    |H_bk| >= eps
strong           |H_bk| >= eps1
weak             eps2 <= |H_bk| < eps1
```

A zero cutoff requests complete direct enumeration. Each ket receives the
requested number of valid weak draws; their coefficients form an unbiased
retained-window sum.

`local_conn` requires `eps2 > 0` and positive counts when `eps2 < eps1`.

## Stable development rules

- Preserve configuration layout, bra/ket order, and pointer semantics.
- Return numerical data directly; do not expose internal lookup tables.
- Keep sampling laws, wavefunctions, and estimator state outside C++.
- Add a primitive only when it expresses an independent Hamiltonian operation.
- Verify new actions against explicit small-space matrices.
