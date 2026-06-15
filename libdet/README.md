# libdet

`libdet` is a small C++ kernel library with Python bindings for determinant-space quantum chemistry Hamiltonians.

It provides row-local primitives for building methods around a common idea: a screened Hamiltonian graph over Slater determinants. The library focuses on fast, deterministic Hamiltonian operations; solver policy and scientific workflow remain in Python or downstream codes.

`libdet` is suitable as a backend for Selected CI, Fock-space VMC, FCIQMC-style methods, deterministic PT2, and semi-stochastic PT2.

## What it does

Given a batch of determinants, `libdet` can evaluate diagonal and off-diagonal Hamiltonian elements, build exact sparse blocks, apply Hamiltonian matrix-vector products, generate screened connected determinants, and sample weak external connections.

The central abstraction is the screened graph

```text
E_eps(i) = { (a, H_ai) : a != i, |H_ai| >= eps }
```

or, when coefficients are involved,

```text
E_eps(psi) = { (a, i, H_ai c_i) : |H_ai c_i| >= eps }
```

Most higher-level algorithms can be expressed as different ways of querying, reducing, or sampling this graph.

## Non-goals

`libdet` is not a solver framework. It intentionally does not manage:

- CI selection policy;
- walker populations;
- neural-network or variational parameters;
- MPI layout;
- checkpointing;
- convergence logic.

Those decisions belong in downstream applications. `libdet` only supplies the Hamiltonian primitives.

## Determinants

Determinants are stored as packed `uint64` spin strings:

```python
(N, 2, nword)
```

The second axis stores alpha words first and beta words second.

```python
import libdet

dets = libdet.to_dets(dets)
```

## Basic use

```python
import libdet

ham = libdet.Hamiltonian.rhf(h1, eri, ecore=0.0)

diag = ham.diags(dets)
H = ham.matrix(bras, kets)
y = ham.matvec(bras, x, kets=kets)

ext = ham.expand(kets, eps)

proj_known = ham.project(bras, kets, coeffs)
proj_ext = ham.project(None, kets, coeffs, eps=eps)
```

If `kets` is omitted in finite-space operations, it defaults to `bras`.

## Row queries

The public row-query surface consists of three operations:

```python
weight, nconn = ham.degrees(kets, eps)

sample = ham.sample_conns(
    kets,
    counts,
    eps1=np.inf,
    eps2=eps,
    seed=seed,
)

graph = ham.conns(
    kets,
    eps,
    sample=n_sample,
    sample_eps=weak_eps,
    seed=seed,
)
```

`degrees` returns row weights and connection counts without materializing
destinations.

`sample_conns` performs categorical sampling in
`eps2 <= |H_ai| < eps1`. `counts` has shape `(N,)` or `(S, N)`; streams
share one candidate scan and use independent deterministic targets.

`conns` returns exact CSR rows and can sample the weak window
`sample_eps <= |H_ai| < eps` in the same determinant pool.

Both result types store a global `dets` pool whose first `N` entries equal the
input kets. All `col` arrays index this pool directly.

`ConnSamples` stores stream-major CSR rows:

```text
dets, ptr, col, h, count, weight
```

`Conns` stores exact and sampled rows in the same pool:

```text
dets, diag, ptr, col, h, weight
sample_ptr, sample_col, sample_h, sample_count, sample_weight
```

## Design principles

`libdet` is designed to stay:

- row-local;
- deterministic unless sampling is requested;
- solver-agnostic;
- explicit about determinant and coefficient alignment;
- small enough to audit.

Internal acceleration structures, screening tables, finite-space indices, and scheduling details are backend-private. The public interface should remain centered on Hamiltonian semantics rather than implementation controls.

## Direction

Future backends or Hamiltonian types should preserve the same primitive model:

```text
determinant batches -> Hamiltonian rows -> screened graph -> reduction or sampling
```

Possible extensions include UHF/spin-orbital kernels, symmetry-resolved Hamiltonians, spin-adapted bases, periodic systems, relativistic spinors, and GPU backends.
