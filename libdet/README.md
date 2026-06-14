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

## Screened graph queries

Generated connections can be materialized:

```python
conns = ham.conns(kets, eps)
```

or summarized without storing connected bras:

```python
deg = ham.degrees(kets, eps)
```

`conns` represents a ket-partitioned graph. `degrees` computes the same per-ket connection counts and absolute Hamiltonian weights without materializing the graph.

## Sampling weak connections

Weak connections can be sampled from a window

```text
eps2 <= value < eps1
```

where `value` is either `|H_ai|` or `|H_ai c_i|`.

```python
samples = ham.sample_conns(kets, counts, eps1, eps2, seed=0)
proj_samples = ham.sample_project(kets, coeffs, eps1, eps2, counts, n_rep=2)
```

Sampling probabilities are proportional to absolute Hamiltonian weight within each ket row.

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