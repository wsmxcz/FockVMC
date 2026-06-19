# detnqs.operator.libdet

`libdet` is the compiled Hamiltonian oracle behind
`detnqs.operator.Hamiltonian`. Normal Python code should use
`detnqs.operator.Hamiltonian`, not this module directly.

## Boundary

- Python validates inputs and exposes the public API.
- C++ assumes contiguous project-native arrays.
- `libdet` computes matrix elements, screened connections, projections,
  sparse matrices, samples, and matrix-vector products.
- Solver policy, sampling policy, models, optimizers, and drivers stay outside
  `libdet`.

## Determinant Encoding

A determinant batch is a contiguous `uint64` array:

```text
(n_det, 2, nword)
```

Axis 1 stores alpha words first and beta words second. Names follow Dirac
notation: `ket` is the source state, `bra` is the destination state, and `det`
is used when no direction is implied.

Integral tensors follow the Python `Hamiltonian` convention: spatial-orbital
`h1`, flattened `eri`, and scalar `ecore`.

## Primitive Model

The backend implements the primitive operator actions used by the Python
Hamiltonian:

- `hij`, `diag`
- `expand`, `project`
- `conn`, `sample_conn`
- `sample_project`
- `matrix`, `matvec`

These primitives are row-local and mergeable. Higher-level algorithms should be
expressed as reductions, sampling schemes, or solvers around them.

## Source Layout

- `module.cpp`: nanobind boundary.
- `csrc/libdet/spatial`: packed determinant operations and spaces.
- `csrc/libdet/rhf`: determinant electronic Hamiltonian backend.
- `csrc/libdet/guga`: spin-adapted backend.
- `csrc/libdet/hamiltonian.hpp`: common C++ facade.

## Kernel Shape

```text
integral -> element
               |
               v
         screen, cache -> internal
                       -> external
                       -> sample
```

This layout is descriptive. It should remain an implementation detail behind
the unified Hamiltonian interface.
