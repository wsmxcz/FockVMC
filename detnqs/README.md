# detnqs Package

`detnqs` is the Python variational layer. It owns the public Fock-space API,
model evaluation, estimators, optimizers, and the VMC loop. Compiled electronic
Hamiltonian work is delegated to `detnqs.operator.libdet`.

## Module Boundaries

- `hilbert`: sectors and the packed `x` configuration language.
- `operator`: physical operators; `Hamiltonian` is the public operator-action
  boundary.
- `model`: Flax/JAX wavefunction models and log-amplitude conventions.
- `vstate`: exact, selected-basis, and Monte Carlo estimators.
- `sampler`: Metropolis chains used by `MCState`.
- `optimizer`: Optax-compatible SR, minSR, and PSR transforms.
- `driver`: the minimal `VMC` iteration loop.
- `utils`: JAX batching, precision policy, math helpers, timing, and logging.

## Data Flow

```text
Sector + Hamiltonian + Model
        |
        v
state.expect_and_grad()
        |
        v
loss, gradient, statistics, geometry
        |
        v
optimizer update
```

The sector owns the basis representation. The Hamiltonian owns operator action.
The model owns wavefunction evaluation. The variational state owns estimators
and geometry construction. The driver owns iteration only.

## Public Conventions

- `Sector` denotes a constrained Fock-space sector.
- `x` denotes a batch of Fock-basis configurations.
- `det` denotes determinant-specialized `x`: alpha block, then beta block.
- `bra` and `ket` denote the axes of `H[bra, ket]`.
- `eri` is PySCF chemist s8: $$g_{pqrs} = (pq|rs).$$
- Public symmetry names should be mathematical group names, for example `U1`
  or `SU2`.

Validation belongs at user-facing boundaries. Internal paths should assume the
project representation and avoid repeated conversions.

## Theory Notes

- `optimizer/sr.md`: stochastic reconfiguration and sample-space SR.
- `sampler/vmc.md`: Fock-space VMC measures, reweighting, and blurred sampling.
