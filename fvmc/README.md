# Design and conventions

FockVMC keeps the VMC algorithm visible through a small set of objects. Each
layer owns one mathematical responsibility and depends only on upstream layers.

## Architecture

```text
Sector
  ↓
Hamiltonian ──────┐
                  ↓
Model → Sampler → VState → Optimizer → VMC
```

```text
hilbert
    Sector, DetSector
    configuration layout, particle number, spin, finite enumeration

operator
    Hamiltonian, create, annihilate, number, S2
    matrix elements, connections, projections, finite-space action

model
    RBM and backflow ansatzes
    parameterized log wavefunction

sampler
    ChainState, MCSampler
    burn-in, proposal distributions, MH transitions, Markov kernels

vstate
    ExactState, SelectedState, MCState
    energy, observables, gradients, Geometry

optimizer
    sr, psr
    Optax-compatible geometry transformations

driver
    VMC
    estimation, update, logging, MC checkpoints
```

`utils` contains numerical operations without owning a physical object. New
features should stay in the layer that owns their state and mathematics; shared
syntax alone is not a reason to add a base class or manager.

## Configuration

A configuration is a packed alpha/beta occupation bitstring:

```text
x.dtype = uint64
x.shape = (batch, 2, nword)

x[:, 0] = alpha occupations
x[:, 1] = beta occupations
nword    = ceil(norb / 64)
```

Arrays are batch-first and C-contiguous at the Python/C++ boundary. `Sector`
owns conversion, uniqueness, and sector properties; `DetSector` implements
determinant reference, random sampling, and finite enumeration.

Documentation uses *configuration*. Code uses `x`. Use *determinant* only when
the determinant structure matters.

## Hamiltonian

Hamiltonian actions use Dirac order

$$
H_{bk}=\langle b|H|k\rangle.
$$

`ket` is the input configuration and `bra` the connected output. Therefore

$$
\mathrm{ratio}=\frac{\psi(\mathrm{bra})}{\psi(\mathrm{ket})}.
$$

`Hamiltonian` owns the electronic action. It does not own a model, sampler, or
estimator. Its compiled `libdet` backend follows the same convention.

## Model

Every model implements

```python
logpsi = model.apply(params, x)
```

The result may be a real log-amplitude, a complex log-amplitude, or a
signed-real pair. `Model.logabs`, `Model.coord`, and `Model.cotangent` provide
the views used by sampling and real-coordinate autodiff. Parameters remain
explicit; models do not contain Hamiltonians, chains, or optimizers.

## Initialization

Mean-field choice, model initialization, and chain initialization are separate.
RHF, ROHF, UHF, broken symmetry, orbital localization, and perturbations belong
to the preparation of occupied orbitals.

```python
ref_mat = slater_reference(alpha, beta)
```

The real arrays `alpha` and `beta` have shapes `(norb, n_alpha)` and
`(norb, n_beta)` in the orthonormal Hamiltonian basis. For AO coefficients `C`,
basis coefficients `B`, and overlap `S`, transform orbitals with
`B.T @ S @ C`.

For FCIDUMP in an ordered orbital basis, unit occupied orbitals define the
ordered reference. Alternative references must be expressed in that same basis.

Initial chains are chosen explicitly:

```python
chains = sector.reference(n)
chains = sector.random(n, seed)
chains = sample_slater(sector, ref_mat, n=n, seed=seed)
```

This choice changes burn-in, not the Born distribution.

## Estimation and optimization

The estimator interface is direct:

```python
state, stats = state.expect()
state, energy, grad, stats, geometry = state.expect_and_grad(geometry=True)
```

`ExactState` uses the full sector, `SelectedState` a finite selected space, and
`MCState` a self-normalized importance estimator. Calls return the next state
because chains and tempering may advance. Only `MCState` has a checkpoint
contract.

SR and PSR consume `Geometry`; Optax supplies the learning rate:

```python
optimizer = optax.chain(
    psr(mu=0.95, shift=1.0e-3),
    optax.scale_by_learning_rate(5.0e-2),
)
```

`VMC` performs only

```text
estimate → optimizer update → parameter update → logging
```

## Stable development rules

- Preserve the dependency direction and explicit object construction.
- Keep configuration layout and bra/ket order unchanged across Python and C++.
- Add parameters only when they represent a scientific method choice.
- Prefer a complete mathematical function over a hierarchy of small wrappers.
- Validate new algorithms against exact or selected-space results on a small
  sector before extending large-system scripts.
