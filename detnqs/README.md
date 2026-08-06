# Package design

DetNQS separates the VMC calculation by mathematical responsibility. A layer
owns its data and operation, then passes a small explicit result to the next
layer. This keeps the algorithm visible in ordinary Python code.

## Layers and ownership

```text
hilbert
    Sector, DetSector
    configuration layout, particle number, spin sector, finite enumeration

operator
    Hamiltonian, create, annihilate, number, S2
    matrix elements, connections, projection, and finite-space action

model
    RBM and backflow ansatzes
    parameterized log wavefunction

sampler
    ChainState, MCSampler
    burn-in, Markov transitions, proposals, and observation blur

vstate
    ExactState, SelectedState, MCState
    energy, observables, gradient, and SR geometry

optimizer
    sr, psr
    Optax-compatible geometry preconditioners

driver
    VMC
    estimator call, optimizer call, parameter update, logging, checkpointing
```

`utils` contains shared numerical operations such as batching, precision,
statistics, timing, logging, and checkpoint serialization. It does not own a
domain object.

## End-to-end flow

```text
FCIDUMP
  ↓
Sector + Hamiltonian
  ↓
model parameters + ChainState
  ↓
ExactState / SelectedState / MCState
  ↓
energy + gradient + Geometry
  ↓
sr / psr + Optax learning-rate transform
  ↓
parameter update in VMC
```

Construction is explicit: a model does not know the Hamiltonian, a sampler does
not compute energy, and an optimizer does not sample configurations. The
variational state is the point where these objects meet to form an estimator.

## Configuration convention

A configuration is a packed alpha/beta occupation bitstring:

```text
x.dtype = uint64
x.shape = (batch, 2, nword)

x[:, 0] = alpha occupations
x[:, 1] = beta occupations
nword    = ceil(norb / 64)
```

Arrays are batch-first and C-contiguous at the Python/C++ boundary. `Sector`
owns conversion, uniqueness, and sector properties; `DetSector` adds reference,
random, and complete finite-sector construction.

Public documentation uses *configuration*. Code uses `x`. The word
*determinant* is reserved for determinant-specific algorithms and data.

## Hamiltonian convention

Hamiltonian actions use Dirac order:

$$
H_{bk}=\langle b|H|k\rangle.
$$

`ket` is the input configuration and `bra` is a connected output
configuration. Excitations run from ket to bra, and wavefunction ratios are
therefore written as

$$
\mathrm{ratio}=\frac{\psi(\mathrm{bra})}{\psi(\mathrm{ket})}.
$$

`operator.Hamiltonian` is the Python interface. Its compiled `libdet` backend
provides matrix elements and connection data without owning sampling or
wavefunction state.

## Model convention

Every model supports:

```python
logpsi = model.apply(params, x)
```

The raw logarithmic representation may be a positive real log-amplitude, a
complex log-amplitude, or a signed-real pair. `Model.logabs`, `Model.coord`, and
`Model.cotangent` provide the common views needed by sampling and real-coordinate
autodiff.

Model parameters are passed explicitly. Models do not contain a Hamiltonian,
sampler, optimizer, or chain state.

## Variational-state convention

The three estimator implementations share direct call shapes:

```python
state, stats = state.expect()
state, energy, grad, stats, geometry = state.expect_and_grad(geometry=True)
```

- `ExactState` sums the complete sector.
- `SelectedState` sums a finite selected basis and can evolve that basis.
- `MCState` samples an auxiliary law and reweights observations to the Born
  target.

Estimator calls return the next state because sampling can advance chains and
adapt tempering. `MCState` checkpoints parameters and chain state. Exact and
selected states are deterministic working objects and have no checkpoint
contract.

## Optimizer and driver convention

`sr` and `psr` transform a gradient using `Geometry`. Learning rates and
schedules are separate Optax transforms:

```python
optimizer = optax.chain(
    psr(mu=0.95, shift=1.0e-3),
    optax.scale_by_learning_rate(5.0e-2),
)
```

The driver follows the ordinary Optax update flow with one additional geometry
argument:

```python
state, energy, grad, stats, geometry = state.expect_and_grad(geometry=True)
updates, opt_state = optimizer.update(
    grad,
    opt_state,
    state.params,
    geometry=geometry,
)
params = optax.apply_updates(state.params, updates)
state = state.replace(params=params)
```

This is the complete optimization boundary: estimation belongs to the state,
geometry transformation belongs to the optimizer, and orchestration belongs to
`VMC`.
