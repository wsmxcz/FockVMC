# detnqs

`detnqs` is the JAX variational layer of the project. It defines Fock-space
sectors, physical operators, neural wavefunctions, variational states,
sampling, natural-gradient optimizers, and the optimization loop. Fast
Hamiltonian kernels are provided by [`libdet`](../libdet/README.md).

## Components

- `detnqs.hilbert`: Fock-space sectors such as `DetSpace` and `CsfSpace`.
- `detnqs.operator`: physical operators, primarily `H`.
- `detnqs.model`: `RBM`, `Backflow`, and `RBackflow` wavefunctions.
- `detnqs.vstate`: exact, selected-space, and Monte Carlo estimators.
- `detnqs.sampler`: Metropolis chains and proposal rules.
- `detnqs.optimizer`: SR, minSR, and PSR.
- `detnqs.driver`: the minimal `VMC` optimization loop.
- `detnqs.utils`: shared batching and numerical precision policy.

The main data flow is:

```text
Space + Hamiltonian + model
        |
        v
variational state -> loss, gradient, statistics, geometry
        |
        v
optimizer -> parameter update
```

The variational state owns physical state and estimators. The driver owns only
optimizer state and iteration.

## Variational States

All states expose the same optimization contract:

```python
state, loss, grad, stats, geometry = state.expect_and_grad(
    geometry=True,
)
```

The available state types differ only in how the distribution over Fock
configurations is represented:

- `ExactState`: complete finite sector.
- `SelectedState`: explicit basis that can be enlarged.
- `MCState`: samples configurations with `MCSampler`.

## Minimal Workflow

```python
import jax
import optax

from detnqs import hilbert, operator
from detnqs.driver import VMC
from detnqs.model import RBM
from detnqs.optimizer import sr
from detnqs.vstate import ExactState

space = hilbert.DetSpace(norb, n_alpha, n_beta)
H = operator.Hamiltonian(space, h1, eri, ecore=ecore)

model = RBM(norb=norb, alpha=1)
state = ExactState.init(
    model=model,
    H=H,
    key=jax.random.key(0),
)

optimizer = optax.chain(
    sr(shift=1.0e-3),
    optax.scale(-1.0e-2),
)
vmc = VMC.init(state, optimizer)
stats = vmc.step()
```

Use `geometry=False` with ordinary Optax optimizers that do not require SR
geometry. Complete workflows are in [`examples`](../examples).

## Conventions

Models return a logarithmic wavefunction representation. Spaces own
the concrete `x` encoding. Current determinant and CSF sectors use the compact
`uint64` occupation layout required by the `libdet` backend.
