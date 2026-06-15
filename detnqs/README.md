# detnqs

`detnqs` is the JAX variational layer of the project. It defines neural
wavefunctions, variational states, sampling, natural-gradient optimizers, and
the optimization loop. Determinant Hamiltonian operations are provided by
[`libdet`](../libdet/README.md).

## Components

- `detnqs.model`: `RBM`, `Backflow`, and `RBackflow` wavefunctions.
- `detnqs.vstate`: exact, selected-space, and Monte Carlo estimators.
- `detnqs.sampler`: Enhanced Markov chains.
- `detnqs.optimizer`: SR, minSR, and PSR.
- `detnqs.driver`: the minimal `VMC` optimization loop.
- `detnqs.utils`: shared batching and numerical precision policy.

The main data flow is:

```text
model + Hamiltonian
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

The available state types differ only in how the determinant distribution is
represented:

- `ExactState`: complete fixed determinant space.
- `SelectedState`: explicit determinant space that can be enlarged.
- `MCState`: samples determinants with `MCSampler`.

## Minimal Workflow

Given a `libdet.Hamiltonian` named `ham`:

```python
import jax
import optax

from detnqs.driver import VMC
from detnqs.model import RBM
from detnqs.optimizer import sr
from detnqs.vstate import ExactState

model = RBM(norb=ham.norb, alpha=1)
state = ExactState.init(
    model=model,
    hamiltonian=ham,
    n_alpha=n_alpha,
    n_beta=n_beta,
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

Models return a logarithmic wavefunction representation. Determinants use the
`uint64` layout defined by `libdet`, and JAX 64-bit mode is enabled because
occupation strings are stored as `uint64`.
