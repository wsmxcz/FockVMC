# Stochastic reconfiguration

`sr` and `psr` transform an energy gradient using local wavefunction geometry.
They do not own sampling, local energies, or learning rates.

## Geometry

Let $q_\theta(x)\in\mathbb R^d$ be the real differentiable model coordinate.
For normalized weights $w_n$, define

$$
J_n=\frac{\partial q_\theta(x_n)}{\partial\theta},
\qquad
\bar J=\sum_n w_nJ_n,
$$

$$
O_n=\sqrt{w_n}(J_n-\bar J),
\qquad
S=O^\dagger O.
$$

`Geometry` is the estimator-to-optimizer contract:

```text
params    parameters at which the estimate was evaluated
coord     q(params, x)
x         estimator configurations
weight    normalized weights
b         sample-space energy residual
```

The optimizer normalizes `weight`. The estimator constructs every other field.

## Parameter-space SR

For gradient $g$ and shift $\lambda$,

$$
(S+\lambda I)d=g.
$$

`sr(mode="dense")` forms $S$ explicitly and is intended for small reference
problems. `sr(mode="matvec")` applies $S$ through JVP/VJP products and solves
iteratively. Both return the unscaled direction $d$.

## Sample-space SR

When $g=O^\dagger b$,

$$
(O^\dagger O+\lambda I)^{-1}O^\dagger
=O^\dagger(OO^\dagger+\lambda I)^{-1}.
$$

Therefore

$$
(K+\lambda I)a=b,
\qquad
K=OO^\dagger,
\qquad
d=O^\dagger a.
$$

`psr(mu=0)` implements this sample-space solve.

## Predictive SR

For `mu > 0`, PSR predicts from the previous unscaled direction:

$$
p_t=\mu d_{t-1},
\qquad
r_t=b_t-O_tp_t,
$$

$$
(K_t+\lambda I)a_t=r_t,
\qquad
d_t=p_t+O_t^\dagger a_t.
$$

The optimizer state stores $d_t$ before the learning-rate transform. Thus
`mu` controls prediction in geometry space, not step size.

## Optax composition

SR or PSR precedes the Optax learning-rate transform:

```python
optimizer = optax.chain(
    psr(mu=0.95, shift=1.0e-3),
    optax.scale_by_learning_rate(5.0e-2),
)
```

The driver passes `geometry` through the standard update boundary:

```python
updates, opt_state = optimizer.update(
    grad,
    opt_state,
    params,
    geometry=geometry,
)
params = optax.apply_updates(params, updates)
```

Public method parameters are limited to the solve:

```text
shift      diagonal regularization
mode       dense or matrix-free parameter-space SR
maxiter    matrix-free iteration limit
mu         predictive strength; zero gives sample-space SR
```

## Stable development rules

- Geometry construction belongs to the estimator; geometry solves belong here.
- Directions remain unscaled so Optax controls learning rates and schedules.
- Dense and matrix-free SR must represent the same shifted equation.
- New solvers should first agree with dense SR on a small exact problem.
