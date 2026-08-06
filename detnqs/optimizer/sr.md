# Stochastic reconfiguration

`sr` and `psr` turn an energy gradient into a direction measured in local
wavefunction geometry. They are Optax-compatible preconditioners: geometry is
their responsibility, while the learning rate and update schedule remain
ordinary Optax transforms.

## Geometry

Models expose a real differentiable coordinate

$$
q_\theta(x)\in\mathbb R^d.
$$

Depending on the model, this coordinate represents a log-amplitude alone or a
real view of log-amplitude and phase. For normalized Born weights $w_n$, define

$$
J_n=\frac{\partial q_\theta(x_n)}{\partial\theta},
\qquad
\bar J=\sum_n w_nJ_n,
$$

$$
O_n=\sqrt{w_n}(J_n-\bar J).
$$

After flattening configuration and coordinate axes, $O$ is the centered
weighted Jacobian and

$$
S=O^\dagger O
$$

is the sampled quantum-geometric tensor.

The `Geometry` object carries the estimator side of this contract:

```text
params    parameters at which the estimate was evaluated
coord     real model coordinate q(params, x)
x         configurations used by the estimate
weight    Born weights for those configurations
b         sample-space energy residual
```

The optimizer normalizes `weight` before constructing the centered geometry.
The Hamiltonian, sampler, and local-energy calculation remain outside the
optimizer.

## Parameter-space SR

For gradient $g$ and diagonal shift $\lambda$, parameter-space SR solves

$$
(S+\lambda I)d=g.
$$

The returned tree $d$ is an unscaled natural-gradient direction.

`sr(mode="dense")` forms $S$ explicitly. It is appropriate for small models
and reference calculations. `sr(mode="matvec")` applies $S$ through JVP/VJP
products and solves iteratively, avoiding a dense parameter-space matrix.
Both modes implement the same equation.

## Sample-space SR

When the gradient is represented by the sample-space residual,

$$
g=O^\dagger b,
$$

the identity

$$
(O^\dagger O+\lambda I)^{-1}O^\dagger
=O^\dagger(OO^\dagger+\lambda I)^{-1}
$$

gives the equivalent solve

$$
(K+\lambda I)a=b,
\qquad
K=OO^\dagger,
\qquad
d=O^\dagger a.
$$

This form stores a matrix in sample-coordinate space rather than parameter
space. `psr(mu=0)` implements this sample-space SR equation.

## Predictive SR

For `mu > 0`, PSR predicts the next direction from the previous unscaled SR
direction $d_{t-1}$:

$$
p_t=\mu d_{t-1}.
$$

It removes the tangent response already explained by that predictor,

$$
r_t=b_t-O_tp_t,
$$

then solves and corrects in the current geometry:

$$
(K_t+\lambda I)a_t=r_t,
$$

$$
d_t=p_t+O_t^\dagger a_t.
$$

The optimizer state stores $d_t$ before downstream Optax transforms apply a
learning rate. Thus `mu` controls prediction in geometry space and does not act
as an additional learning-rate schedule.

## Optax composition

SR/PSR must precede the learning-rate transform:

```python
optimizer = optax.chain(
    psr(mu=0.95, shift=1.0e-3),
    optax.scale_by_learning_rate(5.0e-2),
)
```

A schedule is composed in the same way:

```python
rate = optax.linear_schedule(0.0, 5.0e-2, 100)
optimizer = optax.chain(
    sr(shift=1.0e-3),
    optax.scale_by_learning_rate(rate),
)
```

The VMC driver uses the standard update pattern with `geometry` passed as an
extra argument:

```python
updates, opt_state = optimizer.update(
    grad,
    opt_state,
    params,
    geometry=geometry,
)
params = optax.apply_updates(params, updates)
```

## Public parameters

```text
shift      diagonal regularization lambda
mode       dense or matrix-free parameter-space SR
maxiter    iteration limit for matrix-free SR
mu         predictive strength; zero selects sample-space SR
```

The estimator determines configurations, Born weights, and the residual.
Optax determines the learning rate and its schedule. This separation keeps SR
focused on the geometry solve.
