# Monte Carlo estimator

`MCSampler` advances auxiliary chains. `MCState` turns their observations into
Born-distribution estimates. Sampling may change coverage and variance, never
the variational objective.

## Probability laws

For $\ell_\theta(x)=\log|\psi_\theta(x)|$, keep three laws distinct:

$$
\rho_\theta(x)
\quad\text{source distribution sampled by the chains},
$$

$$
\nu_\theta(y|x)
\quad\text{observation kernel},
$$

$$
\pi_\theta(y)=
\frac{|\psi_\theta(y)|^2}{\sum_z|\psi_\theta(z)|^2}
\quad\text{variational target}.
$$

The distinction between $\rho$, $\nu$, and $\pi$ is the central estimator
contract.

## Source chains

For the Hamiltonian proposal, define

$$
b(x,y)=|H_{yx}|,
\qquad
d(x)=\sum_{y\ne x}b(x,y).
$$

A connected configuration is proposed with

$$
q_H(y|x)=\frac{b(x,y)}{d(x)},
$$

and the stationary source law is

$$
\rho_{\theta,\alpha}(x)
\propto s(x)e^{\alpha\ell_\theta(x)},
\qquad
s(x)=\begin{cases}d(x),&d(x)>0,\\1,&d(x)=0.\end{cases}
$$

The degree factor cancels proposal asymmetry. The `single` proposal instead
uses uniform single excitations and $s(x)=1$. `alpha=2` uses the Born amplitude
exponent; smaller values temper the source.

`ChainState` contains only persistent sampling state:

```text
key       random key
x         source configurations
logabs    log|psi(x)| at the current parameters
alpha     source tempering exponent
```

```text
thermal_steps    transitions whenever a sampler state is initialized
discard_steps    transitions before observations are collected in each draw
sweep_steps      transitions between successive collection rounds
```

Initial configurations are passed to `MCState.init`; the sampler does not
choose an initialization policy. Burn-in occurs once, then estimator calls
advance the persistent chains.

## Observation and reweighting

With blur probability $\beta$,

$$
\nu_\beta(y|x)
=(1-\beta)\delta_{xy}
+\beta\frac{b(x,y)}{d(x)}.
$$

The connected term is absent when $d(x)=0$.

The induced unnormalized observation density is

$$
r_{\theta,\alpha,\beta}(y)
=\sum_x s(x)e^{\alpha\ell_\theta(x)}\nu_\beta(y|x).
$$

Repeated observations are merged. If $M_y$ is their empirical mass, the Born
weight is

$$
\widetilde w_y=
M_y\frac{e^{2\ell_\theta(y)}}{r_{\theta,\alpha,\beta}(y)},
\qquad
w_y=\frac{\widetilde w_y}{\sum_z\widetilde w_z}.
$$

Energy, observables, gradients, and geometry all use the same normalized
`weight`.

## Local energy

For an observed ket $x$,

$$
E_{\mathrm{loc}}(x)
=\sum_y H_{yx}\frac{\psi_\theta(y)}{\psi_\theta(x)}.
$$

`Hamiltonian.local_conn` partitions the retained action:

```text
strong    |H_yx| >= eps1                 deterministic
weak      eps2 <= |H_yx| < eps1          sampled
```

Weak connections are sampled from equal-probability strata over the complete
unique-ket batch and carry their final unbiased coefficients. `eloc_sample` is
the total number of weak proposal targets. With `eps1=eps2=0`, the full action
is deterministic.

Weak sampling requires `eps2 > 0` and `eloc_sample > 0` when `eps2 < eps1`.

Kets and all connected bras share one `logpsi` evaluation pool. The final
energy, residual, gradient cotangent, and SR residual are

$$
E=\sum_x w_xE_{\mathrm{loc}}(x),
\qquad
R_x=E_{\mathrm{loc}}(x)-E,
$$

$$
2w_xR_x,
\qquad
2\sqrt{w_x}R_x.
$$

## Execution order

Every `MCState.expect` or `expect_and_grad` call follows one flow:

```text
1. synchronize chain log amplitudes
2. sample sources and observations
3. merge repeated observations
4. generate Hamiltonian and observable connections
5. evaluate the shared logpsi pool
6. assemble local quantities
7. reweight to the Born target
8. reduce energy, gradient, statistics, and Geometry
9. return the advanced chains and tempering state
```

## Stable development rules

- A proposal change must define its source law and transition ratio together.
- An observation kernel must provide the density needed for Born reweighting.
- Sampling ends at observations; estimation and differentiation stay in
  `MCState`.
- New observables reuse the shared configuration pool and normalized weights.
- `alpha` changes only the auxiliary source. `alpha=None` may adapt future
  chains, but never changes the target of the current estimate.
