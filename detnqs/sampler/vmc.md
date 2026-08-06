# Monte Carlo estimator

`MCSampler` and `MCState` implement the stochastic VMC path. The central design
rule is that auxiliary sampling may change coverage and variance, but the
reported energy and gradient always target the Born distribution of the model.

## Probability laws

Let $x$ and $y$ be configurations in a fixed sector and define

$$
\ell_\theta(x)=\log|\psi_\theta(x)|.
$$

The estimator keeps three probability objects distinct:

$$
\rho_\theta(x)
\quad\text{source law sampled by persistent chains},
$$

$$
\nu(y|x)
\quad\text{observation kernel applied to a source},
$$

$$
\pi_\theta(y)
=\frac{|\psi_\theta(y)|^2}{\sum_z|\psi_\theta(z)|^2}
\quad\text{Born target of the variational objective}.
$$

`MCSampler` advances the source chains and draws observations. `MCState`
evaluates the induced observation density and importance-reweights it to
$\pi_\theta$.

## Source sampling

The Hamiltonian proposal draws a connected bra with probability proportional
to the matrix-element magnitude. Define

$$
b(x,y)=|H_{yx}|,
\qquad
d(x)=\sum_{y\ne x}b(x,y).
$$

Then

$$
q_H(y|x)=\frac{b(x,y)}{d(x)},
$$

and the corresponding source law is

$$
\rho_{\theta,\alpha}(x)
\propto s(x)e^{\alpha\ell_\theta(x)},
\qquad
s(x)=\begin{cases}d(x),&d(x)>0,\\1,&d(x)=0.\end{cases}
$$

The degree factor cancels the proposal asymmetry in the Metropolis ratio. The
alternative `single` proposal uses a uniformly chosen single excitation and an
untilted source factor $s(x)=1$.

`ChainState` contains the persistent sampling state:

```text
key       random key
x         current source configurations
logabs    log|psi(x)| synchronized with params
alpha     source tempering exponent
```

Burn-in initializes this state. Subsequent estimator calls advance it instead
of restarting the chains.

## Observation blur

The observation may remain at its source or move along one Hamiltonian
connection. With blur probability $\beta$,

$$
\nu_\beta(y|x)
=(1-\beta)\delta_{xy}
+\beta\frac{b(x,y)}{d(x)}.
$$

Blur changes which configurations are evaluated without changing the source
chain transition. Its induced unnormalized observation density is

$$
r_{\theta,\alpha,\beta}(y)
=\sum_x s(x)e^{\alpha\ell_\theta(x)}\nu_\beta(y|x).
$$

Only density ratios are required, so the normalization of $r$ is never formed.

## Merge and Born reweighting

Repeated observations are merged before Hamiltonian action. If $M_y$ is the
empirical mass assigned to a unique observed configuration, its unnormalized
Born weight is

$$
\widetilde w_y
=M_y\frac{e^{2\ell_\theta(y)}}{r_{\theta,\alpha,\beta}(y)}.
$$

The normalized estimator weight is

$$
w_y=\frac{\widetilde w_y}{\sum_z\widetilde w_z}.
$$

In code, source mass, observation density, and normalized Born `weight` are
separate arrays. Energy, observables, gradients, and geometry all use the same
Born weights.

## Local energy

For an observed ket $x$,

$$
E_{\mathrm{loc}}(x)
=\sum_y H_{yx}\frac{\psi_\theta(y)}{\psi_\theta(x)}.
$$

`Hamiltonian.local_conn` divides the off-diagonal action into a deterministic
strong region and a sampled weak region:

```text
strong    |H_yx| >= eps1
weak      eps2 <= |H_yx| < eps1
```

Weak records carry the degree and multiplicity needed for an unbiased estimate
of the retained window. Setting both cutoffs to zero makes the full action
deterministic.

All observed kets, Hamiltonian-connected bras, and observable-connected bras
are merged into one evaluation pool. The model is evaluated once on this pool,
and every local ratio reuses those values.

The final energy and residual are

$$
E=\sum_xw_xE_{\mathrm{loc}}(x),
\qquad
R_x=E_{\mathrm{loc}}(x)-E.
$$

The energy-gradient cotangent is built from $2w_xR_x$. SR geometry uses the
same configurations and weights, with sample-space residual
$2\sqrt{w_x}R_x$.

## Execution flow

One `MCState.expect` or `expect_and_grad` call follows this order:

```text
1. synchronize source-chain log amplitudes with the current parameters
2. sample source configurations from rho and observations through nu
3. merge repeated observation configurations and their empirical mass
4. generate Hamiltonian and observable connections
5. evaluate logpsi once for the shared configuration pool
6. assemble local energies and local observables
7. reweight the observation law to the Born target pi
8. reduce energy, statistics, gradient, and optional Geometry
9. return the advanced ChainState and updated tempering exponent
```

This order is the core Monte Carlo algorithm. Sampling stops at observations;
the variational state owns density evaluation, reweighting, differentiation,
and geometry construction.

## Tempering and state updates

`alpha` controls the concentration of the source law. `alpha=2` uses the Born
amplitude exponent, while smaller values broaden source coverage. With
`alpha=None`, `MCState` adapts the exponent for the next estimator call.

Tempering affects only the auxiliary source. The current estimate is always
Born-reweighted. The returned `MCState` therefore carries updated chain and
tempering state while leaving the variational objective unchanged.
