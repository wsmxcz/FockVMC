# Monte Carlo estimator

`MCSampler` advances chains for an auxiliary distribution. `MCState` constructs
a self-normalized importance estimator for the Born distribution. Sampling may
change coverage and variance, never the variational objective.

## Distributions and kernels

For $\ell_\theta(x)=\log|\psi_\theta(x)|$, keep four objects distinct:

$$
p_\theta(x)=
\frac{|\psi_\theta(x)|^2}{\sum_z|\psi_\theta(z)|^2}
\quad\text{Born distribution},
$$

$$
\rho_\theta(x)
\quad\text{auxiliary distribution},
$$

$$
K_\theta(y|x)
\quad\text{Markov kernel},
$$

$$
r_\theta(y)=\sum_x\rho_\theta(x)K_\theta(y|x)
\quad\text{induced distribution}.
$$

The distinction between $p$, $\rho$, $K$, and $r$ is the estimator contract.

## Auxiliary chains

For cutoff $\varepsilon_1$, define

$$
b_{\varepsilon_1}(x,y)
=|H_{yx}|\,\mathbf 1(|H_{yx}|\ge\varepsilon_1),
\qquad
d_{\varepsilon_1}(x)
=\sum_{y\ne x}b_{\varepsilon_1}(x,y).
$$

The Hamiltonian proposal distribution is

$$
q_{\varepsilon_1}(y|x)
=\frac{b_{\varepsilon_1}(x,y)}{d_{\varepsilon_1}(x)}.
$$

The MH transition leaves the following auxiliary distribution invariant:

$$
\rho_{\theta,\alpha}(x)
\propto s(x)e^{\alpha\ell_\theta(x)},
\qquad
s(x)=\begin{cases}
d_{\varepsilon_1}(x),&d_{\varepsilon_1}(x)>0,\\
1,&d_{\varepsilon_1}(x)=0.
\end{cases}
$$

The degree factor cancels proposal asymmetry, giving acceptance probability

$$
A(x\to y)
=\min\!\left(1,e^{\alpha[\ell_\theta(y)-\ell_\theta(x)]}\right).
$$

The MH chain must be ergodic and leave $\rho_{\theta,\alpha}$ invariant. The
`single` proposal is a Born distribution baseline restricted to `alpha=2` and
`beta=0`. For Hamiltonian proposals, smaller `alpha` tempers the auxiliary
distribution.

`ChainState` contains only persistent sampling state:

```text
key       random key
x         auxiliary configurations
logabs    log|psi(x)| at the current parameters
alpha     auxiliary tempering exponent
```

```text
burn_in          transitions whenever a sampler state is initialized
discard          transitions before observations are collected in each draw
sweep            transitions between successive collection rounds
```

Initial configurations are passed to `MCState.init`; the sampler does not
choose an initialization policy. Burn-in occurs once, then estimator calls
advance the persistent chains.

## Markov kernel and importance weights

The Markov kernel is

$$
K_\beta(y|x)
=(1-\beta)\delta_{xy}
+\beta q_{\varepsilon_1}(y|x).
$$

The proposal term is absent when $d_{\varepsilon_1}(x)=0$.

Using the unnormalized form of the auxiliary distribution, the induced
distribution is proportional to

$$
r_{\theta,\alpha,\beta}(y)
\propto\sum_x s(x)e^{\alpha\ell_\theta(x)}K_\beta(y|x).
$$

For an output $y_n$, define the unnormalized importance weight and normalized
importance weight

$$
\omega_{\theta,n}
=\frac{e^{2\ell_\theta(y_n)}}{r_{\theta,\alpha,\beta}(y_n)},
\qquad
w_n=\frac{\omega_{\theta,n}}{\sum_m\omega_{\theta,m}}.
$$

The self-normalized importance estimator is

$$
\widehat\mu_f=\sum_n w_n f(y_n).
$$

Repeated outputs are merged by summing their empirical mass. Energy,
observables, gradients, and geometry use the same normalized importance
weights.

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

Weak connections are sampled from equal-probability strata and carry their
final unbiased coefficients. With `eps1=eps2=0`, the full action is deterministic.

Weak sampling requires `eps2 > 0` and `n_eloc > 0` when `eps2 < eps1`.

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
2. advance auxiliary chains and apply the Markov kernel
3. merge repeated outputs
4. generate Hamiltonian and observable connections
5. evaluate the shared logpsi pool
6. assemble local quantities
7. construct normalized importance weights
8. evaluate self-normalized importance estimators and Geometry
9. return the advanced chains and tempering state
```

## Stable development rules

- A proposal distribution must define its auxiliary distribution and acceptance
  probability together.
- A Markov kernel must provide its induced distribution.
- Sampling ends at Markov-kernel outputs; estimation and differentiation stay
  in `MCState`.
- New observables reuse the shared configuration pool and normalized importance
  weights.
- `alpha` changes only the auxiliary distribution. `alpha=None` may adapt future
  chains, but never changes the Born distribution.
