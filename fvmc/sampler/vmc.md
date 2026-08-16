# Monte Carlo estimator

`MCSampler` and `MCState` provide the Born sampling path. `HamSampler` and
`IRState` provide Hamiltonian-guided importance resampling. Both states use a
semistochastic local energy. Sampling may change coverage and variance, never
the variational objective.

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

The distinction between $p$, $\rho$, $K$, and $r$ is the `IRState` estimator
contract.

## Born chains

`MCSampler` targets

$$
p_\alpha(x)\propto |\psi_\theta(x)|^\alpha.
$$

`rank=k` mixes excitation ranks $1,\ldots,k$ with geometric probabilities;
`rank=None` mixes every legal rank. A mapping selects exact ranks and their
probabilities, for example `rank={1: 0.25, 2: 0.5, 4: 0.25}`. Spin splits are
weighted by their number of legal determinants. The resulting proposal is
symmetric, so

$$
A(x\to y)=\min\!\left(1,e^{\alpha[\ell_\theta(y)-\ell_\theta(x)]}\right).
$$

After repeated samples are merged, `MCState` uses

$$
w_x\propto N_x|\psi_\theta(x)|^{2-\alpha}
$$

and evaluates the same semistochastic local energy as `IRState`.

## Hamiltonian chains

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

The MH chain must be ergodic and leave $\rho_{\theta,\alpha}$ invariant.
Smaller `alpha` tempers the auxiliary distribution.

`ChainState` contains only persistent sampling state:

```text
key       random key
x         auxiliary configurations
logabs    log|psi(x)| at the current parameters
```

```text
thermal_steps    transitions during initialization
discard_steps    transitions between observation rounds
```

Initial configurations are passed to `MCState.init` or `IRState.init`; the
sampler does not choose an initialization policy. Thermalization occurs once,
then estimator calls advance the persistent chains. `alpha` belongs to the
State; `alpha=None` enables adaptation.

## Markov kernel and importance weights

The Markov kernel is

$$
K_\beta(y|x)
=(1-\beta)\delta_{xy}
+\beta q_{\varepsilon_1}(y|x).
$$

The proposal term is absent when $d_{\varepsilon_1}(x)=0$.
The same Hamiltonian proposal supplies both the observation and MH transition.

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

Repeated outputs are merged into integer multiplicities. Energy, observables,
gradients, and geometry use the same normalized importance weights.

## Local energy

For an observed ket $x$,

$$
E_{\mathrm{loc}}(x)
=\sum_y H_{yx}\frac{\psi_\theta(y)}{\psi_\theta(x)}.
$$

Both Monte Carlo states use `Hamiltonian.local_conn`, which partitions the
retained action:

```text
strong    |H_yx| >= eps1                 deterministic
weak      eps2 <= |H_yx| < eps1          sampled
```

Weak connections are sampled from equal-probability strata and carry their
final unbiased coefficients. With `eps1=eps2=0`, the full action is deterministic.

Weak sampling requires `eps2 > 0` and `n_eloc > 0` when `eps2 < eps1`.
`n_eloc` is the number of weak samples per unique outer ket.
The defaults are `eps1=1e-3`, `eps2=1e-12`, and `n_eloc=1024`.

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

Every `MCState.expect` or `IRState.expect` call follows one flow:

```text
1. synchronize chain log amplitudes
2. draw outer samples
3. merge repeated outputs
4. generate Hamiltonian and observable connections
5. evaluate the shared logpsi pool
6. assemble local quantities
7. construct normalized importance weights
8. evaluate self-normalized importance estimators and Geometry
9. return the advanced chains and adaptive state
```

## Stable development rules

- A proposal distribution must define its auxiliary distribution and acceptance
  probability together.
- A Markov kernel must provide its induced distribution.
- Sampling ends at raw observations; estimation and differentiation stay in
  the State.
- New observables reuse the shared configuration pool and normalized importance
  weights.
- `alpha` changes only the sampling distribution. `alpha=None` may adapt future
  chains, but never changes the Born objective.
