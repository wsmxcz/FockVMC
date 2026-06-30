# Fock-Space Variational Monte Carlo

This note summarizes the probability structure of Fock-space variational Monte
Carlo (VMC). The main distinction is between the physical Born measure, which
is determined by the wave function, and auxiliary probability laws, which are
introduced to estimate Born-measure averages efficiently.

The setting is a finite Fock-space sector $\mathcal X$. It may be specified by
particle number, spin projection, spatial symmetry, spin adaptation, or any
other constraints used to define the variational problem. A variational state
assigns amplitudes $\psi_\theta(x)$ to configurations $x\in\mathcal X$. We use

$$
\ell_\theta(x)=\log |\psi_\theta(x)|
$$

for the log-amplitude. Markov chains, Hamiltonian-informed proposals, degree
tilting, blurred observations, importance weights, and stochastic local-energy
estimators all belong to the estimation procedure. They do not change the
Rayleigh quotient being minimized.

## 1. Born Measure and VMC Objective

The variational energy is the Rayleigh quotient

$$
E(\theta)
=
\frac{\langle \psi_\theta|H|\psi_\theta\rangle}
{\langle \psi_\theta|\psi_\theta\rangle}.
$$

Define the Born measure

$$
\pi_\theta(x)
=
\frac{|\psi_\theta(x)|^2}{Z_2},
\qquad
Z_2=\sum_{z\in\mathcal X}|\psi_\theta(z)|^2 .
$$

For any local quantity $f_\theta(x)$, the corresponding expectation is

$$
\langle f\rangle_{\pi_\theta}
=
\sum_{x\in\mathcal X}\pi_\theta(x)f_\theta(x).
$$

The local energy is

$$
E_{\mathrm{loc},\theta}(x)
=
\frac{(H\psi_\theta)(x)}{\psi_\theta(x)}
=
\sum_{y:H_{xy}\ne0}
H_{xy}\frac{\psi_\theta(y)}{\psi_\theta(x)}.
$$

Thus

$$
E(\theta)
=
\mathbb E_{\pi_\theta}
\left[E_{\mathrm{loc},\theta}\right].
$$

The same probability measure appears in gradients and stochastic
reconfiguration. If

$$
O_\theta(x)=\nabla_\theta\log\psi_\theta(x),
$$

then the standard VMC gradient may be written as

$$
\nabla_\theta E
=
2\,\mathrm{Re}\,
\mathbb E_{\pi_\theta}
\left[
\left(E_{\mathrm{loc},\theta}-E\right)O_\theta^*
\right].
$$

The central object is therefore the Born measure $\pi_\theta$. Auxiliary laws
are introduced only to estimate its expectations.

## 2. Auxiliary Source Laws and Markov Chains

Direct sampling from $\pi_\theta$ is not required. One may instead sample from
an auxiliary source law

$$
\eta_{\theta,\alpha}(x)
=
\frac{a(x)e^{\alpha\ell_\theta(x)}}{Z_\eta},
\qquad
Z_\eta=
\sum_{z\in\mathcal X}a(z)e^{\alpha\ell_\theta(z)}.
$$

Here $a(x)\ge0$ is a base measure and $\alpha\in[0,2]$ controls the
concentration of the auxiliary distribution. When $a(x)=1$, the endpoint
$\alpha=2$ is Born-like, while smaller values of $\alpha$ give broader sampling
laws.

A Markov chain with proposal $q(y|x)$ may be constructed to leave
$\eta_{\theta,\alpha}$ invariant. The Metropolis-Hastings acceptance probability
is

$$
A(x\to y)
=
\min\left\{
1,
\frac{a(y)e^{\alpha\ell_\theta(y)}q(x|y)}
{a(x)e^{\alpha\ell_\theta(x)}q(y|x)}
\right\}.
$$

The proposal $q$ controls the movement of the chain, the autocorrelation time,
and the cost of one transition. The invariant law is an auxiliary law used for
estimation; it is not the physical Born measure unless that choice is made
explicitly.

Hamiltonian-informed moves use the sparse connectivity induced by matrix
elements. Let

$$
b(x,y)\ge0,
\qquad
b(x,x)=0,
\qquad
d(x)=\sum_y b(x,y).
$$

The corresponding degree factor is

$$
s(x)=
\begin{cases}
d(x),& d(x)>0,\\
1,& d(x)=0.
\end{cases}
$$

For $d(x)>0$, a heat-bath proposal along Hamiltonian connections is

$$
q_H(y|x)=\frac{b(x,y)}{d(x)}.
$$

If $b(x,y)$ is symmetric and the source law uses $a(x)=s(x)$, the degree factors
in the proposal and in the auxiliary source law cancel in the acceptance ratio.
For connected non-isolated configurations,

$$
\frac{s(y)e^{\alpha\ell_\theta(y)}q_H(x|y)}
{s(x)e^{\alpha\ell_\theta(x)}q_H(y|x)}
=
e^{\alpha(\ell_\theta(y)-\ell_\theta(x))}.
$$

This is the role of degree tilting. It is a choice of auxiliary source measure
that makes Hamiltonian-informed transitions depend locally on amplitude ratios
rather than explicit degree corrections. The Born objective is unchanged.

## 3. Observation Laws and Born Reweighting

The configuration used in an estimator need not be the same as the source state
of the chain. Given a source configuration $x$, introduce an observation kernel

$$
B_\beta(y|x)\ge0,
\qquad
\sum_y B_\beta(y|x)=1.
$$

The unnormalized observed auxiliary density is

$$
r_{\theta,\alpha,\beta}(y)
=
\sum_{x\in\mathcal X}
a(x)e^{\alpha\ell_\theta(x)}B_\beta(y|x).
$$

The normalized observed law is

$$
\nu_{\theta,\alpha,\beta}(y)
=
\frac{r_{\theta,\alpha,\beta}(y)}{Z_\eta}.
$$

Identity observation corresponds to $B_0(y|x)=\delta_{xy}$. Blurred observation
moves part of the source mass to nearby configurations, often using the same
Hamiltonian connectivity as the proposal. This changes the law seen by the
estimator, not the Born measure.

Born expectations are recovered by importance reweighting. The unnormalized
weight is

$$
\omega_{\theta,\alpha,\beta}(y)
=
\frac{|\psi_\theta(y)|^2}{r_{\theta,\alpha,\beta}(y)}
=
\frac{e^{2\ell_\theta(y)}}{r_{\theta,\alpha,\beta}(y)}.
$$

For observed configurations with empirical mass $M_y$, the self-normalized
estimator is

$$
\widehat{\mathbb E}_{\pi_\theta}[f]
=
\frac{\sum_y M_y\omega_y f_\theta(y)}
{\sum_y M_y\omega_y}.
$$

The energy estimator is the special case

$$
\widehat E
=
\frac{\sum_y M_y\omega_yE_{\mathrm{loc},\theta}(y)}
{\sum_y M_y\omega_y}.
$$

With normalized weights

$$
w_y
=
\frac{M_y\omega_y}{\sum_zM_z\omega_z},
$$

this becomes

$$
\widehat E
=
\sum_y w_yE_{\mathrm{loc},\theta}(y).
$$

The gradient estimator has the analogous form

$$
\widehat{\nabla_\theta E}
=
2\,\mathrm{Re}\,
\sum_yw_y
\left(E_{\mathrm{loc},\theta}(y)-\widehat E\right)O_\theta(y)^*.
$$

All auxiliary choices enter the estimator through
$r_{\theta,\alpha,\beta}$. Once this density is known, the calculation is an
ordinary self-normalized importance-sampling estimate of a Born-measure
expectation.

## 4. Hamiltonian Blur and Local-Energy Estimation

Hamiltonian blur is most naturally described as part of the observation law.
For the connection weights $b(x,y)$, a typical blurred kernel is

$$
B_\beta(y|x)
=
(1-\beta)\delta_{xy}
+
\beta\frac{b(x,y)}{d(x)},
\qquad d(x)>0,
$$

with identity observation for isolated configurations.

Combining this kernel with the degree-tilted source law $a(x)=s(x)$ gives

$$
r_{\theta,\alpha,\beta}(y)
=
c_\beta(y)e^{\alpha\ell_\theta(y)}
+
\beta\sum_{x\ne y}b(x,y)e^{\alpha\ell_\theta(x)},
$$

where

$$
c_\beta(y)=
\begin{cases}
(1-\beta)d(y),& d(y)>0,\\
1,& d(y)=0.
\end{cases}
$$

This expression is the observed auxiliary density associated with
Hamiltonian-informed proposal structure, degree tilting, and blurred
observation. It is the quantity required for Born reweighting.

The same Hamiltonian connectivity may also be used to estimate local energies.
If part of the Hamiltonian action is sampled using an auxiliary random variable
$\xi$, the natural condition is

$$
\mathbb E_{\xi|y}
\left[
\widehat E_{\mathrm{loc},\theta}(y,\xi)
\right]
=
E_{\mathrm{loc},\theta}(y).
$$

The corresponding estimator is

$$
\widehat E
=
\frac{\sum_yM_y\omega_y\widehat E_{\mathrm{loc},\theta}(y,\xi_y)}
{\sum_yM_y\omega_y}.
$$

This separates two sources of stochastic error. The empirical distribution of
observed configurations controls sampling error, while the conditional
estimator of the Hamiltonian action controls local-energy noise. Both affect
finite-sample behavior, but they are distinct components of the VMC estimator.

## 5. Adaptive Auxiliary Exponent

The exponent $\alpha$ is not a variational parameter. It selects a member of the
observed auxiliary family

$$
\nu_{\theta,\alpha,\beta},
\qquad 0\le\alpha\le2.
$$

Large values of $\alpha$ produce sharper, Born-like auxiliary laws. Smaller
values produce broader laws, which may improve exploration but increase the
importance-reweighting burden. The useful value of $\alpha$ depends on the
current wave function, the observation law, and the local-energy fluctuations.

For estimating the energy, the variance-relevant importance law is

$$
\nu_\star(y)
\propto
\pi_\theta(y)
\left|E_{\mathrm{loc},\theta}(y)-E(\theta)\right|.
$$

It gives more probability to configurations that contribute strongly to the
fluctuation of the energy estimator. A one-dimensional adaptive choice of
$\alpha$ can be defined by the projection

$$
\alpha^\star
=
\arg\min_{\alpha\in[0,2]}
D_{\mathrm{KL}}
\left(
\nu_\star
\Vert
\nu_{\theta,\alpha,\beta}
\right).
$$

Let

$$
S_\alpha(y)
=
\partial_\alpha\log r_{\theta,\alpha,\beta}(y).
$$

Writing

$$
r_{\theta,\alpha,\beta}(y)
=
\sum_x A_\beta(y,x)e^{\alpha\ell_\theta(x)},
\qquad
A_\beta(y,x)=a(x)B_\beta(y|x),
$$

gives

$$
S_\alpha(y)
=
\frac{\sum_xA_\beta(y,x)e^{\alpha\ell_\theta(x)}\ell_\theta(x)}
{\sum_xA_\beta(y,x)e^{\alpha\ell_\theta(x)}}.
$$

Equivalently, $S_\alpha(y)$ is the conditional mean of the source
log-amplitude given the observed configuration $y$ under the auxiliary joint
law. The first-order condition for the KL projection is

$$
\mathbb E_{\nu_\star}[S_\alpha]
=
\mathbb E_{\nu_{\theta,\alpha,\beta}}[S_\alpha].
$$

For identity observation, $S_\alpha(y)=\ell_\theta(y)$, and the projection
matches the mean log-amplitude of the residual-weighted Born law to that of the
auxiliary observed law.

The projection should be applied on the outer VMC timescale. During a sampling
phase, $\theta$, $\alpha$, and $\beta$ define a fixed observed law. After the
corresponding estimates have been formed, the exponent may be updated for the
next phase.

Since $\theta$ changes during optimization, the projected exponent is itself a
moving quantity. A simple stable update is local tracking with a bounded step:

$$
\alpha_{t+1}
=
\Pi_{[0,2]}
\left[
\alpha_t+
\operatorname{clip}(\widehat\alpha_t-\alpha_t,-\delta_\alpha,\delta_\alpha)
\right].
$$

Here $\widehat\alpha_t$ is the current moment projection and $\delta_\alpha$ is
a small step bound. The purpose of the bound is to keep the auxiliary observed
measure from moving abruptly between successive sampling phases. Because the
step is symmetric, the exponent can move back when the projected direction
changes.

Effective sample size and acceptance rate are diagnostics of different parts
of the estimator. The effective sample size reflects the concentration of Born
weights. The acceptance rate reflects movement of the Markov chain. They are
useful for interpreting finite-sample behavior, but they do not define the
adaptive objective.

## References

1. N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and
   E. Teller, "Equation of State Calculations by Fast Computing Machines,"
   *Journal of Chemical Physics* **21**, 1087--1092 (1953).
   https://doi.org/10.1063/1.1699114
2. W. K. Hastings, "Monte Carlo Sampling Methods Using Markov Chains and Their
   Applications," *Biometrika* **57**, 97--109 (1970).
   https://doi.org/10.1093/biomet/57.1.97
3. W. M. C. Foulkes, L. Mitas, R. J. Needs, and G. Rajagopal, "Quantum Monte
   Carlo Simulations of Solids," *Reviews of Modern Physics* **73**, 33--83
   (2001). https://doi.org/10.1103/RevModPhys.73.33
4. S. Sorella, "Green Function Monte Carlo with Stochastic Reconfiguration,"
   *Physical Review Letters* **80**, 4558--4561 (1998).
   https://doi.org/10.1103/PhysRevLett.80.4558
5. C. J. Umrigar, J. Toulouse, C. Filippi, S. Sorella, and R. G. Hennig,
   "Alleviation of the Fermion-Sign Problem by Optimization of Many-Body Wave
   Functions," *Physical Review Letters* **98**, 110201 (2007).
   https://doi.org/10.1103/PhysRevLett.98.110201
6. J. R. Trail, "Alternative Sampling for Variational Quantum Monte Carlo,"
   *Physical Review E* **77**, 016704 (2008).
   https://doi.org/10.1103/PhysRevE.77.016704
7. A. Owen, *Monte Carlo Theory, Methods and Examples*, Chapter 9:
   Importance Sampling (2013). https://artowen.su.domains/mc/
8. A. Kong, J. S. Liu, and W. H. Wong, "Sequential Imputations and Bayesian
   Missing Data Problems," *Journal of the American Statistical Association*
   **89**, 278--288 (1994).
   https://doi.org/10.1080/01621459.1994.10476469
9. NetKet Documentation, "Quantum Geometric Tensor and Stochastic
   Reconfiguration." https://netket.readthedocs.io/en/latest/user-guides/sr.html
10. NetKet Documentation, "The Sampler Module."
    https://netket.readthedocs.io/en/latest/user-guides/sampler.html
11. M. Medvidovic and J. Robledo Moreno, "Neural-Network Quantum States for
    Many-Body Physics," arXiv:2402.11014 (2024).
    https://arxiv.org/abs/2402.11014
12. A. Misery, F. Vicentini, and G. Carleo, "Looking Elsewhere: Improving
    Variational Monte Carlo Gradients by Importance Sampling,"
    arXiv:2507.05352 (2025). https://arxiv.org/abs/2507.05352
13. Z.-Q. Wan, R. Wiersema, and S. Zhang, "Removing Nodal and
    Support-Mismatch Pathologies in Variational Monte Carlo via Blurred
    Sampling," arXiv:2603.18148 (2026). https://arxiv.org/abs/2603.18148
