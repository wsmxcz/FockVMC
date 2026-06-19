# Fock-Space Variational Monte Carlo

This note summarizes the probability structure behind Fock-space variational
Monte Carlo (VMC). The main point is to separate the physical Born measure from
the auxiliary laws used for numerical sampling. Markov chains, broadened
reference laws, observation kernels, importance weights, and stochastic
local-energy estimators change how an expectation is estimated; they do not
change the variational objective.

The setting is a discrete Fock-space sector $\mathcal X$, such as a
fixed-particle-number or fixed-spin sector.

## 1. Born Measure and Observables

Let $\psi_\theta(x)$ be a variational amplitude. The energy is

$$
E(\theta)=
\frac{\langle\psi_\theta|H|\psi_\theta\rangle}
{\langle\psi_\theta|\psi_\theta\rangle}.
$$

The physical probability law is the Born measure

$$
\pi_\theta(x)=
\frac{|\psi_\theta(x)|^2}{Z_2},
\qquad
Z_2=\sum_{z\in\mathcal X}|\psi_\theta(z)|^2.
$$

For an observable $f_\theta(x)$,

$$
\langle f\rangle_\pi
=
\sum_{x\in\mathcal X}\pi_\theta(x)f_\theta(x).
$$

The local energy is

$$
E_{\mathrm{loc}}(x)
=
\frac{(H\psi_\theta)(x)}{\psi_\theta(x)}
=
\sum_{y:H_{xy}\ne 0}
H_{xy}\frac{\psi_\theta(y)}{\psi_\theta(x)},
$$

and therefore

$$
E(\theta)=\langle E_{\mathrm{loc}}\rangle_\pi.
$$

Gradient and geometry estimators have the same basic form: they are
Born-measure averages of local quantities involving the local energy residual
and logarithmic derivatives.

## 2. Reference Laws and Markov Chains

A simulation may sample from a reference law broader than the Born measure,

$$
\eta_{\theta,\alpha}(x)
=
\frac{|\psi_\theta(x)|^\alpha}{Z_\alpha},
\qquad
Z_\alpha=\sum_{z\in\mathcal X}|\psi_\theta(z)|^\alpha.
$$

Here $\alpha=2$ gives direct Born sampling, while $0\le\alpha<2$ gives a
broader law. The numerical chain may target $\eta_{\theta,\alpha}$, but the
reported quantity remains a Born-measure expectation.

Let $q_A(y|x)$ be a proposal kernel. A Metropolis-Hastings transition targeting
$\eta_{\theta,\alpha}$ accepts

$$
A(x\to y)
=
\min\left\{
1,
\frac{|\psi_\theta(y)|^\alpha q_A(x|y)}
{|\psi_\theta(x)|^\alpha q_A(y|x)}
\right\}.
$$

The proposal geometry controls exploration and autocorrelation. It may encode
local particle moves or Hamiltonian-informed screened connections, but it does
not define the physical measure.

## 3. Observation Kernels

The configuration entering the estimator may be drawn from an observation
kernel $B(y|x)$ after the Markov state $x$ is generated:

$$
\sum_{y\in\mathcal X}B(y|x)=1.
$$

If $x\sim\eta_{\theta,\alpha}$ and $y\sim B(\cdot|x)$, the observed law is

$$
\nu(y)=
\sum_{x\in\mathcal X}\eta_{\theta,\alpha}(x)B(y|x).
$$

Equivalently, using an unnormalized density,

$$
r_\nu(y)=
\sum_{x\in\mathcal X}|\psi_\theta(x)|^\alpha B(y|x),
\qquad
\nu(y)=\frac{r_\nu(y)}{Z_\alpha}.
$$

The identity observation kernel gives $\nu=\eta_{\theta,\alpha}$. A blurred
observation kernel may mix the current configuration with Hamiltonian-connected
neighbors. This can improve support for local-energy and gradient estimators,
while leaving the Born objective unchanged.

## 4. Reweighting

When samples are distributed according to $\nu$ but the target is $\pi_\theta$,
use the unnormalized importance weight

$$
\omega(y)=
\frac{|\psi_\theta(y)|^2}{r_\nu(y)}.
$$

The self-normalized estimator is

$$
\widehat{\langle f\rangle}_\pi
=
\frac{\sum_{k=1}^N\omega(y_k)f_\theta(y_k)}
{\sum_{k=1}^N\omega(y_k)}.
$$

For identity observation, $r_\nu(y)=|\psi_\theta(y)|^\alpha$, so

$$
\omega(y)=|\psi_\theta(y)|^{2-\alpha}.
$$

For a nontrivial observation kernel, $r_\nu$ includes the probability mass
transported to $y$ by that kernel.

## 5. Support, Tails, and ESS

Reweighting requires the observed law to cover the relevant target
contributions. For an observable $f_\theta$,

$$
\pi_\theta(y)|f_\theta(y)|>0
\quad\Rightarrow\quad
\nu(y)>0.
$$

This condition is observable-dependent. Energy, gradient, and geometry
estimators involve different local factors and may have different stability
requirements.

Even when support is adequate, weighted observables can have heavy tails,
especially near zeros of $\psi_\theta$. A useful diagnostic is the effective
sample size

$$
\mathrm{ESS}
=
\frac{\left(\sum_{k=1}^N\omega_k\right)^2}
{\sum_{k=1}^N\omega_k^2}.
$$

ESS diagnoses weight concentration, not Markov-chain mixing. A chain may mix
well while producing poor weights, or mix slowly while weights are benign.

## 6. Local-Energy Estimation

Local energy evaluation can itself be expensive. If the Hamiltonian row is not
summed exactly, write a stochastic estimator as

$$
\widehat E_{\mathrm{loc}}(y,\xi),
$$

where $\xi$ denotes auxiliary sampling. A natural correctness condition is
conditional unbiasedness:

$$
\mathbb E_{\xi|y}
\left[
\widehat E_{\mathrm{loc}}(y,\xi)
\right]
=
E_{\mathrm{loc}}(y).
$$

The corresponding reweighted energy estimator is

$$
\widehat E
=
\frac{\sum_{k=1}^N
\omega(y_k)\widehat E_{\mathrm{loc}}(y_k,\xi_k)}
{\sum_{k=1}^N\omega(y_k)}.
$$

This separates configuration-sampling error from Hamiltonian-connection
estimation error. In screened Hamiltonian workflows, deterministic strong
connections and sampled weak connections are two parts of the same row-sum
estimator.

## 7. Validation

A VMC sampling specification is described by

$$
\pi_\theta,\qquad
\eta_{\theta,\alpha},\qquad
q_A\text{ or }T,\qquad
B,\qquad
\omega=\frac{|\psi_\theta|^2}{r_\nu}.
$$

Useful diagnostics include:

- effective sample size and weight concentration;
- acceptance rate and autocorrelation;
- number of unique observed configurations;
- stability of energy, gradient, and geometry estimates;
- number of wave-function evaluations;
- cost of Hamiltonian connections or local-energy sampling.

These diagnostics distinguish support mismatch, heavy-tailed weights, slow
mixing, and expensive Hamiltonian-row evaluation. They interact in practice,
but they are conceptually different and should be examined separately.

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
