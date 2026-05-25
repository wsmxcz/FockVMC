# Fock-Space Variational Monte Carlo: Measures, Reweighting, and Validation

## Overview

This note gives a compact framework for sampling in Fock-space variational Monte Carlo (VMC). The main distinction is between the physical probability law defined by the variational wave function and the auxiliary laws used to generate numerical samples. The physical objective is an expectation under the Born distribution. Markov chains, alternative reference measures, observation kernels, importance weights, and stochastic local-energy estimators change how the expectation is estimated, not what the expectation means.

The discussion is written for a discrete symmetry sector of Fock space, such as a fixed-particle-number or fixed-spin sector. This setting covers common occupation-number and determinant-basis formulations and allows the relevant probability statements to be written as finite or countable sums.

## 1. Born Measure and Variational Observables

Let $\mathcal X$ be the set of configurations in the chosen Fock-space sector, and let $\psi_\theta(x)$ be a variational amplitude. The variational energy is the Rayleigh quotient

$$
E(\theta)=
\frac{\langle \psi_\theta|H|\psi_\theta\rangle}
{\langle \psi_\theta|\psi_\theta\rangle}.
$$

The physical probability distribution associated with the state is the Born measure

$$
\pi_\theta(x)=\frac{|\psi_\theta(x)|^2}{Z_2},
\qquad
Z_2=\sum_{z\in\mathcal X}|\psi_\theta(z)|^2.
$$

For a configuration-dependent quantity $f_\theta(x)$, the physical expectation is

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
\sum_{y:H_{xy}\neq 0}
H_{xy}\frac{\psi_\theta(y)}{\psi_\theta(x)},
$$

on configurations where the ratio is well defined. The VMC energy identity is then

$$
E(\theta)=\langle E_{\mathrm{loc}}\rangle_\pi.
$$

This identity fixes the physical measure of the problem: the energy is a Born-measure expectation. The same measure also appears in variational optimization. If

$$
O_i(x)=\partial_{\theta_i}\log\psi_\theta(x),
$$

then the force used in stochastic reconfiguration and the quantum geometric tensor can be written, up to conventional real-part and factor conventions, as covariance-type quantities under $\pi_\theta$:

$$
F_i=
\langle O_i^*E_{\mathrm{loc}}\rangle_\pi
-
\langle O_i^*\rangle_\pi\langle E_{\mathrm{loc}}\rangle_\pi,
$$

$$
S_{ij}=
\langle O_i^*O_j\rangle_\pi
-
\langle O_i^*\rangle_\pi\langle O_j\rangle_\pi.
$$

Thus energy, gradients, and geometric estimators share the same basic structure: different observables are averaged with respect to the same Born distribution.

## 2. Auxiliary Sampling Laws and Observed Configurations

A simulation may sample from a distribution other than $\pi_\theta$. One useful reference family is

$$
\eta_{\theta,\alpha}(x)
=
\frac{|\psi_\theta(x)|^\alpha}{Z_\alpha},
\qquad
Z_\alpha=\sum_{z\in\mathcal X}|\psi_\theta(z)|^\alpha,
$$

where $\alpha=2$ gives the Born distribution and $0<\alpha<2$ gives a broader reference distribution. This choice changes the numerical sampling law, while the physical target remains the Born-measure expectation.

A Markov chain targeting $\eta_{\theta,\alpha}$ can be built from a proposal kernel $q_A(y|x)$. A convenient parametrization is

$$
q_A(y|x)=\frac{a(x,y)}{d_A(x)},
\qquad
d_A(x)=\sum_{z\in\mathcal X}a(x,z),
$$

where $a(x,y)\ge 0$ describes the transition geometry. The score $a$ may encode local moves, particle moves, exchange moves, or Hamiltonian-inspired connectivity. Its role is to shape exploration of configuration space.

The Metropolis-Hastings acceptance probability for the reference law $\eta_{\theta,\alpha}$ is

$$
A(x\to y)=
\min\left\{
1,
\frac{|\psi_\theta(y)|^\alpha q_A(x|y)}
{|\psi_\theta(x)|^\alpha q_A(y|x)}
\right\}.
$$

The corresponding transition kernel is

$$
T(y|x)=
q_A(y|x)A(x\to y)
+
\delta_{xy}
\left(1-\sum_{z\ne x}q_A(z|x)A(x\to z)\right).
$$

When the usual irreducibility conditions are satisfied, this construction gives a chain with stationary distribution $\eta_{\theta,
\alpha}$. Detailed balance is a common sufficient route to stationarity; it does not by itself describe the finite-sample mixing rate.

The configuration used in an estimator may also be drawn from an observation kernel $B(y|x)$ applied after the Markov state $x$ is generated:

$$
\sum_{y\in\mathcal X}B(y|x)=1.
$$

If $x\sim\eta_{\theta,\alpha}$ and $y\sim B(\cdot|x)$, then the observed law is

$$
\nu(y)=\sum_{x\in\mathcal X}\eta_{\theta,\alpha}(x)B(y|x).
$$

Equivalently, with an unnormalized density

$$
r_\nu(y)=\sum_{x\in\mathcal X}|\psi_\theta(x)|^\alpha B(y|x),
\qquad
\nu(y)=\frac{r_\nu(y)}{Z_\alpha}.
$$

For the identity observation kernel $B(y|x)=\delta_{xy}$, one has $\nu=\eta_{\theta,\alpha}$. More general kernels can broaden or smooth the observed distribution without redefining the physical Born measure.

## 3. Reweighting to Born-Measure Expectations

Suppose the observed samples are distributed according to $\nu$, while the desired expectation is under $\pi_\theta$. In the discrete setting, the change of distribution is expressed by an importance weight. Since both $\pi_\theta$ and $\nu$ may be known only up to normalization, it is enough to use the unnormalized weight

$$
\omega(y)=\frac{|\psi_\theta(y)|^2}{r_\nu(y)}.
$$

The corresponding self-normalized estimator is

$$
\widehat{\langle f\rangle}_\pi
=
\frac{\sum_{k=1}^N \omega(y_k)f_\theta(y_k)}
{\sum_{k=1}^N \omega(y_k)}.
$$

This form is often the natural one in VMC because the normalizing constants are not separately needed. It is consistent under standard support and moment assumptions, although the ratio form generally introduces a finite-sample bias. This finite-sample effect is distinct from a systematic error caused by missing support.

For compact notation define

$$
\langle f\rangle_w
=
\frac{\sum_{k=1}^N \omega(y_k)f(y_k)}
{\sum_{k=1}^N \omega(y_k)}.
$$

Then energy, force, and metric estimators are obtained by replacing the Born averages in Section 1 with weighted empirical averages:

$$
\widehat E=\langle E_{\mathrm{loc}}\rangle_w,
$$

$$
\widehat F_i=
\langle O_i^*E_{\mathrm{loc}}\rangle_w
-
\langle O_i^*\rangle_w\langle E_{\mathrm{loc}}\rangle_w,
$$

$$
\widehat S_{ij}=
\langle O_i^*O_j\rangle_w
-
\langle O_i^*\rangle_w\langle O_j\rangle_w.
$$

If $B(y|x)=\delta_{xy}$, then $r_\nu(y)=|\psi_\theta(y)|^\alpha$ and

$$
\omega(y)=|\psi_\theta(y)|^{2-
\alpha}.
$$

With a nontrivial observation kernel, the denominator is the corresponding $r_\nu(y)$. In either case, the reported quantity remains an estimator of a Born-measure expectation.

## 4. Support, Tails, and Stability

Reweighting is meaningful only on the part of configuration space that can affect the target expectation. For a given observable $f_\theta$, a basic support condition is

$$
\pi_\theta(y)|f_\theta(y)|>0
\quad\Rightarrow\quad
\nu(y)>0.
$$

In standard probability language, the relevant target contribution is dominated by the observed law. In the present discrete setting, this simply says that configurations contributing to the Born expectation should be observable under the numerical sampling procedure.

This condition depends on the observable. Energy estimation involves $E_{\mathrm{loc}}$. Force and metric estimation involve additional factors such as $O_i$, $O_iE_{\mathrm{loc}}$, and $O_i^*O_j$. A sampling law that is adequate for the energy may therefore be less stable for gradient or geometric quantities.

Support is only the first issue. Variance can still be large when weighted observables have heavy tails. Relevant random variables include

$$
\omega(y)E_{\mathrm{loc}}(y),
\qquad
\omega(y)O_i^*(y)E_{\mathrm{loc}}(y),
\qquad
\omega(y)O_i^*(y)O_j(y).
$$

Near zeros of $\psi_\theta$, ratios in the local energy and logarithmic derivatives can become large. In discrete systems, exact zeros may also create support mismatch if an auxiliary procedure observes configurations where a ratio is poorly controlled or fails to observe configurations that matter for the Born expectation.

A useful diagnostic for weight concentration is the effective sample size

$$
\mathrm{ESS}
=
\frac{\left(\sum_{k=1}^N\omega_k\right)^2}
{\sum_{k=1}^N\omega_k^2}.
$$

Low ESS indicates that a small number of samples dominate the weighted average. This is a diagnostic for importance-weight degeneracy, not for Markov-chain mixing. A chain can mix rapidly while producing unstable weights, and a well-matched reference distribution can still be sampled with high autocorrelation.

## 5. Markov-Chain Mixing and Local-Energy Cost

The Metropolis-Hastings construction concerns the stationary law of the Markov chain. Finite-sample efficiency also depends on how quickly the chain explores the relevant parts of configuration space. Proposal geometry, acceptance rates, barriers between sectors, and the shape of the reference law all affect autocorrelation and the number of effectively independent configurations.

For this reason, it is useful to separate three statistical layers:

1. the outer Markov chain that generates states from the reference law;
2. the observation kernel that produces the configurations entering the estimator;
3. the estimator for the local energy or other observables at each observed configuration.

The third layer matters because local energy evaluation can itself be expensive or stochastic. If the Hamiltonian-connected row is not summed exactly, write the internal estimator as

$$
\widehat E_{\mathrm{loc}}(y,\xi),
$$

where $\xi$ denotes auxiliary randomness. A natural correctness condition is conditional unbiasedness,

$$
\mathbb E_{\xi|y}
\left[
\widehat E_{\mathrm{loc}}(y,\xi)
\right]
=
E_{\mathrm{loc}}(y).
$$

The corresponding reweighted estimator is

$$
\widehat E
=
\frac{\sum_{k=1}^N
\omega(y_k)\widehat E_{\mathrm{loc}}(y_k,\xi_k)}
{\sum_{k=1}^N\omega(y_k)}.
$$

This expression separates configuration-sampling error from Hamiltonian-row estimation error. It also separates statistical variance from computational cost. A broader reference law or an observation kernel may improve support or reduce tail pathologies, but it may also increase the number of wave-function amplitudes or Hamiltonian-connected configurations that need to be evaluated.

## 6. Validation Criteria and Scope

A complete sampling specification for Fock-space VMC can be summarized by five objects:

$$
\pi_\theta,
\qquad
\eta_{\theta,\alpha},
\qquad
q_A \text{ or } T,
\qquad
B,
\qquad
\omega=\frac{|\psi_\theta|^2}{r_\nu}.
$$

The Born measure $\pi_\theta$ defines the physical variational problem. The reference law $\eta_{\theta,\alpha}$ defines the stationary distribution of the outer chain. The proposal and transition kernel affect exploration and autocorrelation. The observation kernel determines the law of configurations entering the estimator. The weight converts observed-law averages back to Born-measure expectations.

Validation is therefore more informative when it reports several diagnostics rather than a single error bar. Useful quantities include

- effective sample size and weight concentration;
- integrated autocorrelation time;
- tail behavior of weighted observables;
- number of unique observed configurations;
- stability of energy, force, and metric estimates under changes of reference law;
- number of wave-function evaluations;
- cost of Hamiltonian-row or local-energy evaluation.

These diagnostics distinguish different failure modes. Support mismatch is a correctness issue. Heavy-tailed weights are a stability issue. Slow mixing is a Markov-chain efficiency issue. Expensive local-energy evaluation is a cost issue. They interact in practice, but they are conceptually different and are best examined separately.

The framework is intentionally independent of a particular neural-network architecture or sampler implementation. It is meant to clarify the probability structure behind Fock-space VMC calculations: auxiliary sampling laws are useful numerical tools, while the final reported quantities are Born-measure expectations of the variational state.

## References

1. N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller, "Equation of State Calculations by Fast Computing Machines," *Journal of Chemical Physics* **21**, 1087--1092 (1953). https://doi.org/10.1063/1.1699114
2. W. K. Hastings, "Monte Carlo Sampling Methods Using Markov Chains and Their Applications," *Biometrika* **57**, 97--109 (1970). https://doi.org/10.1093/biomet/57.1.97
3. W. M. C. Foulkes, L. Mitas, R. J. Needs, and G. Rajagopal, "Quantum Monte Carlo Simulations of Solids," *Reviews of Modern Physics* **73**, 33--83 (2001). https://doi.org/10.1103/RevModPhys.73.33
4. S. Sorella, "Green Function Monte Carlo with Stochastic Reconfiguration," *Physical Review Letters* **80**, 4558--4561 (1998). https://doi.org/10.1103/PhysRevLett.80.4558
5. C. J. Umrigar, J. Toulouse, C. Filippi, S. Sorella, and R. G. Hennig, "Alleviation of the Fermion-Sign Problem by Optimization of Many-Body Wave Functions," *Physical Review Letters* **98**, 110201 (2007). https://doi.org/10.1103/PhysRevLett.98.110201
6. J. R. Trail, "Alternative Sampling for Variational Quantum Monte Carlo," *Physical Review E* **77**, 016704 (2008). https://doi.org/10.1103/PhysRevE.77.016704
7. A. Owen, *Monte Carlo Theory, Methods and Examples*, Chapter 9: Importance Sampling (2013). https://artowen.su.domains/mc/
8. A. Kong, J. S. Liu, and W. H. Wong, "Sequential Imputations and Bayesian Missing Data Problems," *Journal of the American Statistical Association* **89**, 278--288 (1994). https://doi.org/10.1080/01621459.1994.10476469
9. NetKet Documentation, "Quantum Geometric Tensor and Stochastic Reconfiguration." https://netket.readthedocs.io/en/latest/user-guides/sr.html
10. NetKet Documentation, "The Sampler Module." https://netket.readthedocs.io/en/latest/user-guides/sampler.html
11. M. Medvidovic and J. Robledo Moreno, "Neural-Network Quantum States for Many-Body Physics," arXiv:2402.11014 (2024). https://arxiv.org/abs/2402.11014
12. A. Misery, F. Vicentini, and G. Carleo, "Looking Elsewhere: Improving Variational Monte Carlo Gradients by Importance Sampling," arXiv:2507.05352 (2025). https://arxiv.org/abs/2507.05352
13. Z.-Q. Wan, R. Wiersema, and S. Zhang, "Removing Nodal and Support-Mismatch Pathologies in Variational Monte Carlo via Blurred Sampling," arXiv:2603.18148 (2026). https://arxiv.org/abs/2603.18148
