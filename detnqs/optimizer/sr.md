# Stochastic Reconfiguration

This note presents stochastic reconfiguration (SR) as a local tangent-space
method for variational Monte Carlo. Imaginary-time evolution defines a
physically preferred direction in Hilbert space; a variational ansatz can only
move through parameter-induced tangent vectors. SR chooses the parameter
displacement whose induced wave-function change best follows that direction.

The discussion uses real coordinates. A complex wave function may be represented
locally by real coordinate functions such as log-amplitude and phase. Signed
real wave functions use the log-amplitude as the differentiable coordinate and
treat the sign as non-differentiated data.

## 1. Variational State and Coordinates

Let $\mathcal X$ be a Fock-space sector and let $\psi_\theta(x)$ be a
variational wave function with real parameters

$$
\theta\in\mathbb R^P .
$$

The energy is the Rayleigh quotient

$$
E(\theta)=
\frac{\langle\psi_\theta|H|\psi_\theta\rangle}
{\langle\psi_\theta|\psi_\theta\rangle},
$$

and the Born measure is

$$
\pi_\theta(x)=
\frac{|\psi_\theta(x)|^2}
{\sum_{z\in\mathcal X}|\psi_\theta(z)|^2}.
$$

The local energy satisfies

$$
E_{\mathrm{loc}}(x)
=
\frac{(H\psi_\theta)(x)}{\psi_\theta(x)},
\qquad
E(\theta)=\mathbb E_{\pi_\theta}[E_{\mathrm{loc}}].
$$

Choose a real coordinate representation

$$
q_\theta(x)\in\mathbb R^d .
$$

For a positive real wave function one may take
$q_\theta(x)=\log\psi_\theta(x)$. For a complex wave function a useful local
choice is

$$
q_\theta(x)=
\bigl(\log|\psi_\theta(x)|,\phi_\theta(x)\bigr),
$$

where $\phi_\theta$ is the phase. The local tangent map is

$$
J(x)=
\frac{\partial q_\theta(x)}{\partial\theta}
\in\mathbb R^{d\times P}.
$$

It maps a small parameter displacement to the first-order coordinate change,

$$
q_{\theta+\delta\theta}(x)-q_\theta(x)
\approx J(x)\delta\theta .
$$

## 2. Weighted Tangent Geometry

Consider sampled configurations $x_1,\ldots,x_N$ with normalized weights
$w_n\ge 0$ and $\sum_n w_n=1$. These weights may come from exact enumeration,
direct Born sampling, or a reweighted estimator, provided they represent
Born-measure averages for the observables being used.

Define

$$
\bar J=\sum_{n=1}^N w_n J(x_n),
$$

and the centered tangent blocks

$$
O_n=
\sqrt{w_n}\bigl(J(x_n)-\bar J\bigr).
$$

Stacking the sample and coordinate-channel axes gives

$$
O\in\mathbb R^{M\times P},
\qquad M=dN.
$$

The empirical SR metric is

$$
S=O^\mathsf T O.
$$

This is the sampled pullback of wave-function geometry to parameter space. A
small displacement has squared tangent length

$$
|\delta\psi|^2
\approx
\delta\theta^\mathsf T S,\delta\theta,
$$

up to the normalization and coordinate convention used to construct $O$.

## 3. Projection and Natural Gradient

Let

$$
\Delta E_n=E_{\mathrm{loc}}(x_n)-E.
$$

In the positive real convention, the sample-space force can be written as

$$
b_n=2\sqrt{w_n},\Delta E_n.
$$

For complex or signed representations, the same object is mapped through the
model's real-coordinate cotangent convention. The only essential requirement is
that $b$ and $O$ live in the same real coordinate space. The parameter-space
gradient is then

$$
g=O^\mathsf T b.
$$

The local tangent projection problem is

$$
\min_\delta |O\delta+b|^2 .
$$

Its normal equation is

$$
O^\mathsf T O,\delta=-O^\mathsf T b,
$$

or

$$
S\delta=-g.
$$

With damping,

$$
(S+\lambda I)\delta=-g,
\qquad \lambda\ge 0 .
$$

This is SR as a natural-gradient or imaginary-time tangent projection. More
generally, one may include a signed step scale $\eta$ and write the actual
parameter displacement as

$$
(S+\lambda I)\Delta=\eta g.
$$

The descent convention corresponds to $\eta<0$. In what follows, $\Delta$
always denotes the actual parameter displacement.

## 4. Rank, Cost, and Structured Approximations

Because

$$
O\in\mathbb R^{M\times P},
$$

the sampled metric obeys

$$
\operatorname{rank}(S)
=
\operatorname{rank}(O^\mathsf T O)
\le \min(M,P).
$$

When $P$ is large and the effective sample dimension $M$ is modest, $S$ is
low-rank before regularization. Forming a dense parameter-space metric costs
$O(P^2)$ storage and a dense direct solve costs $O(P^3)$.

One possible response is to replace $S$ by a structured metric model,

$$
S\approx \widehat S,
$$

for example diagonal, block-diagonal, low-rank, or Kronecker-factored forms.
Such approximations are useful metric models, but they are not algebraic
rewrites of exact sampled SR. Their quality depends on whether the chosen
structure captures the tangent correlations relevant to the ansatz and
Hamiltonian.

Matrix-free parameter-space solvers avoid explicitly forming $S$ by applying
$O$ and $O^\mathsf T$ through JVP/VJP operations. They preserve the sampled
metric but move the cost to repeated tangent products and iterative convergence.

## 5. Sample-Space SR and minSR

For $\lambda>0$, the identity

$$
(O^\mathsf T O+\lambda I)^{-1}O^\mathsf T
=
O^\mathsf T(OO^\mathsf T+\lambda I)^{-1}
$$

gives an equivalent shifted update in sample space. Define

$$
K=OO^\mathsf T\in\mathbb R^{M\times M}.
$$

With signed step scale $\eta$, sample-space SR solves

$$
(K+\lambda I)a=\eta b,
\qquad
\Delta=O^\mathsf T a.
$$

This is advantageous when $M\ll P$. In the unshifted underdetermined case, it
selects the Euclidean minimum-norm displacement satisfying the sampled tangent
equation, when that equation is consistent.

This is the minimum-step SR viewpoint: the tangent problem is unchanged, but
the linear algebra is expressed through the sample-space kernel $K$ rather than
the parameter-space metric $S$.

## 6. Predictive SR

Predictive SR (PSR) adds short memory to the same sample-space geometry. The
memory variable is the previous actual parameter displacement, not an
unscaled direction. Let

$$
\Delta_{t-1}
$$

be the previous displacement and define the predictor

$$
p_t=\mu\Delta_{t-1},
\qquad 0\le \mu <1 .
$$

The current correction is obtained by matching the residual tangent equation

$$
O_t\Delta_t \approx \eta_t b_t.
$$

Writing

$$
\Delta_t=p_t+q_t,
$$

the residual equation becomes

$$
O_tq_t\approx \eta_t b_t-O_tp_t.
$$

Thus PSR solves

$$
r_t=\eta_t b_t-O_tp_t,
$$

$$
(K_t+\lambda I)a_t=r_t,
\qquad
q_t=O_t^\mathsf T a_t,
$$

and forms

$$
\Delta_t=p_t+q_t.
$$

The predictor is therefore corrected inside the current tangent-space SR
equation. It is not an external momentum step. The stored memory after the
update is

$$
\Delta_t,
$$

so the next predictor remains in the same parameter-space scale as the actual
wave-function displacement.

## 7. Summary

The SR family can be summarized by one tangent object and several linear
algebra choices:

$$
O_n=\sqrt{w_n}\bigl(J(x_n)-\bar J\bigr),
\qquad
S=O^\mathsf T O,
\qquad
K=OO^\mathsf T.
$$

Parameter-space SR solves with $S$. Sample-space SR solves the equivalent
shifted problem with $K$. PSR solves a residual sample-space problem around a
predicted actual displacement. These forms differ in numerical representation
and stabilization, not in the underlying variational objective.

## References

1. P. A. M. Dirac, "Note on Exchange Phenomena in the Thomas Atom,"
   *Proceedings of the Cambridge Philosophical Society* **26**, 376--385
   (1930).
2. J. Frenkel, *Wave Mechanics: Advanced General Theory*, Clarendon Press,
   Oxford (1934).
3. A. D. McLachlan, "A Variational Solution of the Time-Dependent Schrodinger
   Equation," *Molecular Physics* **8**, 39--44 (1964).
   https://doi.org/10.1080/00268976400100041
4. J. P. Provost and G. Vallee, "Riemannian Structure on Manifolds of Quantum
   States," *Communications in Mathematical Physics* **76**, 289--301 (1980).
   https://doi.org/10.1007/BF02193559
5. S. Amari, "Natural Gradient Works Efficiently in Learning,"
   *Neural Computation* **10**, 251--276 (1998).
   https://doi.org/10.1162/089976698300017746
6. S. Sorella, "Green Function Monte Carlo with Stochastic Reconfiguration,"
   *Physical Review Letters* **80**, 4558--4561 (1998).
   https://doi.org/10.1103/PhysRevLett.80.4558
7. S. Sorella, "Generalized Lanczos Algorithm for Variational Monte Carlo,"
   *Physical Review B* **64**, 024512 (2001).
   https://doi.org/10.1103/PhysRevB.64.024512
8. G. Carleo and M. Troyer, "Solving the Quantum Many-Body Problem with
   Artificial Neural Networks," *Science* **355**, 602--606 (2017).
   https://doi.org/10.1126/science.aag2302
9. C.-Y. Park and M. J. Kastoryano, "Geometry of Learning Neural Quantum
   States," *Physical Review Research* **2**, 023232 (2020).
   https://doi.org/10.1103/PhysRevResearch.2.023232
10. F. Vicentini, D. Hofmann, A. Szabo, et al., "NetKet 3: Machine Learning
    Toolbox for Many-Body Quantum Systems," *SciPost Physics Codebases*, 7
    (2022). https://doi.org/10.21468/SciPostPhysCodeb.7
11. J. Martens and R. Grosse, "Optimizing Neural Networks with
    Kronecker-Factored Approximate Curvature," in *Proceedings of the 32nd
    International Conference on Machine Learning*, PMLR **37**, 2408--2417
    (2015). https://proceedings.mlr.press/v37/martens15.html
12. A. Chen and M. Heyl, "Empowering Deep Neural Quantum States through
    Efficient Optimization," *Nature Physics* **20**, 1476--1481 (2024).
    https://doi.org/10.1038/s41567-024-02566-1
13. R. Rende, L. L. Viteritti, L. Bardone, F. Becca, and S. Goldt, "A Simple
    Linear Algebra Identity to Optimize Large-Scale Neural Network Quantum
    States," *Communications Physics* **7**, 260 (2024).
    https://doi.org/10.1038/s42005-024-01732-4
14. G. Goldshlager, N. Abrahamsen, and L. Lin, "A Kaczmarz-Inspired Approach
    to Accelerate the Optimization of Neural Network Wavefunctions,"
    *Journal of Computational Physics* **516**, 113351 (2024).
    https://doi.org/10.1016/j.jcp.2024.113351
15. Z. Y. Wang, Y. C. Wang, D. S. Lv, et al., "Solving the Hubbard Model with
    Neural Quantum States," arXiv:2507.02644 (2025).
    https://arxiv.org/abs/2507.02644
16. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in
    *Proceedings of the 3rd International Conference on Learning
    Representations* (ICLR, 2015). https://arxiv.org/abs/1412.6980
