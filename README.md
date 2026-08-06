# DetNQS

DetNQS is a compact research codebase for Fock-space variational Monte Carlo.
It combines neural wavefunctions with a compiled electronic-Hamiltonian oracle
while keeping the sampling, estimation, and optimization flow explicit.

The project is intended for researchers and developers who want to inspect,
modify, and reproduce algorithms directly. Its design favors a small number of
clear components over a general-purpose framework.

## Install and run

DetNQS requires Python 3.12 or later and a C++20 compiler.

```bash
git clone https://github.com/wsmxcz/DetNQS_dev.git
cd DetNQS_dev
pip install -e .
```

Run the short end-to-end check from the repository root:

```bash
python scripts/test.py
```

It loads a small FCIDUMP, compares exact and Monte Carlo estimates, evaluates a
gradient, applies one PSR update, and checks MC checkpoint restore.

The complete research-oriented VMC script is:

```bash
python examples/vmc.py
```

It contains the full experiment setup: Hamiltonian loading, model and chain
initialization, auxiliary sampling, observables, Optax composition, logging,
and checkpoints. FCIDUMP data used by the scripts live in `scripts/FCIDUMP/`.

## Design

Each layer owns one part of the calculation:

```text
hilbert     configurations and particle/spin sectors
operator    electronic Hamiltonians and fermion primitives
model       wavefunction ansatzes
sampler     Markov chains, proposals, and observation sampling
vstate      exact, selected-space, and Monte Carlo estimators
optimizer   SR and predictive sample-space SR transforms
driver      estimate -> update -> apply -> log
```

The main dependency and data flow is:

```text
FCIDUMP
   ↓
Sector + Hamiltonian
   ↓
Model + Sampler
   ↓
VState estimate + geometry
   ↓
Optax optimizer
   ↓
VMC parameter update
```

Models describe only `logpsi = model.apply(params, x)`. Samplers own persistent
chains and auxiliary observations. Variational states turn configurations into
energies, gradients, and SR geometry. The driver only coordinates these calls.

## Supported methods

- Fixed-particle, fixed-$S_z$ configuration sectors.
- `RBM`, `Backflow`, `GBackflow`, `SBackflow`, and `PBackflow` models.
- Hamiltonian and single-excitation Markov proposals with optional observation
  blur and amplitude tempering.
- `ExactState`, `SelectedState`, and Born-reweighted `MCState` estimators.
- Parameter-space `sr` and predictive sample-space `psr`; `psr(mu=0)` is
  sample-space SR.
- Screened connections, local-energy action, projection, finite matrices, and
  matrix-vector products through `libdet`.

## Documentation

- [Package design and conventions](detnqs/README.md)
- [Monte Carlo estimator](detnqs/sampler/vmc.md)
- [SR and PSR](detnqs/optimizer/sr.md)
- [Compiled Hamiltonian oracle](detnqs/operator/libdet/README.md)

## Citation

```bibtex
@article{che2026detnqs,
  title = {A Deterministic Framework for Neural Network Quantum States in Quantum Chemistry},
  author = {Che, Zheng},
  journal = {Journal of Chemical Theory and Computation},
  year = {2026},
  doi = {10.1021/acs.jctc.6c00445}
}
```

Apache-2.0 licensed. See [LICENSE](LICENSE).
