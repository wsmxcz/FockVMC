# DetNQS

DetNQS is a compact research codebase for Fock-space variational Monte Carlo.

## Install and run

Python 3.12+ and a C++20 compiler are required.

```bash
git clone https://github.com/wsmxcz/DetNQS_dev.git
cd DetNQS_dev
pip install -e .
```

```bash
python scripts/test.py
python examples/vmc.py
```

`scripts/test.py` checks the complete small-system path. `examples/vmc.py` is
the standard VMC experiment. FCIDUMP inputs are stored in `scripts/FCIDUMP/`.

## Scope

```text
hilbert    sectors and configurations
operator   electronic Hamiltonians and libdet primitives
model      RBM and backflow wavefunctions
sampler    Markov chains and proposals
vstate     exact, selected-space, and Monte Carlo estimators
optimizer  SR and predictive sample-space SR
driver     optimization loop, logging, and MC checkpoints
```

Models include `RBM`, `Backflow`, `GBackflow`, `SBackflow`, and `PBackflow`.
Optimizers include parameter-space SR and PSR; `psr(mu=0)` is sample-space SR.

## Documentation

- [Design and conventions](detnqs/README.md)
- [Monte Carlo estimator](detnqs/sampler/vmc.md)
- [SR and PSR](detnqs/optimizer/sr.md)
- [libdet](detnqs/operator/libdet/README.md)

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

Apache-2.0. See [LICENSE](LICENSE).
