# DetNQS

DetNQS is a compact research codebase for Fock space variational Monte Carlo. Now in active development.

## Start

Python 3.12+ and a C++20 compiler are required.

```bash
git clone https://github.com/wsmxcz/DetNQS_dev.git
cd DetNQS_dev
pip install -e .
```

Run the small end-to-end check:

```bash
python scripts/test.py
```

Run the complete VMC experiment:

```bash
python examples/vmc.py
```

FCIDUMP inputs are kept in `scripts/FCIDUMP/`.

## Documentation

- [Design and conventions](detnqs/README.md)
- [VMC sampling and estimation](detnqs/sampler/vmc.md)
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
