# DetNQS

DetNQS is a compact experimental framework for determinant-space neural
quantum states in quantum chemistry. It combines JAX variational models with
an OpenMP-parallel C++ determinant Hamiltonian backend.

The repository supports three main workflows:

- exact and selected determinant-space optimization;
- variational Monte Carlo with determinant-valued Markov chains;
- heat-bath CI and perturbative estimation.

## Structure

- [`detnqs`](detnqs/README.md): models, variational states, samplers,
  optimizers, and the VMC driver.
- [`libdet`](libdet/README.md): screened Slater-Condon operations, connections,
  sampling, sparse matrices, and matrix-vector products.
- [`examples`](examples): executable research workflows.
- [`tests`](tests): focused numerical and integration tests.

`detnqs` owns the variational algorithm. `libdet` owns determinant Hamiltonian
operations. Solver and workflow policy remain outside `libdet`.

## Installation

DetNQS requires Python 3.11 or later and a C++20 compiler.

```bash
pip install .
```

For an editable development installation:

```bash
pip install -e ".[dev]"
```

This installs both Python packages, `detnqs` and `libdet`.

## Usage

The examples are the primary usage reference:

- `examples/exact.py`: exact-space optimization;
- `examples/selected.py`: selected-space optimization;
- `examples/vmc.py`: Monte Carlo optimization;
- `examples/hci.py`: heat-bath CI and perturbative estimation;
- `examples/hchain.py`: hydrogen-chain calculations.

Run them from the repository root:

```bash
python examples/exact.py
```

Run the test suite with:

```bash
pytest
```

## Citation

```bibtex
@article{che2026detnqs,
  title={A Deterministic Framework for Neural Network Quantum States in Quantum Chemistry},
  author={Che, Zheng},
  journal={arXiv preprint arXiv:2601.21310},
  year={2026}
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
