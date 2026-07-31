# DetNQS

DetNQS is an experimental library for Fock-space variational states, with a
current focus on quantum-chemistry Hamiltonians.

## Scope

Current strengths:

- Determinant Fock-space sectors.
- Exact, selected-basis, and Monte Carlo variational states.
- Backflow/RBM wavefunction models.
- SR, minSR, and PSR optimizers through Optax.
- Screened Hamiltonian expansion, projection, sampling, sparse matrices, and
  matrix-vector products.

Longer term, DetNQS is intended to grow toward a shared many-body operator
core: basis, symmetry, and term definitions compiled to sector rules and
operator primitives. The present code keeps that direction in mind while
remaining specialized for quantum chemistry.

## Install

DetNQS requires Python 3.11 or later and a C++20 compiler.

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Testing

Python tests:

```bash
python -m pytest
```

C++ libdet tests without building the Python extension:

```bash
cmake -S . -B build-cpp -DDETNQS_BUILD_PYTHON=OFF -DDETNQS_BUILD_CPP_TESTS=ON
cmake --build build-cpp
ctest --test-dir build-cpp --output-on-failure
```

## Citation

```bibtex
@article{che2026detnqs,
  title = {A Deterministic Framework for Neural Network Quantum States in Quantum Chemistry},
  author = {Che, Zheng},
  journal = {Journal of Chemical Theory and Computation},
  year = {2026},
  doi = {10.1021/acs.jctc.6c00445},
  url = {https://pubs.acs.org/doi/10.1021/acs.jctc.6c00445}
}
```

## License

Apache License 2.0. See `LICENSE`.
