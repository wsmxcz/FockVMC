# DetNQS

DetNQS is an experimental library for Fock-space variational states, with a
current focus on quantum-chemistry Hamiltonians.

The project combines a compact Python/JAX variational layer with a compiled
Hamiltonian oracle for determinant and spin-adapted electronic structure
workflows.

## Scope

Current strengths:

- Determinant and spin-adapted Fock-space sectors.
- Exact, selected-basis, and Monte Carlo variational states.
- Backflow/RBM wavefunction models.
- SR, minSR, and PSR optimizers through Optax.
- Screened Hamiltonian expansion, projection, sampling, sparse matrices, and
  matrix-vector products.

Longer term, DetNQS is intended to grow toward a shared many-body operator
core: graph, site, symmetry, and term definitions compiled to basis rules and
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

## Minimal Workflow

```python
import jax
import optax

from detnqs import hilbert, operator
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.vstate import ExactState

sector = hilbert.DetSector(norb, n_alpha, n_beta)
H = operator.Hamiltonian(sector, h1, eri, ecore=ecore)

model = Backflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta)
state = ExactState.init(model, H, key=jax.random.key(0))

optimizer = optax.chain(psr(shift=1e-3), optax.scale(-5e-2))
vmc = VMC.init(state, optimizer)
stats = vmc.step()
```

See `examples/` for complete exact, selected-basis, Monte Carlo, and HCI-style
workflows.

## Project Map

- `detnqs/`: Python package and public API.
- `detnqs/operator/libdet/`: compiled Hamiltonian backend.
- `examples/`: runnable research workflows.
- `tests/`: public-contract and workflow checks.

For module boundaries, see `detnqs/README.md`.

## Testing

```bash
python -m pytest
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
