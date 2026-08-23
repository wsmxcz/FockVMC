# FockVMC

FockVMC is a research codebase for Fock-space variational Monte Carlo, currently tailored for molecular electronic structure Hamiltonians.

Please note that this is an experimental codebase under active development, subject to frequent refactoring and breaking changes.

## Install

Python 3.12+ and a C++20 compiler are required.

```bash
git clone https://github.com/wsmxcz/FockVMC.git
cd FockVMC
pip install -e .
```

## Scope

```text
hilbert    sectors and configurations
operator   Hamiltonians and operators
model      RBM and backflow wavefunctions
sampler    Markov chains and proposals
vstate     exact, selected-space, Monte Carlo, and IR estimators
optimizer  SR and sample-space SR
driver     optimization loop, logging, and MC checkpoints
```

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

The early implementation corresponding to this paper is archived in the 1.0 branch. Starting from version 2.0, the codebase has undergone a major refactor and is now built around SelectedState.

## License

MIT License. See [LICENSE](LICENSE).

**Note**: I am open to PhD opportunities for Fall 2027. If our research interests align, feel free to reach out at [wsmxcz@gmail.com](mailto:wsmxcz@gmail.com) or visit my [Personal Homepage](https://wsmxcz.github.io/).