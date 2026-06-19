# AGENTS.md

Guidance for coding agents working in this repository.

## Project Shape

DetNQS is a compact experimental research codebase for Fock-space neural
quantum states. Keep changes light, direct, and easy to inspect.

Important boundaries:

- `detnqs/` is the Python package root.
- `detnqs.hilbert` owns sectors and the `x` configuration encoding.
- `detnqs.operator.Hamiltonian` is the public Hamiltonian boundary.
- `detnqs.operator.libdet` is the compiled determinant oracle behind
  `Hamiltonian`.
- `detnqs.model`, `detnqs.vstate`, `detnqs.sampler`, `detnqs.optimizer`, and
  `detnqs.driver` form the VMC path.

## Design

- Prefer direct data flow over wrappers, adapters, or forwarding layers.
- Do not add compatibility aliases unless explicitly requested.
- Do not add broad base classes or plugin systems for narrow needs.
- Keep state small and visible.
- Prefer NumPy vectorization on host-side array code.
- Use JAX where model evaluation, differentiation, or optimizer geometry
  requires it.
- Treat future many-body generality as a boundary-design constraint, not as a
  reason to add a premature DSL.

Current logic flow:

```text
Sector + Hamiltonian + Model
        |
        v
VState.expect_and_grad()
        |
        v
optimizer geometry/update
        |
        v
VMC iteration
```

## Interfaces

- Use `Sector` for constrained Hilbert-space sectors.
- Use `x` for generic Fock-basis batches and `det` for determinant batches.
- Use `bra` and `ket` for Hamiltonian matrix indices.
- Use `h1`, `eri`, and `ecore` for PySCF-aligned electronic integrals.
- Prefer group-style public symmetry names such as `U1` and `SU2`.

Validation belongs at user-facing boundaries. Internal paths should assume the
project convention and avoid repeated conversions.

## Documentation

- Root `README.md`: user-facing scope, install, workflow, examples, tests,
  citation, license.
- `detnqs/README.md`: package boundaries and public conventions.
- `detnqs/operator/libdet/README.md`: compiled backend contract.
- `detnqs/optimizer/sr.md` and `detnqs/sampler/vmc.md`: theory notes.

Keep these files orthogonal. Do not duplicate tutorials or implementation
details across them.

## Testing

Use the WSL conda environment named `detnqs`:

```bash
~/miniconda3/bin/conda run -n detnqs python -m pytest
```

After C++ binding changes:

```bash
~/miniconda3/bin/conda run -n detnqs python -m pip install -e .
~/miniconda3/bin/conda run -n detnqs python -m pytest
```

For quick syntax checks:

```bash
~/miniconda3/bin/conda run -n detnqs python -m compileall detnqs tests examples
```

## Editing Discipline

- Keep edits scoped to the requested design or bug.
- Do not revert unrelated user changes.
- Update examples and tests together with public API changes.
- Keep comments short and technical.
- Prefer deleting redundant code over adding new machinery.
- Treat a dirty worktree as user-owned unless cleanup is requested.
