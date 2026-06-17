# libdet

`libdet` is the determinant Hamiltonian backend used by DetNQS. It provides a
small Python interface to screened Slater-Condon operations implemented in
C++ with OpenMP.

It does not define selection, optimization, Monte Carlo state, or solver
policy. Those belong to the calling workflow.

## Determinants

A determinant batch is a contiguous `uint64` array with shape:

```text
(n_det, 2, nword)
```

The second axis contains the alpha and beta occupation strings.

Names follow Dirac notation:

- `ket`: source determinant;
- `bra`: destination determinant;
- `det`: determinant without a directional role.

## Hamiltonian

Construct an RHF spatial-orbital Hamiltonian from one- and two-electron
integrals:

```python
import libdet

ham = libdet.Hamiltonian.rhf(h1, eri, ecore=0.0)
```

### Explicit spaces

```python
h = ham.hij(bra, ket)
diag = ham.diag(x)
projection = ham.project(bra, ket, coeffs)
matrix = ham.matrix(bra, ket)
y = ham.matvec(bra, x, ket=ket)
```

These operations evaluate `H[bra, ket]` on supplied determinant spaces.
Omitting `ket` from `matrix` or `matvec` uses the bra space for both axes.

### Generated Bra

```python
bra = ham.expand(ket, eps, coeffs=coeffs, exclude=ket)
projection = ham.project(
    None,
    ket,
    coeffs,
    eps=eps,
    exclude=ket,
)
```

`expand` returns unique connected bras. Generated `project` also accumulates
their projected amplitudes. Screening uses `|H_ai| >= eps`, or
`|H_ai c_i| >= eps` when coefficients are supplied.

### Connections

```python
weight, degree = ham.degrees(ket, eps)
conns = ham.conns(ket, eps)
```

`degrees` returns the total absolute off-diagonal weight and number of
screened connections for each ket. `conns` additionally returns the connected
bras and matrix elements.

### Sampling

```python
samples = ham.sample_conns(
    ket,
    counts,
    eps1=eps1,
    eps2=eps2,
    seed=seed,
)

projection_samples = ham.sample_project(
    ket,
    coeffs,
    eps1,
    eps2,
    counts,
    exclude=ket,
    n_rep=2,
    seed=seed,
)
```

Sampling covers `eps2 <= |H_ai| < eps1`, with coefficient scaling for
`sample_project`. A fixed seed produces deterministic results.

## Execution

Ket-local connections are reused across repeated connection queries.
Matrix-vector products also reuse prepared determinant spaces. Independent
ket work is parallelized with OpenMP.
