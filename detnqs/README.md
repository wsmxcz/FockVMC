# DetNQS

DetNQS is a lightweight determinant-space neural quantum state library for
quantum chemistry.

The goal is a small, explicit, high-performance research code:

- `libdet` owns determinant Hamiltonian physics.
- `detnqs.model` owns neural wavefunction ansatzes.
- `detnqs.vstate` owns variational estimators and dynamic physical state.
- `detnqs.sampler` owns Markov-chain dynamics.
- `detnqs.preconditioner` owns SR and minSR geometry.
- `detnqs.driver` owns optimizer state and the training loop.
- `detnqs.utils` owns shape and precision policy.

## Determinants

A determinant batch is always

```text
dets: uint64 array, shape = (N, 2, nword)
```

axis 1 stores alpha and beta bitstrings.

`dets` has no left/right role.  
`bras` and `kets` mean the two axes of `H[bras, kets]`.

## Wavefunction convention

A model returns one of:

```text
real           : log|psi|
complex        : log|psi| + i phase
(sign, logabs) : signed real wavefunction
```

Autodiff acts on a real coordinate:

```text
real        -> log|psi|
complex     -> [log|psi|, phase]
signed real -> logabs
```

`model.cotangent(logpsi, dlogpsi)` maps a cotangent with respect to
`logpsi` into the coordinate used by autodiff.

## Energy and geometry

The variational energy is

```text
E = <psi|H|psi> / <psi|psi>
```

For Monte Carlo,

```text
E_loc(D) = sum_D' H[D,D'] psi(D') / psi(D)
```

SR and minSR use the centered weighted Jacobian

```text
O = sqrt(w) * (J - <J>_w)
S = O† O
K = O O†
```

SR solves

```text
(S + shift I) delta = grad
```

minSR solves

```text
(K + shift I) a = b
delta = O† a
```

## State contract

All variational states implement

```python
state, loss, grad, stats, geometry = state.expect_and_grad(qgt=True)
```

The returned `state` contains any advanced sampler state.  

## Shape and precision

`utils.batch` centralizes chunking and bucket padding.

- `apply`, `jvp`, `vjp`: streaming over the leading axis.
- `bucket`: fixed-shape padding for non-streaming full-batch kernels.

`utils.precision` centralizes dtype policy.

- `model`: neural-network forward and autodiff
- `calc`: energy and host reductions
- `sr`: SR/minSR linear algebra

JAX x64 is enabled by default because determinants use `uint64`.

## Future direction