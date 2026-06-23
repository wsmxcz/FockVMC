# libdet design notes

`libdet` is the Fock-space Hamiltonian backend used by `detnqs.operator`.
It provides compact C++ primitives for determinant and spin-adapted CSF
calculations, with nanobind bindings for Python.

The implementation follows one data path:

```text
bit/hash -> integral -> element -> screen table -> cache -> action -> binding
```

Two backends share the same public semantics:

```text
rhf/   Slater determinants in alpha/beta occupation form
guga/  Shavitt paths for spin-adapted CSFs
```

The outer interface is uniform.
The inner algorithms are backend-specific.
RHF is excitation based.
GUGA is occupation-move and Shavitt-segment based.

---

## 1. State, integral, and element conventions

Every basis state is stored as two packed `uint64` blocks:

```text
state.shape = (2, nword)
batch.shape = (N, 2, nword)
```

For RHF:

```text
block 0 = alpha occupation bits
block 1 = beta occupation bits
```

For GUGA:

```text
block 0 = Shavitt up-step bits
block 1 = Shavitt down-step bits
step = 2 * up + down
0 = empty, 1 = down, 2 = up, 3 = doubly occupied
```

All Hamiltonian matrix elements use Dirac order:

$$
H_{bk}=\langle b|H|k\rangle .
$$

All excitations, occupation moves, and path differences are oriented from
`ket` to `bra`.

The integral convention follows PySCF:

$$
h_1(p,q)=h_{pq},
\qquad
\texttt{chem}(p,q,r,s)=(pq|rs).
$$

The ERI input is the PySCF 8-fold chemist-packed tensor.
`Integral` is immutable and Hamiltonian-owned.
It stores the original packed tensor plus small projected tables:

```text
hdiag(p)          = h1(p,p)
coulomb(p,q)      = (pp|qq)
exchange(p,q)     = (pq|qp)
coulomb(p,q,k)    = (pq|kk)
exchange(p,q,k)   = (pk|kq)
```

These projected tables use `O(n^2)` and `O(n^3)` memory.
No dense `O(n^4)` integral table is kept.

RHF element kernels implement Slater--Condon rules.
A ket-local scratch object stores the diagonal value and single-excitation
Fock-like tables, so repeated single elements are `O(1)` lookups after loading
the ket.

GUGA element kernels evaluate the spin-free Hamiltonian directly between
Shavitt-path CSFs. The public semantics are the same as in the RHF backend,

$$
H_{bk}=\langle b|H|k\rangle ,
$$

but the internal representation and contraction path are spin-adapted.

The implementation separates three concerns:

```text
path analysis      ket-to-bra path difference and active orbital span
segment algebra    reusable one- and two-body Shavitt segment factors
ket-local work     occupation data and dynamic-programming workspace
```

Low-degree Hamiltonian connections use specialized kernels. Degree-zero terms,
single occupation moves, double occupation moves, and same-occupation open-shell
couplings are handled by direct paths. General segment contractions are kept for
checked or less frequent cases.

This keeps the outer interface determinant-like while allowing the GUGA backend
to use spin-adapted path algebra internally.

---

## 2. Screen tables and caches

`ScreenTable` is an integral-level candidate table.
It is not a ket graph and does not store exact connections.
It stores cutoff-independent candidates above a base cutoff:

$$
T_{\epsilon_0}={m:B(m)\ge \epsilon_0}.
$$

Any requested cutoff satisfying `eps >= base_eps` is served by a prefix or slice
of the same table. Requests with `eps <= 0` do not build a `ScreenTable`;
those paths use direct enumeration.

RHF `ScreenTable` stores exact double-excitation candidates:

```text
ScreenPair { a, b, h }
same_spin(i, j, eps)
mixed_spin(ia, ib, eps)
```

GUGA `ScreenTable` stores safe occupation-move bounds:

```text
ScreenMove { move, bound }
singles(eps)
doubles(eps)
same_bound(ket)
bound(ket, move)
```

For GUGA,

$$
B(k,m)\ge |H_{bk}|,
$$

so exact element evaluation and filtering remain mandatory.

`ConnCache` stores exact sorted connections for one ket and one positive cutoff.
It is a fixed-capacity 4-way set-associative cache with set-local recency and a
small hit counter.

The cache invariant is cutoff-superset reuse:

$$
\epsilon_0\le\epsilon
\quad\Rightarrow\quad
{|H|\ge\epsilon_0}\supseteq{|H|\ge\epsilon}.
$$

`ConnCache` serves only reusable positive-cutoff exact graphs:

```text
conn(eps > 0)
sample_conn(eps2 > 0)
local_conn strong part for counts == 0
```

It never stores full `eps == 0` graphs, stochastic samples, one-shot projector
data, or scale-dependent/exclude-dependent action data.

`SpaceCache` is a one-entry finite-space cache used by repeated `matvec` and
`matmat` calls. One-shot `matrix` and explicit-bra `project` build local spaces.

---

## 3. Public Hamiltonian primitives

The public primitives are deliberately separate. They represent different
Hamiltonian actions and should not be merged.

`conn(kets, eps)` returns deterministic screened connections:

$$
|H_{bk}|\ge\epsilon.
$$

It returns a shared bra pool, diagonal elements, CSR pointers, bra indices,
matrix elements, and Hamiltonian degrees. The output is sorted by decreasing
`|h|`, because it is backed by sorted backend connection objects.

`sample_conn(kets, counts, eps1, eps2, seed)` samples the window

$$
\epsilon_2\le |H_{bk}|<\epsilon_1.
$$

`counts` may describe one or more sampling streams. The output is stream-major
then ket-major. It has no sorting guarantee, and duplicates are allowed.
If `eps2 > 0`, the sampler uses cached sorted connections. If `eps2 == 0`, it
uses direct enumeration and does not write `ConnCache`.

`local_conn(kets, eps1, eps2, counts, seed, assemble_mode)` returns deterministic
strong connections and sampled weak connections. `assemble_mode` is either
`unique` or `flat`.

Strong part:

$$
|H_{bk}|\ge\epsilon_1.
$$

Weak window:

$$
\epsilon_2\le |H_{bk}|<\epsilon_1.
$$

Weak samples are compressed by multiplicity. For weak-window degree (W_k),
draw count (N_k), and multiplicity (m_b), the unbiased correction has the
form

$$
\sum_b
\frac{m_b W_k}{N_k |H_{bk}|}
H_{bk}\frac{\psi(b)}{\psi(k)}.
$$

For kets with zero weak draws, `local_conn` reads the strong part from cached
deterministic connections. For kets with positive weak draws, it directly
enumerates once and splits the result into strong and weak contributions.
Both cases share the same assembly interface.

Other action primitives use the same backend components:

```text
matrix        dense block H[bras, kets]
matvec        repeated finite-space matrix-vector action
matmat        repeated finite-space matrix-matrix action
project       explicit-bra projection
sample_project stochastic one-shot projection
```

`sample_project` remains separate because it is scale-dependent and one-shot.

---

## 4. Action flow and performance model

The backend separates three concepts:

```text
candidate generation   ScreenTable or direct enumeration
exact evaluation       element kernels
result assembly        evaluation batches, CSR arrays, sampled records
```

This separation keeps the hot paths short:

```text
deterministic screened graph
    ScreenTable -> element -> ConnCache -> CSR output

stochastic connection sampling
    ConnCache or direct enumeration -> window degree -> sampled records

finite-space action
    internal search space -> element -> dense/vector output

local strong/weak action
    cached strong graph or direct enumeration -> local assembly -> CSR output
```

`ScreenTable` is global and integral-driven.
`ConnCache` is ket-dependent and exact.
`SpaceCache` is batch-dependent and finite-space specific.
Scratch objects are local or thread-local.

Python-facing local evaluation batches always start with the input kets, and every index array refers to this batch. Python code should pass the batch directly to model evaluation and should not perform additional deduplication or merging.

For local batches, `assemble_mode` controls the tradeoff between hash work and model evaluation:

$$
T_{\mathrm{unique}}
=
T_{\mathrm{hash}}
+
T_{\mathrm{model}}(n_{\mathrm{ket}}+U),
\qquad
T_{\mathrm{flat}}
=
T_{\mathrm{copy}}
+
T_{\mathrm{model}}(n_{\mathrm{ket}}+N).
$$

Here (U) is the number of unique connected bras, and (N) is the number of strong plus sampled weak records. `unique` uses sharded exact deduplication: bras are routed by fingerprint, deduplicated within independent shards, and gathered after the input-ket prefix. `flat` skips global deduplication and appends every recorded bra directly. Both modes return the same `LocalConn` layout.

The design prioritizes:

```text
tight candidate tables
positive-cutoff graph caching
parallel local strong/weak assembly
minimal Python-side graph manipulation
```

---

## 5. Naming, ownership, and parallelism

Hamiltonian graph language uses:

```text
ket
bra
connection
degree
```

Avoid `row`, `neighbor`, `source`, and `reverse` for Hamiltonian graph logic.
Matrix rows, CSR rows, tree nodes, and optimizer objectives may use their
standard terminology.

Ownership rules:

```text
Integral      immutable Hamiltonian data
ScreenTable   immutable candidate table
ConnCache     positive-cutoff exact ket graphs
SpaceCache    last finite-space search object
Scratch       local or thread-local workspace
Results       owned output arrays
```

OpenMP is required. Parallel regions are local and use thread-local scratch.
Cache locks cover only lookup and insertion. Connection construction occurs
outside locks.

Common low-level utilities live in the shared layer:

```text
bit.hpp       packed-bit operations
hash.hpp      hash primitives
sample.hpp    random windows and weighted draws
integral.hpp  integral storage and projected tables
hamiltonian.hpp public facade
```

Backend-specific logic stays in `rhf/` and `guga/`.
Shared abstractions are kept minimal; physical kernels remain specialized.
