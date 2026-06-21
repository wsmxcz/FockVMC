# libdet

`libdet` is the local Hamiltonian kernel behind `detnqs.operator.Hamiltonian`.

It provides exact, screened, cached, and sampled Hamiltonian operations for two bases:

```text
RHF   -> Slater determinants
GUGA  -> spin-adapted Shavitt CSFs
```

Python owns the user workflow, VMC logic, selected-space logic, and future MPI distribution.
`libdet` stays rank-local and uses OpenMP only inside one process.

## Conventions

State layout:

```text
RHF:
  (n_det, 2, nword)
  alpha block, beta block

GUGA:
  (n_path, nword)
  packed Shavitt paths
```

Names:

```text
ket -> source state
bra -> destination state
x   -> generic state batch
```

RHF determinant gauge:

$$
|D\rangle = (-1)^{N_{\beta\alpha}}|\alpha\rangle|\beta\rangle .
$$

GUGA uses canonical Shavitt/GUGA path gauge.

Integrals use PySCF chemist notation:

```text
h(p,q)          one-electron integral
chem(p,q,r,s)   (pq|rs)
ecore           scalar core energy
```

No antisymmetrized integral convention is used internally.

For GUGA,

$$
\hat H =
E_\mathrm{core}
+
\sum_{pq} h_{pq}E_{pq}
+
\frac{1}{2}\sum_{pqrs}(pq|rs)e_{pq,rs}.
$$

$$
E_{pq}=\sum_\sigma a^\dagger_{p\sigma}a_{q\sigma},
\qquad
e_{pq,rs}=E_{pq}E_{rs}-\delta_{qr}E_{ps}.
$$

## Layout

```text
libdet/
  integral.hpp      integral storage
  window.hpp        AbsWindow, ConnWindow
  hamiltonian.hpp   common facade

  rhf/
    det.hpp         determinant basis and residues
    element.hpp     Slater-Condon elements
    screen.hpp      screened excitation pairs
    cache.hpp       Conns, ConnCache, SpaceCache
    internal.hpp    known-space residue search
    external.hpp    external determinant enumeration
    sample.hpp      cached sampling

  guga/
    path.hpp        Shavitt paths and oCFG residues
    segment.hpp     Shavitt segments and two-body LUT
    element.hpp     spin-free GUGA elements
    screen.hpp      screened oCFG moves
    cache.hpp       Conns, ConnCache, SpaceCache
    internal.hpp    known-space residue search
    external.hpp    external CSF enumeration
    sample.hpp      cached sampling
```

## Pipeline

```text
integral
  -> element
  -> screen
  -> cache
  -> internal / external
  -> sample
```

Layer meanings:

```text
integral:
  h, (pq|rs), ecore

element:
  exact H_bra,ket

screen:
  cutoff-specific candidates
  false positives allowed
  false negatives forbidden

cache:
  exact Conns or basis Space

internal:
  search inside supplied spaces

external:
  enumerate connected bras from a ket

sample:
  draw from cached exact Conns
```

## Exact elements

RHF:

```text
determinant excitation degree
  -> Slater-Condon formula
```

```text
degree 0  diagonal
degree 1  single alpha / beta
degree 2  same-spin or opposite-spin double
degree >2 zero
```

Same-spin double:

$$
H_{ab,ij}=(ai|bj)-(aj|bi).
$$

Opposite-spin double:

$$
H_{ab,ij}=(ai|bj).
$$

GUGA:

```text
Shavitt path pair
  -> oCFG move
  -> spin-free element
```

```text
degree 0  same oCFG
degree 1  one electron moved
degree 2  two electrons moved
degree >2 zero
```

```text
coeff1  Shavitt one-body segment product
coeff2  reduced two-body local transition LUT
```

`coeff2` evaluates the spin-free two-body operator

$$
e_{pq,rs}=E_{pq}E_{rs}-\delta_{qr}E_{ps}.
$$

It does not expand determinants and does not enumerate intermediate CSFs.

## Screen

`Screen` is a candidate source, not a stored Hamiltonian.

The invariant is

$$
B(y,x)\ge |H_{yx}|.
$$

RHF:

```text
occupied determinant residue
  -> screened excitation pairs
  -> exact Slater-Condon
```

RHF pair magnitudes are exact up to sign, so both window edges can be used before exact filtering.

GUGA:

```text
occupied oCFG residue
  -> screened oCFG moves
  -> generated CSF paths
  -> exact GUGA element
```

GUGA move values are upper bounds. They use only the lower cutoff before exact filtering. The upper window edge is applied after exact `hij`.

## Cache

Three mechanisms are intentionally separate:

```text
ConnCache:
  repeated-ket exact connection cache
  used by conn, sample_conn, sample_project

SpaceCache:
  selected-space residue index cache
  used by matrix, matvec, matmat, internal project

Scratch:
  thread-local temporary storage
  not persistent
```

`Conns` is an exact screened connection list for one ket:

```text
diag        exact H_ket,ket
offdiag     exact bras with |h| >= cutoff
prefix_abs  prefix sums of |h|
window      eps_lo <= |h| < eps_hi
```

Cutoff reuse:

```text
cached cutoff <= requested cutoff  -> reuse
cached cutoff >  requested cutoff  -> rebuild
```

## Internal and external search

Internal search works inside supplied spaces.

```text
known basis space
  -> residue lookup
  -> exact degree check
  -> exact element
  -> accumulate
```

RHF internal:

```text
DetSpace over known kets
bra-local residue search
static OpenMP
row-local output
```

GUGA internal:

```text
PathSpace over known bras
ket-local oCFG residue search
guided OpenMP
dense thread-local output
```

GUGA residue identities:

$$
n-e_q=n'-e_p
$$

for degree 1, and

$$
n-e_q-e_s=n'-e_p-e_r
$$

for degree 2.

External search enumerates connected bras from one ket.

RHF external:

```text
ket determinant
  -> screened excitation pairs
  -> exact Slater-Condon
  -> emit
```

GUGA external:

```text
ket path
  -> screened oCFG moves
  -> generate CSF paths
  -> exact GUGA element
  -> emit
```

`expand` and external `project` use this streaming path directly.

## Sampling

Sampling consumes cached exact `Conns`.

```text
sample_conn:
  ket_conns(kets, eps2)
  -> window [eps2, eps1)
  -> draw by |h|

sample_project:
  ket_conns(kets, eps2 / max|scale|)
  -> per-ket scaled window
  -> exclude filter
  -> weighted estimator
```

For one ket,

$$
P(y|x)=
\frac{|H_{yx}|}
{\sum_{z\in W_x}|H_{zx}|}.
$$

For sampled projection with coefficient $c_x$,

$$
W_x=
\{
y:
\frac{\epsilon_2}{|c_x|}
\le
|H_{yx}|
<
\frac{\epsilon_1}{|c_x|}
\}.
$$

Randomness is derived per `(ket, stream)`, so it is independent of OpenMP scheduling.

## OpenMP model

```text
shared:
  Integral, Screen, Space, input batches

thread-local:
  KetScratch, VisitScratch, ElementScratch
  ProjectBuffer, SampleBuffer
  targets and hit buffers

hot loop:
  no mutex
  no shared push_back
  no cache lookup
  no screen construction
```

Path summary:

```text
RHF external:
  ket-local guided
  thread-local KetScratch and local output

RHF internal:
  bra-local static
  thread-local VisitScratch
  row-local output

GUGA external:
  ket-local guided
  thread-local KetScratch and local output

GUGA internal:
  ket-local guided
  thread-local VisitScratch / ElementScratch
  dense local output and reduction

sample:
  cached exact Conns
  deterministic per-(ket,stream) RNG
  thread-local buffers

MPI future:
  Python distributes batches
  libdet stays rank-local OpenMP only
```

## Public operations

```text
hij              exact H_bra,ket
diags            exact diagonal
matrix           explicit known-space matrix
matvec/matmat    streaming known-space contractions
project          known-space or external projection
expand           deterministic external bra generation
conn             exact per-ket Conns through ConnCache
sample_conn      samples from cached Conns
sample_project   sampled projection from cached Conns
```

## Names

```text
KetScratch      ket-local external scratch
VisitScratch    internal residue scratch
ProjectBuffer   deterministic projection buffer
SampleBuffer    sampled projection buffer
Conns           sorted exact connections
ConnCache       repeated-ket connection cache
SpaceCache      selected-space residue cache
Screen          cutoff-specific candidate source
Pair            RHF screened excitation candidate
Move            GUGA screened oCFG move
```

Avoid:

```text
heatbath
graph
node
edge
src / dst
thread context
executor
policy wrapper
```

Prefer short names, with at most three underscore-separated fields.

## Non-goals

```text
no complete Hamiltonian graph storage
no persistent CSR cache
no MPI inside libdet
no distributed cache
no runtime fallback tree
no execution-policy wrapper
no global singleton cache
```
