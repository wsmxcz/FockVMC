import jax
import optax
from pyscf import ao2mo, gto, scf

from fvmc import Hamiltonian, MCState, VMC
from fvmc.hilbert import DetSector
from fvmc.model import Backflow
from fvmc.optimizer import psr
from fvmc.sampler import MCSampler
from fvmc.utils import Logger


mol = gto.M(
    atom="H 0 0 0; H 0 0 2.00",
    basis="cc-pvdz",
    unit="Angstrom",
    verbose=0,
)
mf = scf.RHF(mol).run()

norb = mf.mo_coeff.shape[1]
n_alpha, n_beta = mol.nelec
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
eri = ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb)

sector = DetSector(norb, nelec=n_alpha + n_beta, spin=mol.spin)
hamiltonian = Hamiltonian(sector, h1, eri, ecore=mol.energy_nuc())

model = Backflow(
    norb=norb,
    n_alpha=n_alpha,
    n_beta=n_beta,
)
sampler = MCSampler(n_samples=128, n_chains=128, rank=None)
state = MCState.init(
    model,
    hamiltonian,
    sampler=sampler,
    chains=sector.reference(sampler.n_chains),
    key=jax.random.key(0),
)
optimizer = optax.chain(
    psr(),
    optax.scale_by_learning_rate(5.0e-2),
)

vmc = VMC.init(state, optimizer)
vmc.run(200, log=Logger(every=10))
