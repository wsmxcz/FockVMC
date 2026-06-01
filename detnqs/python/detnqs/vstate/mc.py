from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import libdet
import numpy as np

from detnqs import utils
from ..model.base import Model
from ..model.base import to_logabs
from ..model.base import to_ratio
from ..optimizer import Geometry
from ..sampler.mcmc import MCSampler
from ..sampler.mcmc import WalkerState
from ..sampler.proposal import unique_dets
from ..utils import precision
from .base import VState


@dataclass(slots=True)
class MCState(VState):
    """Monte Carlo variational state.

    Physical target:
        pi_theta(x) proportional to |psi_theta(x)|^2.

    Markov reference sampled by MCSampler:
        eta_alpha(x) proportional to |psi_theta(x)|^alpha.

    Observation law:
        x ~ eta_alpha, y ~ B(y|x).

    Degree-tilted blurred density:
        The observed sample carries a source mass s(x). For Hamiltonian blur,

            s(x) = d_B(x) if d_B(x) > 0 else 1.

        The unnormalized density used for reweighting is

            r_tilde(y) = sum_x |psi(x)|^alpha s(x) B(y|x).

        For non-empty Hamiltonian blur kets this becomes

            r_tilde(y)
              = (1 - beta) d_B(y) |psi(y)|^alpha
                + beta sum_x |H_yx| |psi(x)|^alpha,

        which only requires first-order Hamiltonian connections from y.

    Importance weight:
        omega(y) = sample_mass(y) |psi_theta(y)|^2 / r_tilde(y).

    libdet builds Hamiltonian connections on host. The neural model is
    evaluated once on a shared determinant pool and then reused by local
    energy, observed-density, gradient, and geometry reductions.
    """

    model: Model
    params: Any
    hamiltonian: Any
    sampler: MCSampler
    sampler_state: WalkerState

    n_alpha: int
    n_beta: int
    init_method: str | Any = "hf"

    eloc_eps1: float = 1.0e-3
    eloc_eps2: float = 1.0e-6
    eloc_sample: int = 64

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Any,
        *,
        sampler: MCSampler,
        n_alpha: int,
        n_beta: int,
        key: jax.Array,
        init_method: str | Any = "hf",
        eloc_eps1: float = 1.0e-3,
        eloc_eps2: float = 1.0e-6,
        eloc_sample: int = 64,
    ) -> MCState:
        key, init_key, sample_key = jax.random.split(key, 3)

        params = model.init(
            init_key,
            jnp.zeros((1, 2, hamiltonian.nword), dtype=jnp.uint64),
        )["params"]

        sampler_state = sampler.init(
            params,
            hamiltonian,
            model,
            key=sample_key,
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            init_method=init_method,
        )

        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            sampler=sampler,
            sampler_state=sampler_state,
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            init_method=init_method,
            eloc_eps1=float(eloc_eps1),
            eloc_eps2=float(eloc_eps2),
            eloc_sample=int(eloc_sample),
        )

    def replace(self, **updates: Any) -> MCState:
        return replace(self, **updates)

    def expect(self) -> tuple[MCState, dict[str, float]]:
        new_state, _, _, stats, _ = self._run(grad=False, geometry=False)
        return new_state, stats

    def expect_and_grad(self, *, geometry: bool = False):
        return self._run(grad=True, geometry=geometry)

    def _run(self, *, grad: bool, geometry: bool):
        """Run one estimator pass.

        Timing fields:
            sample:
                Counted-state bookkeeping, resampling, accept/reject, and
                observation bookkeeping outside connection/model work.

            graph:
                libdet connection work, including deterministic and sampled
                weak connections.

            forward:
                Neural-network forward evaluations.

            reduce:
                Local energy, log-density, weights, energy, and variance.

            backward:
                VJP and optional geometry construction.
        """
        timer = utils.Timer()

        sampler_state = self.sampler_state

        if self.sampler.reset_chains:
            with timer("sample"):
                sampler_state = self.sampler.init(
                    self.params,
                    self.hamiltonian,
                    self.model,
                    key=sampler_state.key,
                    n_alpha=int(self.n_alpha),
                    n_beta=int(self.n_beta),
                    init_method=self.init_method,
                    alpha=(
                        float(sampler_state.alpha)
                        if self.sampler.alpha == "adaptive"
                        else None
                    ),
                    alpha_step=(
                        int(sampler_state.alpha_step)
                        if self.sampler.alpha == "adaptive"
                        else 0
                    ),
                )

        sampler_state, batch, sampler_stats = self.sampler.draw(
            self.params,
            self.hamiltonian,
            self.model,
            sampler_state,
        )

        alpha_step = float(sampler_state.alpha)

        timer.add("sample", sampler_stats.get("time_sample", 0.0))
        timer.add("graph", sampler_stats.get("time_graph", 0.0))
        timer.add("forward", sampler_stats.get("time_forward", 0.0))

        with timer("reduce"):
            kets = libdet.to_dets(batch.dets)

            count_i64 = np.asarray(batch.count, dtype=np.int64)
            count = precision.asarray(count_i64, "calc", "real", host=True)
            sample_mass = precision.asarray(
                np.asarray(batch.mass),
                "calc",
                "real",
                host=True,
            )

            n_ket = int(kets.shape[0])

            # Weak local-energy sampling advances only the RNG key.
            seed = 0
            if self.eloc_sample > 0:
                key, subkey = jax.random.split(sampler_state.key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                sampler_state = replace(sampler_state, key=key)

        with timer("graph"):
            conn_data = self._conns(kets, seed=seed)

        with timer("forward"):
            logpsi_pool_jax = utils.apply(
                self.model.logpsi,
                self.params,
                conn_data["pool_dets"],
            )
            jax.block_until_ready(logpsi_pool_jax)
            logpsi_pool = utils.host(logpsi_pool_jax)

        with timer("reduce"):
            ket_logpsi = jax.tree.map(lambda a: a[conn_data["ket_uid"]], logpsi_pool)

            ket_logabs = precision.asarray(
                np.asarray(to_logabs(ket_logpsi)).reshape(-1),
                "calc",
                "real",
                host=True,
            )

            lognu = self._lognu(
                ket_logabs=ket_logabs,
                logpsi_pool=logpsi_pool,
                conn_data=conn_data,
                n_ket=n_ket,
                alpha=alpha_step,
            )

            eloc, n_conn_weak = self._eloc(
                logpsi_pool=logpsi_pool,
                conn_data=conn_data,
                n_ket=n_ket,
            )

            w, ess = self._weights(
                count=count,
                sample_mass=sample_mass,
                ket_logabs=ket_logabs,
                lognu=lognu,
            )

            energy = float(np.real(np.dot(w, eloc)))
            residual = eloc - energy
            variance = float(np.real(np.dot(w, np.abs(residual) ** 2)))

        gradient = None
        geom = None

        if grad:
            with timer("backward"):
                rdtype = precision.dtype("calc", "real", host=True)

                # grad E = 2 Re <(E_loc - E) O>.
                dlogpsi = rdtype(2.0) * w * residual
                cot = self.model.cotangent(
                    ket_logpsi,
                    precision.asarray(dlogpsi, "calc", host=True),
                )

                gradient = utils.vjp(
                    self.model.coord,
                    self.params,
                    kets,
                    utils.device(precision.asarray(cot, "model", "real", host=True)),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    # Sample-space right hand side for minSR / AdamSR:
                    # b = 2 sqrt(w) (E_loc - E).
                    b_log = rdtype(2.0) * np.sqrt(w) * residual

                    geom = Geometry(
                        theta=self.params,
                        coord=self.model.coord,
                        x=kets,
                        w=utils.device(precision.asarray(w, "sr", "real", host=True)),
                        b=utils.device(
                            precision.asarray(
                                self.model.cotangent(ket_logpsi, b_log),
                                "sr",
                                "real",
                                host=True,
                            )
                        ),
                    )

        sampler_state = self._adaptive(
            sampler_state=sampler_state,
            ket_logabs=ket_logabs,
            eloc=eloc,
            energy=energy,
            w=w,
        )

        new_state = replace(self, sampler_state=sampler_state)

        n_sample = int(np.sum(count_i64))
        n_unique = int(n_ket)

        ess_frac = float(ess / max(1, n_sample))
        unique_frac = float(n_unique / max(1, n_sample))

        n_conn_blur = float(
            sampler_stats.get("n_conn_blur", 0.0) + conn_data["n_conn_blur"]
        )
        n_conn_proposal = float(sampler_stats.get("n_conn_proposal", 0.0))

        stats = {
            "energy": float(energy),
            "variance": float(variance),
            "accept": float(sampler_stats.get("accept", sampler_state.accept)),
            "ess": float(ess),
            "ess_frac": ess_frac,
            "n_sample": float(n_sample),
            "n_unique": float(n_unique),
            "unique_frac": unique_frac,
            "n_eval": float(conn_data["pool_dets"].shape[0]),
            "n_conn_eloc": float(conn_data["n_conn_eloc"]),
            "n_conn_weak": float(n_conn_weak),
            "n_conn_proposal": n_conn_proposal,
            "n_conn_blur": n_conn_blur,
            "alpha": alpha_step,
        }
        stats.update(timer.stats())

        return new_state, energy, gradient, stats, geom

    def _conns(self, kets: np.ndarray, *, seed: int) -> dict[str, Any]:
        """Build blur/local-energy connections and one shared model pool.

        The blur density uses the degree-tilted form

            r_tilde(y)
              = (1 - beta) d_B(y) |psi(y)|^alpha
                + beta sum_x |H_yx| |psi(x)|^alpha.

        Therefore only first-order connections from kets are needed. No
        degrees() call on generated bras is required.
        """
        n_ket = int(kets.shape[0])
        rdtype = precision.dtype("calc", "real", host=True)

        blur_eps = float(
            self.sampler.proposal_eps
            if self.sampler.blur_eps is None
            else self.sampler.blur_eps
        )
        eloc_eps = float(self.eloc_eps1)
        beta = float(np.clip(self.sampler.blur, 0.0, 1.0))

        eloc_conns = self.hamiltonian.conns(kets, eloc_eps)
        eloc_bras = np.ascontiguousarray(
            np.asarray(eloc_conns.bras, dtype=np.uint64)
        )

        parts: list[np.ndarray] = [np.ascontiguousarray(kets), eloc_bras]
        labels: list[str] = ["ket", "eloc"]

        blur_conns = None
        blur_uid_label = ""
        blur_ket_weight = np.zeros(n_ket, dtype=rdtype)
        n_conn_blur = 0

        if beta > 0.0:
            same_conns = blur_eps == eloc_eps

            if same_conns:
                blur_conns = eloc_conns
                blur_uid_label = "eloc"
            else:
                blur_conns = self.hamiltonian.conns(kets, blur_eps)
                blur_bras = np.ascontiguousarray(
                    np.asarray(blur_conns.bras, dtype=np.uint64)
                )
                parts.append(blur_bras)
                labels.append("blur")
                blur_uid_label = "blur"

            n_conn_blur = int(np.asarray(blur_conns.h).size)

            blur_ket_weight = precision.asarray(
                np.asarray(blur_conns.ket_weight),
                "calc",
                "real",
                host=True,
            )

        weak = None

        if self.eloc_sample > 0:
            weak = self.hamiltonian.sample_conns(
                kets,
                int(self.eloc_sample),
                eps1=float(self.eloc_eps1),
                eps2=float(self.eloc_eps2),
                seed=int(seed),
            )

            weak_bras = np.ascontiguousarray(np.asarray(weak.bras, dtype=np.uint64))
            if weak_bras.shape[0] > 0:
                parts.append(weak_bras)
                labels.append("weak")

        pool_dets, _, inv = unique_dets(np.concatenate(parts, axis=0))

        starts: dict[str, int] = {}
        offset = 0

        for label, part in zip(labels, parts, strict=True):
            starts[label] = offset
            offset += int(part.shape[0])

        uid = {
            label: inv[starts[label] : starts[label] + part.shape[0]].astype(np.int64)
            for label, part in zip(labels, parts, strict=True)
        }

        return {
            "pool_dets": pool_dets,
            "ket_uid": uid["ket"],
            "blur_conns": blur_conns,
            "blur_uid": uid.get(blur_uid_label, np.empty(0, dtype=np.int64)),
            "blur_ket_weight": blur_ket_weight,
            "n_conn_blur": n_conn_blur,
            "eloc_conns": eloc_conns,
            "eloc_uid": uid["eloc"],
            "n_conn_eloc": int(np.asarray(eloc_conns.h).size),
            "weak": weak,
            "weak_uid": uid.get("weak", np.empty(0, dtype=np.int64)),
        }

    def _lognu(
        self,
        *,
        ket_logabs: np.ndarray,
        logpsi_pool: Any,
        conn_data: dict[str, Any],
        n_ket: int,
        alpha: float,
    ) -> np.ndarray:
        """Compute the unnormalized observed density.

        Without blur:

            r(y) = |psi(y)|^alpha.

        With degree-tilted Hamiltonian blur:

            r_tilde(y)
              = s(y) B(y|y) |psi(y)|^alpha
                + beta sum_x |H_yx| |psi(x)|^alpha,

        where s(y)=d_B(y) for non-empty blur kets and s(y)=1 for empty kets.
        Thus

            stay scale = (1 - beta) d_B(y), if d_B(y) > 0,
                         1,                 if d_B(y) = 0.

        The off-diagonal term has no source degree denominator.
        """
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        alpha = rdtype(alpha)
        beta = rdtype(np.clip(self.sampler.blur, 0.0, 1.0))

        logrho_ket = precision.asarray(alpha * ket_logabs, "calc", "real", host=True)

        blur_conns = conn_data["blur_conns"]
        if beta <= 0.0 or blur_conns is None:
            return logrho_ket

        ket_weight = precision.asarray(
            conn_data["blur_ket_weight"],
            "calc",
            "real",
            host=True,
        )

        stay_scale = np.where(
            ket_weight > 0.0,
            (rdtype(1.0) - beta) * ket_weight,
            rdtype(1.0),
        )

        log_stay = np.full(n_ket, -np.inf, dtype=rdtype)
        stay_mask = stay_scale > 0.0
        log_stay[stay_mask] = (
            np.log(np.maximum(stay_scale[stay_mask], tiny))
            + logrho_ket[stay_mask]
        )

        ket_ptr = np.asarray(blur_conns.ket_ptr, dtype=np.int64)
        bra = np.asarray(blur_conns.bra, dtype=np.int64)
        h = precision.asarray(np.asarray(blur_conns.h), "calc", "real", host=True)

        if h.size == 0:
            return precision.asarray(log_stay, "calc", "real", host=True)

        blur_uid = conn_data["blur_uid"]

        bra_logabs = precision.asarray(
            np.asarray(
                to_logabs(jax.tree.map(lambda a: a[blur_uid], logpsi_pool))
            ).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        abs_h = np.abs(h)
        valid = abs_h > 0.0

        terms = np.full(h.size, -np.inf, dtype=rdtype)
        terms[valid] = (
            alpha * bra_logabs[bra[valid]]
            + np.log(np.maximum(abs_h[valid], tiny))
        )

        log_blur = np.log(beta) + utils.segment_logsumexp(ket_ptr, terms, n_ket)
        lognu = np.logaddexp(
            log_stay,
            precision.asarray(log_blur, "calc", "real", host=True),
        )

        return precision.asarray(lognu, "calc", "real", host=True)

    def _eloc(
        self,
        *,
        logpsi_pool: Any,
        conn_data: dict[str, Any],
        n_ket: int,
    ) -> tuple[np.ndarray, int]:
        """Compute local energy on observed kets.

        Deterministic screened part:

            E_loc(ket) = H_kk + sum_bra H_bra,ket psi(bra) / psi(ket).

        The optional weak part is an unbiased sampled correction.
        """
        eloc_conns = conn_data["eloc_conns"]
        ket_uid = conn_data["ket_uid"]
        eloc_uid = conn_data["eloc_uid"]

        diag = precision.asarray(
            np.asarray(eloc_conns.diags).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        eloc = precision.asarray(diag.copy(), "calc", host=True)

        ket_ptr = np.asarray(eloc_conns.ket_ptr, dtype=np.int64)
        bra = np.asarray(eloc_conns.bra, dtype=np.int64)
        h = precision.asarray(np.asarray(eloc_conns.h), "calc", "real", host=True)

        if h.size > 0:
            ket = np.repeat(np.arange(n_ket, dtype=np.int64), np.diff(ket_ptr))

            ratio = precision.asarray(
                np.asarray(
                    to_ratio(
                        jax.tree.map(lambda a: a[eloc_uid[bra]], logpsi_pool),
                        jax.tree.map(lambda a: a[ket_uid[ket]], logpsi_pool),
                    )
                ),
                "calc",
                host=True,
            )

            eloc = eloc.astype(np.result_type(eloc, ratio), copy=False)
            np.add.at(eloc, ket, h * ratio)

        weak = conn_data["weak"]
        weak_uid = conn_data["weak_uid"]
        n_conn_weak = 0

        if weak is not None:
            n_conn_weak = int(np.asarray(weak.ket_nconn, dtype=np.int64).sum())

            if weak_uid.size > 0:
                weak_ket = np.asarray(weak.ket, dtype=np.int64)
                weak_h = precision.asarray(
                    np.asarray(weak.h),
                    "calc",
                    "real",
                    host=True,
                )
                weak_pgen = precision.asarray(
                    np.asarray(weak.pgen),
                    "calc",
                    "real",
                    host=True,
                )
                weak_count = precision.asarray(
                    np.asarray(weak.counts),
                    "calc",
                    "real",
                    host=True,
                )

                ratio = precision.asarray(
                    np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[weak_uid], logpsi_pool),
                            jax.tree.map(lambda a: a[ket_uid[weak_ket]], logpsi_pool),
                        )
                    ),
                    "calc",
                    host=True,
                )

                denom = np.maximum(
                    weak_pgen
                    * precision.dtype("calc", "real", host=True)(self.eloc_sample),
                    precision.tiny("calc"),
                )

                eloc = eloc.astype(np.result_type(eloc, ratio), copy=False)
                np.add.at(eloc, weak_ket, (weak_count * weak_h / denom) * ratio)

        return precision.asarray(eloc, "calc", host=True), n_conn_weak

    def _weights(
        self,
        *,
        count: np.ndarray,
        sample_mass: np.ndarray,
        ket_logabs: np.ndarray,
        lognu: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Normalize importance weights.

        For each unique observed determinant y_i:

            u_i    = |psi(y_i)|^2 / r_tilde(y_i),
            mass_i = sample_mass_i u_i,
            w_i    = mass_i / sum_j mass_j.

        With identity observation, sample_mass_i == count_i and this reduces
        to the usual alpha-reference reweighting.

        The ESS reported here is a grouped-mass diagnostic. It is exact when
        all observations merged into the same determinant have the same source
        mass, and otherwise remains a stable concentration diagnostic.
        """
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        count = precision.asarray(count, "calc", "real", host=True)
        sample_mass = precision.asarray(sample_mass, "calc", "real", host=True)

        logu = rdtype(2.0) * ket_logabs - lognu
        finite = np.isfinite(logu)

        if not finite.any():
            raise FloatingPointError("all importance weights are non-finite")

        u = np.zeros_like(logu, dtype=rdtype)
        u[finite] = np.exp(logu[finite] - rdtype(np.max(logu[finite])))

        mass = sample_mass * u
        z = float(mass.sum())

        if not np.isfinite(z) or z <= 0.0:
            raise FloatingPointError("importance weights have zero total mass")

        w = precision.asarray(mass / max(z, float(tiny)), "calc", "real", host=True)

        # Approximate the per-observation denominator from grouped total mass.
        # This reduces to the old exact formula when sample_mass == count.
        safe_count = np.maximum(count, rdtype(1.0))
        ess_denom = float(np.sum((sample_mass * sample_mass / safe_count) * u * u))
        ess = float(z * z / max(ess_denom, float(tiny)))

        return w, ess

    def _adaptive(
        self,
        *,
        sampler_state: WalkerState,
        ket_logabs: np.ndarray,
        eloc: np.ndarray,
        energy: float,
        w: np.ndarray,
    ) -> WalkerState:
        """Update adaptive alpha by KL moment projection.

        The sampler uses

            eta_alpha(x) proportional to |psi(x)|^alpha.

        The residual-tilted Born law

            q*(x) proportional to pi(x) |E_loc(x) - E|

        is projected onto this one-parameter exponential family by matching

            E_eta_alpha[log|psi|] = E_q*[log|psi|].

        The update is averaged with rate 1 / (alpha_step + 1), so the Markov
        kernel changes with diminishing adaptation.
        """
        if self.sampler.alpha != "adaptive":
            return sampler_state

        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        alpha = float(np.clip(float(sampler_state.alpha), 0.0, 2.0))
        alpha_step = int(sampler_state.alpha_step) + 1

        ell = precision.asarray(
            np.asarray(ket_logabs).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        weight = precision.asarray(
            np.asarray(w).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        resid = precision.asarray(
            np.asarray(np.abs(eloc - energy)).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        valid = np.isfinite(ell) & np.isfinite(weight) & np.isfinite(resid)
        valid &= weight > 0.0

        if not valid.any():
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        ell = ell[valid]
        weight = weight[valid]
        resid = resid[valid]

        # Target law: q*(x) proportional to pi(x) |E_loc(x) - E|.
        target = weight * resid
        z_target = float(np.sum(target))

        if not np.isfinite(z_target) or z_target <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        p_target = target / rdtype(z_target)
        m_target = float(np.sum(p_target * ell))

        # Current eta_alpha estimated from Born-weighted samples:
        # eta_alpha / pi is proportional to exp((alpha - 2) log|psi|).
        log_ratio = (rdtype(alpha) - rdtype(2.0)) * ell
        offset = rdtype(np.max(log_ratio))

        q = weight * np.exp(log_ratio - offset)
        z_q = float(np.sum(q))

        if not np.isfinite(z_q) or z_q <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        p_alpha = q / rdtype(z_q)
        m_alpha = float(np.sum(p_alpha * ell))

        centered = ell - rdtype(m_alpha)
        var_alpha = float(np.sum(p_alpha * centered * centered))

        if not np.isfinite(var_alpha) or var_alpha <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        # Newton step for moment matching:
        #     d E_eta_alpha[ell] / d alpha = Var_eta_alpha[ell].
        alpha_hat = alpha + (m_target - m_alpha) / var_alpha
        alpha_hat = float(np.clip(alpha_hat, 0.0, 2.0))

        # Robbins--Monro averaging gives diminishing adaptation.
        rate = 1.0 / float(alpha_step + 1)
        alpha_next = (1.0 - rate) * alpha + rate * alpha_hat
        alpha_next = float(np.clip(alpha_next, 0.0, 2.0))

        return replace(
            sampler_state,
            alpha=alpha_next,
            alpha_step=alpha_step,
        )