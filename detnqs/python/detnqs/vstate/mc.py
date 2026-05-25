from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import libdet
import numpy as np

from detnqs import utils
from ..model.base import Model, to_logabs, to_ratio
from ..optimizer import Geometry
from ..sampler.mcmc import MCSampler, WalkerState
from ..sampler.proposal import unique_dets
from ..utils import precision
from .base import VState


@dataclass(slots=True)
class MCState(VState):
    """Monte Carlo variational state.

    Physical target:
        pi_theta(x) proportional to |psi_theta(x)|^2.

    Markov reference sampled by MCSampler:
        eta_{theta, alpha}(x) proportional to |psi_theta(x)|^alpha.

    Observation kernel:
        x ~ eta_alpha,  y ~ B(y|x).

    Observed reference:
        nu(y) = sum_x eta_alpha(x) B(y|x).

    Unnormalised observed density:
        r_nu(y) = sum_x |psi(x)|^alpha B(y|x).

    Importance weight:
        omega(y) = |psi(y)|^2 / r_nu(y).

    The Hamiltonian graph is built on host by libdet. The neural model is
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
        """One VMC pass: energy, optional gradient, optional geometry.

        Timing convention:
            sampler:
                Counted-state bookkeeping, resampling, accept/reject, and
                observation bookkeeping outside graph/model work.

            graph:
                All libdet graph work: edges, degrees, sample_edges.

            forward:
                All neural-network forward evaluations.

            reduction:
                Local energy, lognu, weights, energy, and variance from
                already-built graphs and evaluated logpsi values.

            backward:
                VJP and optional geometry construction.
        """
        t0 = perf_counter()
        pa = precision.asarray

        time_sampler = 0.0
        time_graph = 0.0
        time_forward = 0.0
        time_reduction = 0.0
        time_backward = 0.0

        sampler_state = self.sampler_state

        if self.sampler.reset_chains:
            t = perf_counter()
            sampler_state = self.sampler.init(
                self.params,
                self.hamiltonian,
                self.model,
                key=sampler_state.key,
                n_alpha=int(self.n_alpha),
                n_beta=int(self.n_beta),
                init_method=self.init_method,
            )
            time_sampler += perf_counter() - t

        t = perf_counter()
        sampler_state, batch, sampler_stats = self.sampler.draw(
            self.params,
            self.hamiltonian,
            self.model,
            sampler_state,
        )
        draw_wall = perf_counter() - t

        time_sampler += float(sampler_stats.get("time_sampler", draw_wall))
        time_graph += float(sampler_stats.get("time_graph", 0.0))
        time_forward += float(sampler_stats.get("time_forward", 0.0))

        t = perf_counter()
        x = libdet.to_dets(batch.dets)
        count_i64 = np.asarray(batch.count, dtype=np.int64)
        count = pa(count_i64, "calc", "real", host=True)
        n_row = int(x.shape[0])

        # Stochastic weak local-energy sampling advances only the RNG key.
        seed = 0
        if self.eloc_sample > 0:
            key, subkey = jax.random.split(sampler_state.key)
            seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
            sampler_state = replace(sampler_state, key=key)

        time_reduction += perf_counter() - t

        t = perf_counter()
        graph = self._graph(x, seed=seed)
        time_graph += perf_counter() - t

        t = perf_counter()
        logpsi_pool = utils.host(utils.apply(self.model.logpsi, self.params, graph["pool"]))
        time_forward += perf_counter() - t

        t = perf_counter()

        row_logpsi = jax.tree.map(lambda a: a[graph["row_uid"]], logpsi_pool)
        row_logabs = pa(
            np.asarray(to_logabs(row_logpsi)).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        lognu = self._lognu(
            row_logabs=row_logabs,
            logpsi_pool=logpsi_pool,
            graph=graph,
            n_row=n_row,
        )

        eloc, n_weak_edges = self._eloc(
            logpsi_pool=logpsi_pool,
            graph=graph,
            n_row=n_row,
        )

        w, ess, ess_unique = self._weights(
            count=count,
            row_logabs=row_logabs,
            lognu=lognu,
        )

        energy = float(np.real(np.dot(w, eloc)))
        residual = eloc - energy
        variance = float(np.real(np.dot(w, np.abs(residual) ** 2)))

        time_reduction += perf_counter() - t

        gradient = None
        geom = None

        if grad:
            t = perf_counter()
            rdtype = precision.dtype("calc", "real", host=True)

            # Energy gradient:
            #
            #   grad E = 2 Re <(E_loc - E) O>,
            #
            # where O is the log-derivative of the wave function.
            dlogpsi = rdtype(2.0) * w * residual
            cot = self.model.cotangent(row_logpsi, pa(dlogpsi, "calc", host=True))

            gradient = utils.vjp(
                self.model.coord,
                self.params,
                x,
                utils.device(pa(cot, "model", "real", host=True)),
            )

            if geometry:
                # Sample-space right hand side for minSR / AdamSR:
                #
                #   b = 2 sqrt(w) (E_loc - E).
                b_log = rdtype(2.0) * np.sqrt(w) * residual

                geom = Geometry(
                    theta=self.params,
                    coord=self.model.coord,
                    x=x,
                    w=utils.device(pa(w, "sr", "real", host=True)),
                    b=utils.device(
                        pa(
                            self.model.cotangent(row_logpsi, b_log),
                            "sr",
                            "real",
                            host=True,
                        )
                    ),
                )

            time_backward += perf_counter() - t

        new_state = replace(self, sampler_state=sampler_state)

        n_chains = max(1, int(np.sum(sampler_state.count)))
        n_unique_chains = int(sampler_state.dets.shape[0])
        n_samples = int(np.sum(count_i64))
        n_unique_samples = int(n_row)

        chain_unique_ratio = float(n_unique_chains / n_chains)
        sample_unique_ratio = float(n_unique_samples / max(1, n_samples))

        stats = {
            "energy": float(energy),
            "variance": float(variance),
            "accept": float(sampler_stats.get("accept", sampler_state.accept)),
            "ess": float(ess),
            "ess_unique": float(ess_unique),
            
            "unique_chains": float(n_unique_chains),
            "chain_uratio": float(chain_unique_ratio),
            "unique_samples": float(n_unique_samples),
            "sample_uratio": float(sample_unique_ratio),
            "n_eval": float(graph["pool"].shape[0]),

            "time_sampler": float(time_sampler),
            "time_graph": float(time_graph),
            "time_forward": float(time_forward),
            "time_reduction": float(time_reduction),
            "time_backward": float(time_backward),
            "time_total": float(perf_counter() - t0),
        }

        return new_state, energy, gradient, stats, geom

    def _graph(self, x: np.ndarray, *, seed: int) -> dict[str, Any]:
        """Build blur/local-energy graphs and one shared model-eval pool."""
        n_row = int(x.shape[0])
        rdtype = precision.dtype("calc", "real", host=True)
        pa = precision.asarray

        blur_eps = float(
            self.sampler.proposal_eps if self.sampler.blur_eps is None else self.sampler.blur_eps
        )
        eloc_eps = float(self.eloc_eps1)
        beta = float(np.clip(self.sampler.blur, 0.0, 1.0))

        eloc_graph = self.hamiltonian.edges(x, eloc_eps)
        eloc_col_dets = np.ascontiguousarray(np.asarray(eloc_graph.col_dets, dtype=np.uint64))

        parts: list[np.ndarray] = [np.ascontiguousarray(x), eloc_col_dets]
        labels: list[str] = ["row", "eloc"]

        blur_graph = None
        blur_uid_label = ""
        blur_row_weight = np.zeros(n_row, dtype=rdtype)
        blur_source_weight = np.empty(0, dtype=rdtype)
        n_blur_edges = 0

        if beta > 0.0:
            same_graph = blur_eps == eloc_eps

            if same_graph:
                blur_graph = eloc_graph
                blur_uid_label = "eloc"
            else:
                blur_graph = self.hamiltonian.edges(x, blur_eps)
                blur_col_dets = np.ascontiguousarray(
                    np.asarray(blur_graph.col_dets, dtype=np.uint64)
                )
                parts.append(blur_col_dets)
                labels.append("blur")
                blur_uid_label = "blur"

            n_blur_edges = int(np.asarray(blur_graph.h).size)
            blur_row_weight = pa(
                np.asarray(blur_graph.row_weight),
                "calc",
                "real",
                host=True,
            )

            # For lognu(y), edges are traversed as y -> x, so the source
            # degree d_B(x) is needed for every column determinant.
            blur_col_dets_for_deg = (
                eloc_col_dets
                if same_graph
                else np.ascontiguousarray(np.asarray(blur_graph.col_dets, dtype=np.uint64))
            )

            deg = self.hamiltonian.degrees(blur_col_dets_for_deg, blur_eps)
            blur_source_weight = pa(
                np.asarray(deg.row_weight),
                "calc",
                "real",
                host=True,
            )

        weak = None

        if self.eloc_sample > 0:
            weak = self.hamiltonian.sample_edges(
                x,
                int(self.eloc_sample),
                eps1=float(self.eloc_eps1),
                eps2=float(self.eloc_eps2),
                seed=int(seed),
            )

            weak_dets = np.ascontiguousarray(np.asarray(weak.dets, dtype=np.uint64))
            if weak_dets.shape[0] > 0:
                parts.append(weak_dets)
                labels.append("weak")

        pool, _, inv = unique_dets(np.concatenate(parts, axis=0))

        starts: dict[str, int] = {}
        offset = 0

        for label, part in zip(labels, parts):
            starts[label] = offset
            offset += int(part.shape[0])

        uid: dict[str, np.ndarray] = {
            label: inv[starts[label] : starts[label] + parts[i].shape[0]].astype(np.int64)
            for i, label in enumerate(labels)
        }

        return {
            "pool": pool,
            "row_uid": uid["row"],
            "blur_graph": blur_graph,
            "blur_uid": uid.get(blur_uid_label, np.empty(0, dtype=np.int64)),
            "blur_row_weight": blur_row_weight,
            "blur_source_weight": blur_source_weight,
            "n_blur_edges": n_blur_edges,
            "eloc_graph": eloc_graph,
            "eloc_uid": uid["eloc"],
            "n_eloc_edges": int(np.asarray(eloc_graph.h).size),
            "weak": weak,
            "weak_uid": uid.get("weak", np.empty(0, dtype=np.int64)),
        }

    def _lognu(
        self,
        *,
        row_logabs: np.ndarray,
        logpsi_pool: Any,
        graph: dict[str, Any],
        n_row: int,
    ) -> np.ndarray:
        """Compute log unnormalised observed reference density r_nu(y).

        For beta > 0:

            r_nu(y) =
                (1 - beta_y) |psi(y)|^alpha
                + beta sum_x |psi(x)|^alpha |H_xy| / d_B(x).

        If d_B(y) = 0, beta_y = 0 and the stay term has coefficient 1.
        """
        pa = precision.asarray
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        alpha = rdtype(self.sampler.alpha)
        beta = rdtype(np.clip(self.sampler.blur, 0.0, 1.0))

        logrho_row = pa(alpha * row_logabs, "calc", "real", host=True)

        blur_graph = graph["blur_graph"]
        if beta <= 0.0 or blur_graph is None:
            return logrho_row

        row_weight = pa(graph["blur_row_weight"], "calc", "real", host=True)
        beta_y = np.where(row_weight > 0.0, beta, rdtype(0.0))
        stay = rdtype(1.0) - beta_y

        log_stay = np.full(n_row, -np.inf, dtype=rdtype)
        stay_mask = stay > 0.0
        log_stay[stay_mask] = np.log(stay[stay_mask]) + logrho_row[stay_mask]

        row_ptr = np.asarray(blur_graph.row_ptr, dtype=np.int64)
        col = np.asarray(blur_graph.col, dtype=np.int64)
        h = pa(np.asarray(blur_graph.h), "calc", "real", host=True)

        if h.size == 0:
            return pa(log_stay, "calc", "real", host=True)

        blur_uid = graph["blur_uid"]

        source_logabs = pa(
            np.asarray(to_logabs(jax.tree.map(lambda a: a[blur_uid], logpsi_pool))).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        source_weight = pa(graph["blur_source_weight"], "calc", "real", host=True)

        abs_h = np.abs(h)
        source_deg = source_weight[col]
        valid = (source_deg > 0.0) & (abs_h > 0.0)

        terms = np.full(h.size, -np.inf, dtype=rdtype)
        terms[valid] = (
            alpha * source_logabs[col[valid]]
            + np.log(np.maximum(abs_h[valid], tiny))
            - np.log(np.maximum(source_deg[valid], tiny))
        )

        log_blur = np.log(beta) + utils.segment_logsumexp(row_ptr, terms, n_row)
        lognu = np.logaddexp(log_stay, pa(log_blur, "calc", "real", host=True))

        return pa(lognu, "calc", "real", host=True)

    def _eloc(
        self,
        *,
        logpsi_pool: Any,
        graph: dict[str, Any],
        n_row: int,
    ) -> tuple[np.ndarray, int]:
        """Compute local energy on observed determinants.

        Deterministic screened part:

            E_loc(x) = H_xx + sum_y H_xy psi(y) / psi(x).

        Optional weak part is added as an unbiased sampled correction.
        """
        pa = precision.asarray

        eloc_graph = graph["eloc_graph"]
        row_uid = graph["row_uid"]
        eloc_uid = graph["eloc_uid"]

        diag = pa(np.asarray(eloc_graph.diags).reshape(-1), "calc", "real", host=True)
        eloc = pa(diag.copy(), "calc", host=True)

        row_ptr = np.asarray(eloc_graph.row_ptr, dtype=np.int64)
        col = np.asarray(eloc_graph.col, dtype=np.int64)
        h = pa(np.asarray(eloc_graph.h), "calc", "real", host=True)

        if h.size > 0:
            rows = np.repeat(np.arange(n_row, dtype=np.int64), np.diff(row_ptr))

            ratio = pa(
                np.asarray(
                    to_ratio(
                        jax.tree.map(lambda a: a[eloc_uid[col]], logpsi_pool),
                        jax.tree.map(lambda a: a[row_uid[rows]], logpsi_pool),
                    )
                ),
                "calc",
                host=True,
            )

            eloc = eloc.astype(np.result_type(eloc, ratio), copy=False)
            np.add.at(eloc, rows, h * ratio)

        weak = graph["weak"]
        weak_uid = graph["weak_uid"]
        n_weak_edges = 0

        if weak is not None and weak_uid.size > 0:
            weak_rows = np.asarray(weak.rows, dtype=np.int64)
            weak_h = pa(np.asarray(weak.h), "calc", "real", host=True)
            weak_pgen = pa(np.asarray(weak.pgen), "calc", "real", host=True)
            weak_count = pa(np.asarray(weak.counts), "calc", "real", host=True)
            n_weak_edges = int(weak_h.size)

            ratio = pa(
                np.asarray(
                    to_ratio(
                        jax.tree.map(lambda a: a[weak_uid], logpsi_pool),
                        jax.tree.map(lambda a: a[row_uid[weak_rows]], logpsi_pool),
                    )
                ),
                "calc",
                host=True,
            )

            denom = np.maximum(
                weak_pgen * precision.dtype("calc", "real", host=True)(self.eloc_sample),
                precision.tiny("calc"),
            )

            eloc = eloc.astype(np.result_type(eloc, ratio), copy=False)
            np.add.at(eloc, weak_rows, (weak_count * weak_h / denom) * ratio)

        return pa(eloc, "calc", host=True), n_weak_edges

    def _weights(
        self,
        *,
        count: np.ndarray,
        row_logabs: np.ndarray,
        lognu: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Normalise importance weights and compute ESS diagnostics.

        For each unique observed determinant y_i:

            u_i = |psi(y_i)|^2 / r_nu(y_i),
            mass_i = count_i u_i,
            w_i = mass_i / sum_j mass_j.

        ``ess`` is the effective sample size over counted samples.
        ``ess_unique`` is the support-level inverse participation ratio.
        """
        pa = precision.asarray
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        logu = rdtype(2.0) * row_logabs - lognu
        finite = np.isfinite(logu)

        if not finite.any():
            raise FloatingPointError("all importance weights are non-finite")

        u = np.zeros_like(logu, dtype=rdtype)
        u[finite] = np.exp(logu[finite] - rdtype(np.max(logu[finite])))

        mass = count * u
        z = float(mass.sum())

        if not np.isfinite(z) or z <= 0.0:
            raise FloatingPointError("importance weights have zero total mass")

        w = pa(mass / max(z, float(tiny)), "calc", "real", host=True)

        ess = float(z * z / max(float(np.dot(count * u, u)), float(tiny)))
        ess_unique = float(rdtype(1.0) / max(float(np.dot(w, w)), float(tiny)))

        return w, ess, ess_unique