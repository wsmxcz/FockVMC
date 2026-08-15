from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import Timer, batch, precision
from .base import ChainState, rank_table, sample_orbitals


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Metropolis sampler with symmetric determinant proposals."""

    n_samples: int = 1024
    n_chains: int = 1024
    thermal_steps: int = 32
    discard_steps: int = 1
    rank: int | None = 1

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive")
        if self.thermal_steps < 0:
            raise ValueError("thermal_steps must be nonnegative")
        if self.discard_steps <= 0:
            raise ValueError("discard_steps must be positive")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise ValueError("rank must be a positive integer or None")

    def init(
        self,
        params: Any,
        model: Any,
        sector: Any,
        *,
        chains: Any,
        key: jax.Array,
        alpha: float,
        timer: Timer | None = None,
    ) -> ChainState:
        """Initialize chains and thermalize them."""
        timer = Timer(timing=False) if timer is None else timer
        x = sector.asarray(chains)
        if x.shape[0] != self.n_chains:
            raise ValueError("chains size must equal sampler.n_chains")

        with timer("unique"):
            unique, _, index = sector.unique(x)
        with timer("forward", n=unique.shape[0]):
            value = batch.apply(model.logabs, params, unique)
            logabs = precision.host(value, "calc", "real").reshape(-1)[index]

        chain = ChainState(key=key, x=x, logabs=logabs)
        for _ in range(self.thermal_steps):
            chain, _ = self.step(
                params,
                model,
                sector,
                chain,
                alpha=alpha,
                timer=timer,
            )
        return chain

    def draw(
        self,
        params: Any,
        model: Any,
        sector: Any,
        chain: ChainState,
        *,
        alpha: float,
        timer: Timer | None = None,
    ) -> tuple[ChainState, np.ndarray, dict[str, float]]:
        """Draw observations and advance all chains."""
        if chain.x.shape[0] != self.n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = Timer(timing=False) if timer is None else timer
        samples = np.empty(
            (self.n_samples, *chain.x.shape[1:]),
            dtype=np.uint64,
        )
        accepted = 0
        proposed = 0
        offset = 0

        while offset < self.n_samples:
            take = min(self.n_chains, self.n_samples - offset)
            key, subkey = jax.random.split(chain.key)
            seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)
            pick = (
                np.arange(self.n_chains, dtype=np.int64)
                if take == self.n_chains
                else rng.choice(self.n_chains, size=take, replace=False)
            )
            chain = replace(chain, key=key)
            samples[offset : offset + take] = chain.x[pick]
            offset += take

            chain, info = self.step(
                params,
                model,
                sector,
                chain,
                alpha=alpha,
                timer=timer,
            )
            accepted += info["accepted"]
            proposed += info["proposed"]

            for _ in range(self.discard_steps - 1):
                chain, info = self.step(
                    params,
                    model,
                    sector,
                    chain,
                    alpha=alpha,
                    timer=timer,
                )
                accepted += info["accepted"]
                proposed += info["proposed"]

        rec = {
            "acceptance_rate": float(accepted / proposed if proposed else 0.0),
        }
        return chain, samples, rec

    def step(
        self,
        params: Any,
        model: Any,
        sector: Any,
        chain: ChainState,
        *,
        alpha: float,
        timer: Timer | None = None,
    ) -> tuple[ChainState, dict[str, int]]:
        """Make one Metropolis transition."""
        timer = Timer(timing=False) if timer is None else timer
        dtype = precision.real("calc", host=True)
        n_chain = chain.x.shape[0]
        ranks, rank_prob, splits, split_prob = rank_table(
            sector.norb,
            sector.n_alpha,
            sector.n_beta,
            self.rank,
        )

        key, subkey = jax.random.split(chain.key)
        seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
        rng = np.random.default_rng(seed)

        if ranks.size == 0:
            return replace(chain, key=key), {"accepted": 0, "proposed": 0}

        with timer("sample"):
            valid = np.full(
                sector.nword,
                np.iinfo(np.uint64).max,
                dtype=np.uint64,
            )
            if sector.norb & 63:
                valid[-1] = (
                    np.uint64(1) << np.uint64(sector.norb & 63)
                ) - np.uint64(1)

            if ranks.size == 1:
                move_a = sector.n_alpha * (sector.norb - sector.n_alpha)
                n_move = move_a + sector.n_beta * (sector.norb - sector.n_beta)
                item = np.arange(n_chain, dtype=np.int64)
                move = rng.integers(n_move, size=n_chain, dtype=np.int64)
                spin = (move >= move_a).astype(np.int64)
                move -= spin * move_a
                n_occ = np.where(spin == 0, sector.n_alpha, sector.n_beta)
                n_vir = sector.norb - n_occ
                occ_rank = move // n_vir
                vir_rank = move % n_vir
                words = chain.x[item, spin]
                virtual = (~words) & valid
                occ_prefix = np.cumulative_sum(
                    np.bitwise_count(words),
                    axis=1,
                    dtype=np.int32,
                    include_initial=True,
                )
                vir_prefix = np.cumulative_sum(
                    np.bitwise_count(virtual),
                    axis=1,
                    dtype=np.int32,
                    include_initial=True,
                )
                occ_word = np.argmax(
                    occ_prefix[:, 1:] > occ_rank[:, None], axis=1
                )
                vir_word = np.argmax(
                    vir_prefix[:, 1:] > vir_rank[:, None], axis=1
                )
                occ_before = occ_prefix[item, occ_word]
                vir_before = vir_prefix[item, vir_word]
                selected = np.stack(
                    (words[item, occ_word], virtual[item, vir_word]),
                    axis=1,
                )
                bits = np.unpackbits(
                    selected.view(np.uint8),
                    axis=1,
                    bitorder="little",
                ).reshape(n_chain, 2, 64)
                bit_prefix = np.cumsum(bits, axis=2, dtype=np.int16)
                occ_bit = np.argmax(
                    bit_prefix[:, 0]
                    == (occ_rank - occ_before)[:, None] + 1,
                    axis=1,
                ).astype(np.uint64)
                vir_bit = np.argmax(
                    bit_prefix[:, 1]
                    == (vir_rank - vir_before)[:, None] + 1,
                    axis=1,
                ).astype(np.uint64)
                proposal = chain.x.copy()
                proposal[item, spin, occ_word] ^= np.uint64(1) << occ_bit
                proposal[item, spin, vir_word] ^= np.uint64(1) << vir_bit
            else:
                total = rng.choice(ranks, size=n_chain, p=rank_prob)
                rank_a = np.empty(n_chain, dtype=np.int32)
                for i, value in enumerate(ranks):
                    row = np.flatnonzero(total == value)
                    rank_a[row] = rng.choice(
                        splits[i], size=row.size, p=split_prob[i]
                    )
                rank_b = total - rank_a
                proposal = chain.x.copy()

                for spin, count, n_elec in (
                    (0, rank_a, sector.n_alpha),
                    (1, rank_b, sector.n_beta),
                ):
                    words = chain.x[:, spin]
                    virtual = (~words) & valid
                    hole = sample_orbitals(words, count, n_elec, rng)
                    part = sample_orbitals(
                        virtual,
                        count,
                        sector.norb - n_elec,
                        rng,
                    )
                    for orbital in (hole, part):
                        row, col = np.nonzero(orbital >= 0)
                        orb = orbital[row, col]
                        np.bitwise_xor.at(
                            proposal[:, spin],
                            (row, orb >> 6),
                            np.uint64(1)
                            << ((orb & 63).astype(np.uint64)),
                        )

        with timer("unique"):
            unique, first, inverse = sector.unique(
                np.concatenate((chain.x, proposal), axis=0)
            )
        with timer("reduce"):
            known = first < n_chain
            logabs = np.empty(unique.shape[0], dtype=dtype)
            logabs[known] = chain.logabs[first[known]]
            unknown = ~known
        if unknown.any():
            with timer("forward", n=np.count_nonzero(unknown)):
                value = batch.apply(model.logabs, params, unique[unknown])
                logabs[unknown] = precision.host(
                    value,
                    "calc",
                    "real",
                ).reshape(-1)

        with timer("sample"):
            new_logabs = logabs[inverse[n_chain:]]
            log_accept = dtype(alpha) * (new_logabs - chain.logabs)
            accept = np.log(rng.random(n_chain)) < np.minimum(
                dtype(0.0), log_accept
            )
            accepted = int(np.count_nonzero(accept))
            proposal[~accept] = chain.x[~accept]
            new_logabs[~accept] = chain.logabs[~accept]

        return (
            ChainState(key=key, x=proposal, logabs=new_logabs),
            {"accepted": accepted, "proposed": n_chain},
        )
