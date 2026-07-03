from __future__ import annotations

from dataclasses import replace

from detnqs import utils
from vmc import build, configure


CHECKPOINT = "N2_01000.npz"
N_BLOCKS = 8


def main() -> None:
    configure()

    vmc, obs = build(seed=0)
    vmc.load(CHECKPOINT)

    state = vmc.state.replace(
        sampler=replace(
            vmc.state.sampler,
            n_samples=819200,
            alpha=float(vmc.state.sampler_state.alpha),
        )
    )

    # Estimate posterior observables.
    post = utils.analysis.estimate(state, n_blocks=N_BLOCKS, obs=obs, profile=True)
    _, snap, data = state.expect(obs=obs, profile=True, data=True)

    # Analyze support, excitation structure, and tail behavior.
    sup = utils.analysis.support(data["w"])
    exc = utils.analysis.excitation(data["x"], data["w"], state.H.sector.reference(1))
    tail = utils.analysis.tail(data["w"], data["eloc"], snap["energy"])

    # Probe SR geometry at the restored point.
    _, _, _, _, geometry = state.expect_and_grad(geometry=True, obs=obs, profile=True)
    sr = utils.analysis.sr(geometry, shift=1.0e-3)

    print("Posterior")
    print(f"Energy     : {post['energy']:.12f}")
    print(f"Energy SE  : {post['energy_se']:.3e}")
    print(f"Eloc var   : {post['eloc_var']:.6e}")
    print(f"ESS frac   : {post['ess_frac']:.3%}")
    print(f"ESSu frac  : {post['essu_frac']:.3%}")
    print(f"W max      : {post['w_max']:.3e}")
    print(f"S^2        : {post['s2']:.8f}")
    print(f"S^2 var    : {post['s2_var']:.3e}")

    print("\nSupport")
    print(f"N2         : {sup['n2']:.3f}")
    print(f"Entropy    : {sup['entropy']:.6f}")
    print(f"Top 1      : {sup['top1']:.3%}")
    print(f"Top 5      : {sup['top5']:.3%}")
    print(f"Top 10     : {sup['top10']:.3%}")
    print(f"Top 100    : {sup['top100']:.3%}")

    print("\nExcitation")
    for key in sorted(exc, key=lambda s: int(s[3:])):
        print(f"{key:<10s} : {exc[key]:.3%}")

    print("\nTail")
    print(f"R q90      : {tail['r_q90']:.6e}")
    print(f"R q99      : {tail['r_q99']:.6e}")
    print(f"R q999     : {tail['r_q999']:.6e}")
    print(f"Var max    : {tail['var_max']:.6e}")

    print("\nSR")
    print(f"Tr QGT     : {sr['tr_qgt']:.6e}")
    print(f"Rank eff   : {sr['rank_eff']:.3f}")
    print(f"SR force   : {sr['sr_force']:.6e}")
    print(f"SR damp    : {sr['sr_damp']:.6e}")


if __name__ == "__main__":
    main()
