from __future__ import annotations

from detnqs import utils
from vmc import CHECKPOINT_EVERY, CKPT, LOG, build, configure


START = "N2_00500.npz"
TARGET_STEPS = 1000


def main() -> None:
    configure()

    vmc, obs = build(seed=0)
    vmc.load(START)

    log = utils.Logger(file=LOG, every=10, append=True)
    vmc.run(
        TARGET_STEPS - vmc.step_count,
        obs=obs,
        logger=log,
        profile=True,
        checkpoint=CKPT,
        checkpoint_every=CHECKPOINT_EVERY,
    )


if __name__ == "__main__":
    main()
