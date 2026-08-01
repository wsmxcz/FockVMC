from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

DATA_PATHS = [
    Path("fe2s2/fe2s2_GBackflow.jsonl"), 
    Path("fe2s2/fe2s2_SBackflow.jsonl"), 
    Path("fe2s2/fe2s2_PBackflow.jsonl")
]
BENCHMARK = -116.6056091
OUT_PATH = Path("convergence.pdf")

COLORS = {
    "ref1": "#1E2433", "ref2": "#667083", "ref3": "#AAB4C2",
    "method1": "#EE4B96", "method2": "#347ED8", "method3": "#E8BB32",
    "method4": "#16A69C", "method5": "#8457D3", "method6": "#F06A32",
    "method7": "#57A957", "method8": "#292D3A",
    "bar": "#F4F7FB",
}

has_benchmark = BENCHMARK is not None and np.isfinite(float(BENCHMARK))

runs = []
for i, data_path in enumerate(DATA_PATHS):
    with data_path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    step = np.array([r.get("step", j) for j, r in enumerate(rows)], dtype=float)
    energy = np.array([r.get("energy", np.nan) for r in rows], dtype=float)
    eloc_var = np.array([r.get("eloc_var", np.nan) for r in rows], dtype=float)
    accept = np.array([r.get("accept", np.nan) for r in rows], dtype=float)
    alpha = np.array([r.get("alpha", np.nan) for r in rows], dtype=float)
    time_total = np.cumsum(np.array([r.get("time_total", np.nan) for r in rows], dtype=float))
    s2 = np.array([r.get("s2", np.nan) for r in rows], dtype=float)
    error = np.abs(energy - float(BENCHMARK)) if has_benchmark else np.full_like(energy, np.nan)

    runs.append({
        "label": data_path.stem,
        "color": COLORS[f"method{i+1}"],
        "step": step, "energy": energy, "eloc_var": eloc_var,
        "accept": accept, "alpha": alpha, "time_total": time_total,
        "s2": s2, "error": error,
    })

    finite_energy = energy[np.isfinite(energy)]
    print(f"{data_path}: {len(rows)} records")
    if finite_energy.size:
        final_energy = finite_energy[-1]
        print(f"  final energy: {final_energy:.8f}")
        if has_benchmark:
            print(f"  final error: {final_energy - float(BENCHMARK):+.8f}")

fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.0), layout="constrained")
ax = ax.ravel()
alpha_ax = ax[3].twinx()

for run in runs:
    c = run["color"]
    ax[0].plot(run["step"], run["energy"], color=c, lw=1.6, label=run["label"])
    ax[1].plot(run["step"], run["error"], color=c, lw=1.6)
    ax[2].plot(run["step"], run["eloc_var"], color=c, lw=1.6)
    ax[3].plot(run["step"], run["accept"], color=c, lw=1.6)
    alpha_ax.plot(run["step"], run["alpha"], color=c, lw=1.3, ls="--")
    ax[4].plot(run["step"], run["s2"], color=c, lw=1.6)
    ax[5].plot(run["step"], run["time_total"], color=c, lw=1.6)

if has_benchmark:
    ax[0].axhline(float(BENCHMARK), color=COLORS["ref1"], lw=1.0, ls="--", label="benchmark")
    ax[1].set_yscale("log")
else:
    ax[1].set_visible(False)

ax[2].set_yscale("log")

ylabels = ["energy", "|energy - benchmark|", "variance", "accept", r"$S^2$", "wall time"]
for a, label in zip(ax, ylabels):
    if a.get_visible():
        a.set_ylabel(label)
        a.set_xlabel("step")
        a.grid(True, color=COLORS["bar"], lw=0.8)

alpha_ax.set_ylabel("alpha")

if len(runs) > 1 or has_benchmark:
    ax[0].legend(frameon=False)

fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"saved: {OUT_PATH}")
plt.show()