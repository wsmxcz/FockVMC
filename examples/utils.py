import numpy as np
import matplotlib.pyplot as plt


def print_metrics(metrics, stats=None):
    header = "  ".join(f"{label:^{width}}" for label, _, width in metrics.values())
    line = "-" * len(header)

    if stats is None:
        print(line, header, line, sep="\n")
        return line

    row = []
    for key, (_, fmt, width) in metrics.items():
        value = stats.get(key, np.nan)

        try:
            value = float(value)
        except Exception:
            value = np.nan

        if fmt == "d":
            text = f"{int(value):d}" if np.isfinite(value) else "nan"
        else:
            text = f"{value:{fmt}}" if np.isfinite(value) else "nan"

        row.append(f"{text:^{width}}")

    print("  ".join(row))


def print_times(times):
    items = []
    for key, value in times.items():
        name = key[5:] if key.startswith("time_") else key
        if name != "total":
            items.append((name, float(value)))

    items = sorted(items, key=lambda item: item[1], reverse=True)
    if items:
        print("Time:", "  ".join(f"{key}={value:.2f}s" for key, value in items))


COLORS = {
    # References (Neutral slate gradient)
    'ref1':     '#2B2D42',  # Dark slate blue (Primary reference / Ground Truth)
    'ref2':     '#5C677D',  # Slate gray (Secondary reference / Baseline 1)
    'ref3':     '#8D99AE',  # Cool gray (Tertiary reference / Baseline 2)

    # Methods (Muted Morandi spectrum)
    'method1':  '#4A809D',  # Soft ocean blue (Recommended for proposed method)
    'method2':  '#5FA89B',  # Soft teal / Mint
    'method3':  '#8EAF74',  # Sage green
    'method4':  '#E2AF5E',  # Muted amber / Yellow
    'method5':  '#E38562',  # Soft coral / Orange
    'method6':  '#D67B8C',  # Dusty rose / Pink
    'method7':  '#A38CB4',  # Soft lavender / Purple
    'method8':  '#7FA1C3',  # Ice blue

    # Auxiliaries & Backgrounds
    'aux':      '#A2C9E4',  # Soft sky blue (For shaded areas or annotations)
    'bar':      '#EAECEF',  # Light gray (For bar fills or background grids)
    'bar_edge': '#C4CCD3',  # Slate gray (For bar borders or grid lines)
}


def plot_convergence(history, e_ref):
    energy = np.asarray(history, dtype=float)
    step = np.arange(energy.size)
    error = np.abs(energy - float(e_ref))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)

    axes[0].plot(step, energy, color=COLORS['method1'], linewidth=1.5, label='Energy')
    axes[0].axhline(e_ref, color=COLORS['ref1'], linestyle='--', linewidth=1.0, label='FCI')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Energy / Ha')
    axes[0].legend(frameon=False, fontsize='small', loc='best')
    axes[0].tick_params(direction='in')
    axes[0].set_axisbelow(True)
    axes[0].grid(color=COLORS['bar'], linestyle='-', linewidth=0.4, alpha=0.7)

    axes[1].semilogy(step, np.where(error > 0, error, np.nan),
                     color=COLORS['method2'], linewidth=1.5)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel(r'$|E - E_\mathrm{FCI}|$ / Ha')
    axes[1].tick_params(direction='in')
    axes[1].set_axisbelow(True)
    axes[1].grid(color=COLORS['bar'], linestyle='-', linewidth=0.4, alpha=0.7)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)

    return fig, axes