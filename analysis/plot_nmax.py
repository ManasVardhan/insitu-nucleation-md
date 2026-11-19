"""Plot n_max(t) for both label sets from an in situ summary.

    python analysis/plot_nmax.py <prefix>_summary.dat out.png [t_hold0 [t_release]]

Left: largest crystalline cluster n_max(t). Right: second-largest cluster,
which would reveal spontaneous nuclei away from the seed. Optional vertical
markers at the start of the hold and the seed release (reduced time).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from insitu_read import read_summary  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
SERIES = [("mlp", "MLP", "#2a78d6", "-"), ("thr", "q̄6 > 0.308", "#eb6834", "--")]

s = read_summary(sys.argv[1])
marks = [float(x) for x in sys.argv[3:5]]
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, col, title in [(axes[0], "nmax_", "Largest crystalline cluster, n_max"),
                       (axes[1], "n2_", "Second-largest cluster")]:
    for key, lab, c, ls in SERIES:
        y = s[col + key]
        ax.plot(s["time"], y, color=c, lw=2 if col == "nmax_" else 1.2, ls=ls, label=lab)
        if col == "nmax_":
            ax.annotate(f"{lab}  {int(y[-1]):,}", (s["time"][-1], y[-1]),
                        xytext=(-4, 4 if key == "thr" else -6), textcoords="offset points",
                        ha="right", va="top" if key == "mlp" else "bottom", fontsize=8, color=INK)
    for t in marks:
        ax.axvline(t, color=MUTED, lw=0.8, ls=":")
    ax.set_yscale("log" if col == "nmax_" else "linear")
    ax.set_xlabel("time (LJ units)")
    ax.set_ylabel("atoms")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=140)
