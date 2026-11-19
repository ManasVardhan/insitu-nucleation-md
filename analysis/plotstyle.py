"""Shared matplotlib style for the report figures.

Okabe-Ito colors (colorblind safe), in a fixed order; every series also gets
its own marker or line style and a direct label or legend, so identity never
rests on color alone.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
BLUE, ORANGE, GREEN, VERMILLION, PINK, SKY = \
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
GREY = "#8a8984"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.dpi": 100,
})


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, color=INK)


def note(fig, text):
    """Small caption line under the axes (data source, hardware)."""
    fig.text(0.01, -0.02, text, fontsize=8, color=MUTED, ha="left", va="top")
