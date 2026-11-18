"""Plot n_max(t) and the crystalline count from an in situ summary.

    python analysis/plot_nmax.py runs/production/prod_summary.dat out.png [t_hold0]

t_hold0 (optional) marks the start of the hold (reduced time).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from insitu_read import read_summary  # noqa: E402

s = read_summary(sys.argv[1])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for key, lab, c in [("mlp", "MLP", "#2a6fdb"), ("thr", r"$\bar q_6 > 0.308$", "#e07b39")]:
    ax[0].plot(s["time"], s["nmax_" + key], color=c, label=lab)
    ax[1].plot(s["time"], s["n_crystal_" + key], color=c, label=lab)
    ax[1].plot(s["time"], s["n2_" + key], color=c, ls=":", lw=1,
               label=lab + ", 2nd largest cluster")
for a, t in zip(ax, ["largest crystalline cluster $n_{max}$", "crystalline atoms"]):
    a.set_xlabel("time (LJ units)")
    a.set_title(t)
    a.set_yscale("log")
    if len(sys.argv) > 3:
        a.axvline(float(sys.argv[3]), color="gray", lw=0.8, ls="--")
    a.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
