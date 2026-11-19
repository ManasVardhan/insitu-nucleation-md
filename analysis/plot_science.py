"""n_max(t) for the MLP and q6-bar threshold labels, and the crystalline fraction.

    python analysis/plot_science.py <prefix>_summary.dat <input file> out.png [label]
           [--second <summary> <label>]

Reads the protocol (StepMelt, StepQuench, StepRelease, DeltaT) from the
input file and marks the quench ramp, the start of the hold and the seed
release. An optional second run is drawn thin for comparison.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from insitu_read import read_summary  # noqa: E402
from plotstyle import BLUE, GREY, INK, MUTED, ORANGE, note, plt, title  # noqa: E402

args = sys.argv[1:]
second = None
if "--second" in args:
    i = args.index("--second"); second = args[i + 1:i + 3]; args = args[:i]
summ, inp, out = args[:3]
lab = args[3] if len(args) > 3 else ""
kv = {}
for line in open(inp):
    p = line.split("#")[0].split()
    if len(p) >= 2:
        kv[p[0]] = p[1:]
dt = float(kv.get("DeltaT", ["0.005"])[0])
sm, sq, sr = (int(kv[k][0]) for k in ("StepMelt", "StepQuench", "StepRelease"))
t_ramp0, t_hold, t_rel = sm * dt, (sm + sq) * dt, (sm + sq + sr) * dt
s = read_summary(summ)
hdr = open(summ).read().split("\n")[1].split()
N = int(hdr[hdr.index("nglob") + 1])
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for a in ax:
    a.axvspan(t_ramp0, t_hold, color="#f0efe9", lw=0)
    a.axvline(t_rel, color=MUTED, lw=1, ls=":")
L = [("mlp", "MLP", BLUE, "-"), ("thr", "q6-bar > 0.308", ORANGE, "--")]
for key, name, c, ls in L:
    y = s["nmax_" + key]
    ax[0].plot(s["time"], y, color=c, ls=ls, label="largest cluster, " + name)
    ax[0].annotate("{:,}".format(int(y[-1])), (s["time"][-1], y[-1]),
                   xytext=(4, 6 if key == "thr" else -6), textcoords="offset points",
                   va="center", fontsize=8, color=INK)
    ax[1].plot(s["time"], s["n_crystal_" + key] / N, color=c, ls=ls, label=name)
if second:
    s2 = read_summary(second[0])
    for key, name, c, ls in L:
        ax[0].plot(s2["time"], s2["nmax_" + key], color=c, ls=ls, lw=0.9, alpha=0.55)
    ax[0].plot([], [], color=GREY, lw=0.9, label=second[1] + " (thin)")
ax[0].set_yscale("log"); ax[0].set_ylim(100, N)
ax[0].set_xlabel("time (LJ units)"); ax[0].set_ylabel("atoms in the largest crystalline cluster")
ax[0].legend(loc="lower right", fontsize=8)
title(ax[0], "Largest nucleus n_max(t)" + (" - " + lab if lab else ""))
ax[1].set_xlabel("time (LJ units)"); ax[1].set_ylabel("crystalline fraction of all atoms")
ax[1].set_ylim(0, max(0.05, 1.08 * max(s["n_crystal_mlp"].max(), s["n_crystal_thr"].max()) / N))
ax[1].legend(loc="center left", fontsize=8)
title(ax[1], "Crystalline fraction")
for a in ax:
    yl = a.get_ylim()
    ytop = 0.97 * yl[1] if a.get_yscale() == "linear" else yl[1] ** 0.97 * yl[0] ** 0.03
    a.text(t_ramp0, ytop, "quench ", ha="right", va="top", fontsize=8, color=MUTED)
    a.text(t_rel, ytop, " seed released", ha="left", va="top", fontsize=8, color=MUTED)
note(fig, "N = {:,}, rho* = 0.95, T_q = 0.70, FCC seed R = 4.5 (369 atoms) frozen until t = {:.0f}. "
     "Shaded: linear quench from T* = 2.0; the hold starts at t = {:.0f}. Every K = 100 steps.".format(
         N, t_rel, t_hold))
fig.tight_layout(); fig.savefig(out); plt.close(fig)
