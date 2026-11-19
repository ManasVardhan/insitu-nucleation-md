"""I/O written in situ vs a full dump (T3.2 measurement 5).

    python analysis/plot_io.py <prefix>_summary.dat out.png [movie_every_steps]

Left: cumulative bytes over the production run, as logged by the code
(bytes_insitu, bytes_fulldump). The movie snapshots are exactly 13 N + 40
bytes each (src/insitu.h), so the label + summary part is the logged total
minus the snapshots written so far.
Right: bytes for a 100,000-step run vs the analysis interval K for several N.
Full dump: 24 N + 32 bytes per analysis step. In situ: the label + summary
bytes per atom per analysis step measured in the production run (it scales
with the crystalline fraction) times N, plus the same movie (every 1000
steps, independent of K), and without the movie.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from insitu_read import read_summary  # noqa: E402
from plotstyle import BLUE, GREEN, INK, MUTED, ORANGE, VERMILLION, note, plt, title  # noqa: E402

s = read_summary(sys.argv[1])
out = sys.argv[2]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
hdr = open(sys.argv[1]).read().split("\n")[1].split()
N = int(hdr[hdr.index("nglob") + 1]); K = int(hdr[hdr.index("K") + 1])
step = s["step"]
movie = (step // M) * (13 * N + 40)
lab = s["bytes_insitu"] - movie
full = s["bytes_fulldump"]
per_atom_step = lab[-1] / (len(step) * N)   # label + summary bytes per atom per analysis step
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
GB = 1e9
ax[0].plot(s["time"], full / GB, color=VERMILLION, label="full dump every K = %d (24 B/atom)" % K)
ax[0].plot(s["time"], s["bytes_insitu"] / GB, color=BLUE, label="in situ: labels + summary + movie")
ax[0].plot(s["time"], lab / GB, color=GREEN, ls="--", label="in situ: labels + summary only")
for y, c in [(full, VERMILLION), (s["bytes_insitu"], BLUE), (lab, GREEN)]:
    ax[0].annotate("%.2f GB" % (y[-1] / GB), (s["time"][-1], y[-1] / GB), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=8, color=INK)
ax[0].set_yscale("log"); ax[0].set_xlabel("time (LJ units)"); ax[0].set_ylabel("bytes written so far (GB)")
ax[0].legend(loc="lower right", fontsize=8)
title(ax[0], "Production run, N = {:,}: measured".format(N))
S = 100000
Ks = np.array([10, 20, 50, 100, 200, 500, 1000])
for n, c, mk in [(108000, ORANGE, "s"), (1048576, BLUE, "o"), (8388608, GREEN, "D")]:
    na = S / Ks
    f = na * (24 * n + 32)
    ins = na * per_atom_step * n + (S // M) * (13 * n + 40)
    ins0 = na * per_atom_step * n
    ax[1].plot(Ks, f / GB, color=c, marker=mk, label="N = {:,}: full dump".format(n))
    ax[1].plot(Ks, ins / GB, color=c, marker=mk, ls="--", mfc="white", label="in situ (with movie)")
    ax[1].plot(Ks, ins0 / GB, color=c, ls=":", lw=1.5, label="in situ (labels only)")
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("analysis interval K (MD steps)"); ax[1].set_ylabel("bytes for a 100,000-step run (GB)")
ax[1].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
title(ax[1], "Model from the measured label size")
note(fig, "Labels: 5 bytes per crystalline atom (ID + label bits) + 8-byte header; measured mean "
     "%.3f bytes per atom per analysis step over the run (crystal grows to 43%%). "
     "Movie: float32 positions + bits every %d steps." % (per_atom_step, M))
fig.tight_layout(); fig.savefig(out); plt.close(fig)
print("N %d K %d steps %d: full %.2f GB, in situ %.3f GB (ratio %.1f), labels+summary %.3f GB "
      "(ratio %.1f), movie %.3f GB, per atom-step %.4f B" % (
          N, K, len(step), full[-1] / GB, s["bytes_insitu"][-1] / GB, full[-1] / s["bytes_insitu"][-1],
          lab[-1] / GB, full[-1] / lab[-1], movie[-1] / GB, per_atom_step))
for n in [108000, 1048576, 8388608]:
    for k in [50, 100, 200]:
        na = S / k
        print("model N %d K %d: full %.1f GB in situ %.2f GB (labels only %.2f GB)" % (
            n, k, na * (24 * n + 32) / GB, (na * per_atom_step * n + (S // M) * (13 * n + 40)) / GB,
            na * per_atom_step * n / GB))
