"""T3.2 performance figures and tables from runs/perf/results/.

    python analysis/plot_perf.py [results dir] [figure dir]

Reads every <exp>_<jobid>.csv written by scripts/perf_lib.sh (smoke runs are
skipped), the in situ summaries summary_<jobid>_<tag>_r<rep>.dat and the
drift_<jobid>_<tag>.wall files. Writes docs/img/perf_*.png and prints the
tables of docs/results.md (mean +- sample std over repeats).
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from insitu_read import read_summary  # noqa: E402
from plotstyle import (BLUE, GREEN, GREY, INK, MUTED, ORANGE, PINK,  # noqa: E402
                       VERMILLION, note, plt, title)

RES = sys.argv[1] if len(sys.argv) > 1 else "runs/perf/results"
IMG = sys.argv[2] if len(sys.argv) > 2 else "docs/img"
HW = "debug node e23-02: 2x Tesla P100-PCIE-16GB, 2x Xeon E5-2640 v4 (10 cores each)"


def load():
    fs = [f for f in glob.glob(os.path.join(RES, "*.csv")) if "smoke" not in f]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df


def agg(d, keys):
    g = d.groupby(keys)["step_ms"]
    return g.agg(["mean", "std", "count"]).reset_index()


def md(df, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(c for c, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for _, k in cols:
            v = r[k]
            f = fmt.get(k)
            cells.append(f(v) if f else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pm(m, s):
    return "%.2f ± %.2f" % (m, 0 if np.isnan(s) else s)


df = load()
print("rows:", len(df), "experiments:", sorted(df.exp.unique()))

# ---------------------------------------------------------------- speedup
sp = df[df.exp == "speedup"]
if len(sp):
    a = agg(sp, ["config", "nglob"])
    f = sp.groupby(["config", "nglob"])["force_ms"].mean().reset_index()
    a = a.merge(f, on=["config", "nglob"])
    piv = a.pivot(index="nglob", columns="config", values="mean")
    sd = a.pivot(index="nglob", columns="config", values="std")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    S = [("cpu1", "CPU, 1 core", VERMILLION, "s"), ("cpu10", "CPU, 10 cores (1 socket)", ORANGE, "^"),
         ("cpu20", "CPU, 20 cores (node)", PINK, "v"), ("gpu1", "1 P100", BLUE, "o"),
         ("gpu2", "2 P100", GREEN, "D")]
    for c, lab, col, mk in S:
        if c not in piv:
            continue
        ax[0].errorbar(piv.index, piv[c], yerr=sd[c], color=col, marker=mk, label=lab, capsize=2)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("atoms N"); ax[0].set_ylabel("time per MD step (ms)")
    ax[0].legend(loc="upper left")
    title(ax[0], "Step time vs N")
    R = [("cpu1", "gpu1", "1 P100 vs 1 core", VERMILLION, "s"),
         ("cpu10", "gpu1", "1 P100 vs 1 socket", ORANGE, "^"),
         ("cpu20", "gpu1", "1 P100 vs 20-core node", PINK, "v"),
         ("cpu20", "gpu2", "2 P100 vs 20-core node", GREEN, "D")]
    for c0, c1, lab, col, mk in R:
        sr = piv[c0] / piv[c1]
        ax[1].plot(piv.index, sr, color=col, marker=mk, label=lab)
        dy = {"cpu10": -5, "cpu20": 0, "cpu1": 0}[c0] if c1 == "gpu1" else 5
        ax[1].annotate("%.1fx" % sr.iloc[-1], (piv.index[-1], sr.iloc[-1]), xytext=(5, dy),
                       textcoords="offset points", va="center", fontsize=8, color=INK)
    ax[1].axhline(1, color=MUTED, lw=0.8, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("atoms N"); ax[1].set_ylabel("speedup (CPU time / GPU time)")
    ax[1].set_ylim(0.5, 70); ax[1].legend(loc="lower left")
    title(ax[1], "GPU speedup")
    g = a[a.config == "gpu1"].sort_values("nglob")
    x = np.arange(len(g))
    ax[2].bar(x, g["force_ms"], color=BLUE, width=0.6, label="force call (GPU kernels + copies)")
    ax[2].bar(x, g["mean"] - g["force_ms"], bottom=g["force_ms"], color=GREY, width=0.6,
              label="host: integration, halo, migration")
    for i, (fm, m) in enumerate(zip(g["force_ms"], g["mean"])):
        ax[2].text(i, m, "%.0f%%" % (100 * (m - fm) / m), ha="center", va="bottom", fontsize=8)
    ax[2].set_xticks(x, ["%dk" % (n // 1000) for n in g.nglob])
    ax[2].set_xlabel("atoms N"); ax[2].set_ylabel("ms per step, 1 rank + 1 P100")
    ax[2].legend(loc="upper left"); ax[2].grid(axis="x", visible=False)
    title(ax[2], "1-GPU step: force call vs host (host %)")
    note(fig, "Same global box for every layout (rho* = 0.95, FCC start at T* = 2.0, NVE). " + HW +
         ". Error bars: sample std over 3 repeats (cpu1: 2).")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "perf_speedup.png")); plt.close(fig)
    t = a.pivot(index="nglob", columns="config", values="mean").reset_index()
    t["s_core"] = t.cpu1 / t.gpu1; t["s_node"] = t.cpu20 / t.gpu1; t["s_node2"] = t.cpu20 / t.gpu2
    print("\n### speedup\n")
    print(md(t, [("N", "nglob"), ("CPU 1 core", "cpu1"), ("CPU 10 cores", "cpu10"),
                 ("CPU 20 cores", "cpu20"), ("1 P100", "gpu1"), ("2 P100", "gpu2"),
                 ("1 P100 vs 1 core", "s_core"), ("1 P100 vs node", "s_node"),
                 ("2 P100 vs node", "s_node2")],
             {k: (lambda v: "%.2f" % v) for k in ["cpu1", "cpu10", "cpu20", "gpu1", "gpu2"]} |
             {k: (lambda v: "%.1fx" % v) for k in ["s_core", "s_node", "s_node2"]} |
             {"nglob": lambda v: "{:,}".format(int(v))}))
    print("\n(per-rank std)\n")
    print(md(a.sort_values(["config", "nglob"]), [("config", "config"), ("N", "nglob"),
                                                    ("step ms", "mean"), ("std", "std"),
                                                    ("n", "count"), ("force ms", "force_ms")],
             {"mean": lambda v: "%.3f" % v, "std": lambda v: "%.3f" % v,
              "force_ms": lambda v: "%.3f" % v}))

# ---------------------------------------------------------------- scaling
sc = df[df.exp.isin(["scaling", "scaling_multi"])]
if len(sc):
    a = agg(sc, ["exp", "config", "sim_ranks", "nglob"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    S = [("scaling", "cpu_strong_256k", "CPU path, N = 256,000", PINK, "v"),
         ("scaling", "gpu_strong_256k", "GPU path, N = 256,000", GREEN, "D"),
         ("scaling", "gpu_strong_1M", "GPU path, N = 1,048,576", BLUE, "o"),
         ("scaling_multi", "gpu_strong_1M", "GPU path, N = 1,048,576 (multi-node)", ORANGE, "s")]
    rows = []
    for e, c, lab, col, mk in S:
        d = a[(a.exp == e) & (a.config == c)].sort_values("sim_ranks")
        if not len(d):
            continue
        t1 = d["mean"].iloc[0]
        P = d.sim_ranks.values
        s = t1 / d["mean"].values
        eff = s / P
        ds = s * np.sqrt((d["std"].fillna(0).values / d["mean"].values) ** 2 +
                         (d["std"].fillna(0).iloc[0] / t1) ** 2)
        over = (c == "gpu_strong_1M") & (e == "scaling") & (P > 2)
        ax[0].errorbar(P[~over], s[~over], yerr=ds[~over], color=col, marker=mk, label=lab, capsize=2)
        ax[1].plot(P[~over], eff[~over], color=col, marker=mk, label=lab)
        if over.any():
            # 4 ranks sharing 2 P100: still 2 GPUs, so plot at 2 (efficiency per GPU)
            ax[0].plot([2], s[over], color=col, marker=mk, mfc="white", ls="none")
            ax[1].plot([2], s[over] / 2, color=col, marker=mk, mfc="white", ls="none")
            for k_, y_ in [(0, s[over][0]), (1, s[over][0] / 2)]:
                ax[k_].annotate("2 P100, 2 ranks each", (2, y_), xytext=(7, 7),
                                textcoords="offset points", va="center", fontsize=8, color=INK)
        for p_, t_, s_, e_, sd_ in zip(P, d["mean"], s, eff, d["std"]):
            rows.append(dict(path=lab, P=p_, step=t_, std=sd_, S=s_, E=e_))
    pmax = sc.sim_ranks.max()
    ax[0].plot([1, pmax], [1, pmax], color=MUTED, lw=1, ls="--", label="ideal")
    ax[1].axhline(1, color=MUTED, lw=1, ls="--")
    for k in range(2):
        ax[k].set_xscale("log", base=2)
        ax[k].set_xticks([1, 2, 4, 8, 16, 20], ["1", "2", "4", "8", "16", "20"])
        ax[k].set_xlabel("ranks P (cores on the CPU path, GPUs on the GPU path)")
    ax[0].set_yscale("log", base=2); ax[0].set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax[0].set_ylabel("speedup t(1) / t(P)"); ax[1].set_ylabel("parallel efficiency")
    ax[1].set_ylim(0, 1.25)
    ax[0].legend(loc="upper left", fontsize=8)
    title(ax[0], "Strong scaling: speedup"); title(ax[1], "Strong scaling: efficiency")
    note(fig, "Fixed N. " + HW + ". Hollow marker: N = 1,048,576 with 4 ranks sharing the 2 GPUs "
         "(production layout); its efficiency is per GPU.")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "perf_strong_scaling.png")); plt.close(fig)
    print("\n### strong scaling\n")
    print(md(pd.DataFrame(rows), [("run", "path"), ("P", "P"), ("step ms", "step"), ("std", "std"),
                                  ("speedup", "S"), ("efficiency", "E")],
             {"step": lambda v: "%.2f" % v, "std": lambda v: "%.2f" % v,
              "S": lambda v: "%.2f" % v, "E": lambda v: "%.2f" % v}))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    W = [("scaling", "cpu_weak_32k", "CPU path, 32,000 atoms/core", PINK, "v"),
         ("scaling", "gpu_weak_262k", "GPU path, 262,144 atoms/GPU", BLUE, "o"),
         ("scaling_multi", "gpu_weak_262k", "GPU path, 262,144 atoms/GPU (multi-node)", ORANGE, "s")]
    rows = []
    for e, c, lab, col, mk in W:
        d = a[(a.exp == e) & (a.config == c)].sort_values("sim_ranks")
        if not len(d):
            continue
        P = d.sim_ranks.values; t = d["mean"].values; eff = t[0] / t
        over = (c == "gpu_weak_262k") & (e == "scaling") & (P > 2)
        ax[0].errorbar(P[~over], t[~over], yerr=d["std"].fillna(0).values[~over], color=col,
                       marker=mk, label=lab, capsize=2)
        ax[1].plot(P[~over], eff[~over], color=col, marker=mk, label=lab)
        if over.any():
            ax[0].plot(P[over], t[over], color=col, marker=mk, mfc="white", ls="none")
            ax[1].plot(P[over], eff[over], color=col, marker=mk, mfc="white", ls="none")
            ax[1].annotate("4 ranks on 2 P100", (P[over][0], eff[over][0]), xytext=(6, 4),
                           textcoords="offset points", fontsize=8, color=INK)
        for p_, t_, e_, s_, n_ in zip(P, t, eff, d["std"], d.nglob):
            rows.append(dict(path=lab, P=p_, N=n_, step=t_, std=s_, E=e_))
    for k in range(2):
        ax[k].set_xscale("log", base=2)
        ax[k].set_xticks([1, 2, 4, 8, 16, 20], ["1", "2", "4", "8", "16", "20"])
        ax[k].set_xlabel("ranks P")
    ax[0].set_ylabel("time per MD step (ms)"); ax[0].set_ylim(bottom=0)
    ax[1].set_ylabel("weak-scaling efficiency t(1) / t(P)"); ax[1].set_ylim(0, 1.15)
    ax[1].axhline(1, color=MUTED, lw=1, ls="--")
    ax[0].legend(loc="lower right", fontsize=8)
    title(ax[0], "Weak scaling: step time"); title(ax[1], "Weak scaling: efficiency")
    note(fig, "Fixed N per rank. " + HW + ".")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "perf_weak_scaling.png")); plt.close(fig)
    print("\n### weak scaling\n")
    print(md(pd.DataFrame(rows), [("run", "path"), ("P", "P"), ("N", "N"), ("step ms", "step"),
                                  ("std", "std"), ("efficiency", "E")],
             {"step": lambda v: "%.2f" % v, "std": lambda v: "%.2f" % v,
              "E": lambda v: "%.2f" % v, "N": lambda v: "{:,}".format(int(v))}))

# ---------------------------------------------------------------- in situ overhead
ins = df[df.exp.isin(["insitu", "insitu_sep"])].copy()
if len(ins):
    def lab(r):
        if r.ana_ranks == 0:
            return "off"
        s = "K=%d, P_ana=%d" % (r.K, r.ana_ranks)
        return s + (", ANA on a 2nd node" if r.placement == "separate" else "")
    ins["label"] = ins.apply(lab, axis=1)
    g = ins.groupby(["exp", "label"]).agg(
        mean=("step_ms", "mean"), std=("step_ms", "std"), n=("step_ms", "count"),
        stall=("stall_total", "mean"), stall_max=("stall_max", "max"),
        tana=("t_ana_mean", "mean"), tana_max=("t_ana_max", "max"),
        halo=("halo_total", "mean"), pack=("pack_total", "mean"),
        steps=("steps", "first"), K=("K", "first"), P=("ana_ranks", "first")).reset_index()
    order = ["off", "K=50, P_ana=4", "K=100, P_ana=4", "K=200, P_ana=4", "K=100, P_ana=2",
             "K=100, P_ana=1", "K=100, P_ana=4, ANA on a 2nd node"]
    g["o"] = g.label.map({k: i for i, k in enumerate(order)})
    g = g.sort_values(["exp", "o"])
    exps = [e for e in ["insitu", "insitu_sep"] if e in g.exp.values]
    fig, axs = plt.subplots(1, len(exps), figsize=(5.6 * len(exps), 4.3),
                            gridspec_kw={"width_ratios": [2.2, 1][:len(exps)]}, squeeze=False)
    rows = []
    for ax, e in zip(axs[0], exps):
        d = g[g.exp == e]
        off = d[d.label == "off"]["mean"].iloc[0]
        offsd = d[d.label == "off"]["std"].iloc[0]
        x = np.arange(len(d))
        cols = [GREY if l == "off" else (ORANGE if "2nd node" in l else BLUE) for l in d.label]
        ax.bar(x, d["mean"], yerr=d["std"], color=cols, width=0.62, capsize=3,
               error_kw=dict(ecolor=INK, lw=1))
        for i, (m, l) in enumerate(zip(d["mean"], d.label)):
            if l != "off":
                ax.text(i, m * 1.015 + 0.3, "%+.1f%%" % (100 * (m / off - 1)), ha="center",
                        va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [l.replace(", ANA", "\nANA").replace(", P_ana", "\nP_ana")
                          for l in d.label], fontsize=8)
        ax.set_ylabel("SIM time per MD step (ms)")
        ax.set_ylim(0, d["mean"].max() * 1.15)
        ax.grid(axis="x", visible=False)
        title(ax, "Analysis on the GPU node" if e == "insitu" else "Same node vs 2nd node")
        for _, r in d.iterrows():
            rows.append(dict(job=e, run=r.label, n=r.n, step=r["mean"], std=r["std"],
                             over=100 * (r["mean"] / off - 1),
                             over_err=100 * np.hypot(np.nan_to_num(r["std"]) / off,
                                                     r["mean"] * np.nan_to_num(offsd) / off ** 2),
                             tana=r.tana, tana_max=r.tana_max, stall=r.stall,
                             stall_max=r.stall_max,
                             hp=1e3 * (r.halo + r.pack) / r.steps if r.P > 0 else np.nan))
    note(fig, "N = 1,048,576, 4 SIM ranks on 2 P100 (production layout), 3000 steps (melt, quench, "
         "seeded hold), every rank on 2 cores, 2 OpenMP threads per ANA rank. "
         "Bars: mean of 3 runs (off: of all off runs in that job), error bars: sample std.")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "perf_insitu_overhead.png")); plt.close(fig)
    print("\n### in situ overhead\n")
    print(md(pd.DataFrame(rows), [("job", "job"), ("run", "run"), ("n", "n"), ("SIM ms/step", "step"),
                                  ("std", "std"), ("vs off (%)", "over"), ("+-", "over_err"),
                                  ("t_analysis mean (s)", "tana"), ("t_analysis max (s)", "tana_max"),
                                  ("stall total (s)", "stall"), ("stall max (s)", "stall_max"),
                                  ("SIM halo+pack (ms/step)", "hp")],
             {"step": lambda v: "%.2f" % v, "std": lambda v: "%.2f" % v,
              "over": lambda v: "%+.1f" % v, "over_err": lambda v: "%.1f" % v,
              "tana": lambda v: "%.3f" % v, "tana_max": lambda v: "%.3f" % v,
              "stall": lambda v: "%.3f" % v, "stall_max": lambda v: "%.4f" % v,
              "hp": lambda v: "%.3f" % v}))

    # analysis time vs K x step time, per analysis step, from the summaries
    pts = []
    for f in glob.glob(os.path.join(RES, "summary_*_r*.dat")):
        if "smoke" in f:
            continue
        m = re.match(r"summary_(\d+)_(.+)_r(\d+)\.dat", os.path.basename(f))
        job, tag = m[1], m[2]
        s = read_summary(f)
        K = s["step"][1] - s["step"][0]
        P = int(open(f).read().split("nana")[1].split()[0])
        dt = np.diff(s["t_wall"])
        pts.append(dict(tag=tag, K=K, P=P, sep=tag.startswith("sep"),
                        kstep=np.median(dt), tana=np.mean(s["t_analysis"][1:]),
                        tmax=np.max(s["t_analysis"]), stall=np.sum(s["t_stall"])))
    if pts:
        p = pd.DataFrame(pts)
        q = p.groupby(["K", "P", "sep"]).agg(kstep=("kstep", "mean"), tana=("tana", "mean"),
                                             tana_sd=("tana", "std"), tmax=("tmax", "max"),
                                             stall=("stall", "mean")).reset_index()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        mk = {1: "s", 2: "D", 4: "o"}
        cK = {50: VERMILLION, 100: BLUE, 200: GREEN}
        for _, r in q.iterrows():
            ax[0].errorbar(r.kstep, r.tana, yerr=r.tana_sd, marker=mk[r.P],
                           color=ORANGE if r.sep else cK[r.K], ms=8, capsize=2, ls="none",
                           mfc="white" if r.sep else None)
            ax[0].annotate("K=%d, P_ana=%d%s" % (r.K, r.P, ", 2nd node" if r.sep else ""),
                           (r.kstep, r.tana), xytext=(7, -3 if not r.sep else 6),
                           textcoords="offset points", fontsize=8, color=INK)
        lim = max(q.kstep.max(), q.tana.max()) * 1.25
        ax[0].plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1)
        ax[0].text(lim * 0.97, lim * 0.9, "t_analysis = K x step time\n(above: SIM ranks stall)",
                   ha="right", va="top", fontsize=8, color=MUTED)
        ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
        ax[0].set_xlabel("K x SIM step time, wall time between analysis steps (s)")
        ax[0].set_ylabel("t_analysis per analysis step (s)")
        title(ax[0], "Analysis time vs the time it has")
        pr = "runs/production/results/p100/prod_summary.dat"
        if os.path.exists(pr):
            s = read_summary(pr)
            ax[1].plot(s["time"], s["t_analysis"], color=BLUE, lw=1.2, label="t_analysis (4 ANA ranks)")
            ax[1].plot(s["time"], s["t_feat"], color=GREEN, lw=1.0, ls="--",
                       label="of which features + MLP")
            ax[1].axhline(100 * 0.0364, color=VERMILLION, lw=1.5,
                          label="K x step time (100 x 36.4 ms)")
            ax[1].set_xlabel("time (LJ units)"); ax[1].set_ylabel("seconds per analysis step")
            ax[1].set_ylim(0, 4.2); ax[1].legend(loc="upper left")
            title(ax[1], "Production run (P100, K = 100): analysis grows with the crystal")
        note(fig, "Left: N = 1,048,576, 4 SIM ranks on 2 P100, means over the 3000-step runs "
             "(error bars: std over repeats). Right: the 97,400-step production run.")
        fig.tight_layout(); fig.savefig(os.path.join(IMG, "perf_analysis_time.png")); plt.close(fig)
        print("\n### analysis time vs K x step\n")
        print(md(q, [("K", "K"), ("P_ana", "P"), ("2nd node", "sep"), ("K x step (s)", "kstep"),
                     ("t_analysis mean (s)", "tana"), ("t_analysis max (s)", "tmax"),
                     ("stall total (s)", "stall")],
                 {"kstep": lambda v: "%.2f" % v, "tana": lambda v: "%.2f" % v,
                  "tmax": lambda v: "%.2f" % v, "stall": lambda v: "%.3f" % v}))

# ---------------------------------------------------------------- step time drift
wf = [f for f in glob.glob(os.path.join(RES, "drift_*.wall")) if "smoke" not in f]
if wf:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for f in sorted(wf):
        tag = f.rsplit("_", 1)[1].split(".")[0]
        w = np.loadtxt(f, usecols=(1, 2, 3, 4))
        st, wall, fc = w[:, 0], w[:, 1], w[:, 2]
        dstep = np.diff(np.r_[0, st])
        ms = 1e3 * np.diff(np.r_[0, wall]) / dstep
        fms = 1e3 * np.diff(np.r_[0, fc]) / dstep
        col = BLUE if tag == "off" else ORANGE
        ax.plot(st, ms, color=col, marker="o", ms=3,
                label="analysis %s: step" % tag)
        ax.plot(st, fms, color=col, ls=":", lw=1.5, label="analysis %s: force call" % tag)
        print("drift", tag, "mean ms/step melt %.2f ramp %.2f hold %.2f" % (
            ms[st <= 5000].mean(), ms[(st > 5000) & (st <= 7000)].mean(), ms[st > 7000].mean()))
    for x, l in [(5000, "ramp"), (7000, "hold"), (8000, "release")]:
        ax.axvline(x, color=MUTED, lw=0.8, ls=":")
        ax.text(x, ax.get_ylim()[1], " " + l, fontsize=8, color=MUTED, va="top")
    ax.set_xlabel("MD step"); ax.set_ylabel("ms per step (500-step windows)")
    ax.set_ylim(bottom=0); ax.legend(loc="lower right", fontsize=8, ncol=2)
    title(ax, "Step time along the production protocol (N = 1,048,576, 4 SIM on 2 P100)")
    fig.tight_layout(); fig.savefig(os.path.join(IMG, "perf_step_time_drift.png")); plt.close(fig)
