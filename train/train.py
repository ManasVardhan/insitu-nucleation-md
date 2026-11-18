#!/usr/bin/env python3
"""Train the liquid/crystal MLP on features produced by gen_data.

Inputs (from gen_all.sh): data/train.csv, data/test.csv, data/slab.csv.
The test split is disjoint from training by (T, rho) state point.

Outputs:
  model.pt          state dict + standardization constants (small, committed)
  results.json      all numbers quoted in docs/classifier.md
  ../docs/img/*.png figures

Model: 5 -> 16 -> 16 -> 2, ReLU, inputs [qbar4, qbar6, q4, q6, coord]
standardized with the training-set mean and std.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMG = os.path.join(HERE, "..", "docs", "img")
FEATS = ["qbar4", "qbar6", "q4", "q6", "coord"]
XTAL_Q6 = 0.35        # crude "crystal-like" cut used only for the sanity check
MAX_XTAL_FRAC = 0.01  # liquid snapshot rejected if more atoms than this look crystalline
SEED = 0

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                                 nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def load(name):
    return pd.read_csv(os.path.join(DATA, name + ".csv"))


def crystallization_check(df):
    """Per liquid snapshot, fraction of atoms with qbar6 > XTAL_Q6."""
    liq = df[df["set"] == "liquid"]
    g = liq.groupby(["T", "rho", "snap"])["qbar6"].apply(lambda s: float((s > XTAL_Q6).mean()))
    bad = g[g > MAX_XTAL_FRAC]
    keep = df
    for (T, rho, snap) in bad.index:
        keep = keep[~((keep["set"] == "liquid") & (keep["T"] == T) & (keep["rho"] == rho)
                      & (keep["snap"] == snap))]
    worst = g.groupby(level=[0, 1]).max()
    return keep, worst, list(bad.index)


def best_threshold(q, y):
    """Threshold t maximizing accuracy of (q > t) == y; returns (t, acc)."""
    order = np.argsort(q)
    qs, ys = q[order], y[order]
    n = len(y)
    # predict 1 for indices >= k: correct = (#0 in [0,k)) + (#1 in [k,n))
    zeros_before = np.concatenate([[0], np.cumsum(ys == 0)])
    ones_after = np.concatenate([np.cumsum((ys == 1)[::-1])[::-1], [0]])
    correct = zeros_before + ones_after
    k = int(np.argmax(correct))
    if k == 0:
        t = qs[0] - 1e-9
    elif k == n:
        t = qs[-1]
    else:
        t = 0.5 * (qs[k - 1] + qs[k])
    return float(t), float(correct[k] / n)


def per_state(df, pred):
    rows = []
    d = df.assign(ok=(pred == df["label"].values))
    for (T, rho), g in d.groupby(["T", "rho"]):
        r = {"T": T, "rho": rho}
        for lab, name in [(1, "crystal"), (0, "liquid")]:
            r[name] = float(g[g["label"] == lab]["ok"].mean())
        r["all"] = float(g["ok"].mean())
        rows.append(r)
    return rows


def fitted_interface_distance(slab_df, rho=0.95):
    """Signed distance of each slab atom to the current interface planes.

    For each snapshot, bin z into slices one (001) layer thick (a/2), average
    q6 (the non-averaged order parameter) per slice, and find where the
    profile crosses the midpoint between the bulk crystal and bulk liquid
    plateaus. The two crossings nearest the initial planes Lz/4 and 3Lz/4 are
    the current interfaces. Returns (dist, shifts) where shifts lists how far
    each interface moved toward the liquid (positive = crystal grew).
    """
    a = (4.0 / rho) ** (1.0 / 3.0)
    Lz = 16 * a  # gen_all.sh slab: 8 x 8 x 16 cells
    w = a / 2
    nsl = int(round(Lz / w))
    dist = np.zeros(len(slab_df))
    shifts = []
    for (T, snap), g in slab_df.groupby(["T", "snap"]):
        z = np.mod(g["z"].values, Lz)
        q6 = g["q6"].values
        dd = g["dist"].values
        lab = g["label"].values
        qx = q6[(lab == 1) & (dd > 3)].mean()
        ql = q6[(lab == 0) & (dd < -3)].mean()
        mid = 0.5 * (qx + ql)
        k = np.minimum((z / w).astype(int), nsl - 1)
        prof = np.array([q6[k == i].mean() for i in range(nsl)])
        zc = (np.arange(nsl) + 0.5) * w
        crossings = []
        for i in range(nsl):
            j = (i + 1) % nsl
            p0, p1 = prof[i] - mid, prof[j] - mid
            if p0 * p1 < 0:
                zz = zc[i] + w * p0 / (p0 - p1)
                crossings.append(np.mod(zz, Lz))
        crossings = np.array(crossings)

        def nearest(z0):
            dz = crossings - z0
            dz -= Lz * np.round(dz / Lz)
            return z0 + dz[np.argmin(np.abs(dz))]

        z1, z2 = nearest(0.25 * Lz), nearest(0.75 * Lz)
        shifts += [0.25 * Lz - z1, z2 - 0.75 * Lz]
        d1 = z - z1
        d1 -= Lz * np.round(d1 / Lz)
        d2 = z2 - z
        d2 -= Lz * np.round(d2 / Lz)
        inside = np.mod(z - z1, Lz) < (z2 - z1)
        m = np.minimum(np.abs(d1), np.abs(d2))
        dist[g.index.values] = np.where(inside, m, -m)
    return dist, np.array(shifts)


def fit_mlp(X, y, seed, epochs=30, verbose=False):
    """Adam + cosine schedule, batch 1024, cross-entropy. Returns (model, mean, std)."""
    torch.manual_seed(seed)
    mean, std = X.mean(axis=0), X.std(axis=0)
    xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(yt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and epoch % 5 == 4:
            print("epoch %2d  loss %.5f" % (epoch + 1, tot / n))
    model.eval()
    return model, mean, std


def interface_ablation(train_df, test_df, slab_df, d_fit, model, mean, std, t6):
    """Does interface-aware training help? Train on bulk + slabs at T = 0.50,
    0.60 (labels from the fitted current interface, slab rows repeated 3x),
    evaluate on the slabs at T = 0.55, 0.65 and on the bulk test set. The
    exported model is NOT this one; this only measures the effect."""
    lab_fit = (d_fit > 0).astype(np.int64)
    tr_m = slab_df["T"].isin([0.50, 0.60]).values
    te_m = ~tr_m
    Xs, ys = slab_df[FEATS].values, lab_fit
    Xb, yb = train_df[FEATS].values, train_df["label"].values
    Xt, yt = test_df[FEATS].values, test_df["label"].values
    near = te_m & (np.abs(d_fit) < 1.0)

    def ev(pred_fn):
        ps = pred_fn(Xs)
        return {"bulk_test": float((pred_fn(Xt) == yt).mean()),
                "slab_test": float((ps[te_m] == ys[te_m]).mean()),
                "slab_test_near": float((ps[near] == ys[near]).mean())}

    def mlp_pred(m, mu, sd):
        def f(X):
            with torch.no_grad():
                return m(torch.tensor((X - mu) / sd, dtype=torch.float32)).argmax(1).numpy()
        return f

    out = {"bulk_only_mlp": ev(mlp_pred(model, mean, std)),
           "bulk_only_qbar6": ev(lambda X: (X[:, 1] > t6).astype(np.int64))}
    Xa = np.concatenate([Xb] + [Xs[tr_m]] * 3)
    ya = np.concatenate([yb] + [ys[tr_m]] * 3)
    m2, mu2, sd2 = fit_mlp(Xa, ya, SEED)
    out["with_slab_mlp"] = ev(mlp_pred(m2, mu2, sd2))
    t_s, _ = best_threshold(Xs[tr_m][:, 1], ys[tr_m])
    out["slab_fit_qbar6"] = ev(lambda X: (X[:, 1] > t_s).astype(np.int64))
    out["slab_fit_qbar6"]["t"] = t_s
    out["n_slab_test_near"] = int(near.sum())
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(IMG, exist_ok=True)

    train_df, test_df, slab_df = load("train"), load("test"), load("slab")
    train_df, worst_tr, bad_tr = crystallization_check(train_df)
    test_df, worst_te, bad_te = crystallization_check(test_df)
    print("liquid sanity check (max frac qbar6>%.2f over snapshots):" % XTAL_Q6)
    for idx, v in pd.concat([worst_tr, worst_te]).items():
        print("   T=%.3f rho=%.3f  %.4f" % (idx[0], idx[1], v))
    print("rejected liquid snapshots:", bad_tr + bad_te)

    Xtr = train_df[FEATS].values.astype(np.float64)
    ytr = train_df["label"].values.astype(np.int64)
    Xte = test_df[FEATS].values.astype(np.float64)
    yte = test_df["label"].values.astype(np.int64)
    Xsl = slab_df[FEATS].values.astype(np.float64)
    ysl = slab_df["label"].values.astype(np.int64)

    model, mean, std = fit_mlp(Xtr, ytr, SEED, verbose=True)
    n = len(ytr)

    def norm(X):
        return torch.tensor((X - mean) / std, dtype=torch.float32)

    model.eval()

    def predict(X):
        with torch.no_grad():
            logits = model(norm(X)).double()
            p = torch.softmax(logits, dim=1)[:, 1].numpy()
        return (p > 0.5).astype(np.int64), p

    ptr, _ = predict(Xtr)
    pte, prob_te = predict(Xte)
    psl, prob_sl = predict(Xsl)
    acc = lambda p, y: float((p == y).mean())

    # baseline: single threshold on qbar6, fit on training data
    t6, base_tr = best_threshold(Xtr[:, 1], ytr)
    bte = (Xte[:, 1] > t6).astype(np.int64)
    bsl = (Xsl[:, 1] > t6).astype(np.int64)
    t6_oracle, base_te_oracle = best_threshold(Xte[:, 1], yte)

    res = {
        "n_train": int(n), "n_test": int(len(yte)), "n_slab": int(len(ysl)),
        "mlp": {"train": acc(ptr, ytr), "test": acc(pte, yte), "slab": acc(psl, ysl)},
        "qbar6_threshold": {"t": t6, "train": base_tr, "test": acc(bte, yte),
                            "slab": acc(bsl, ysl), "t_oracle_test": t6_oracle,
                            "test_oracle": base_te_oracle},
        "test_per_state_mlp": per_state(test_df, pte),
        "test_per_state_qbar6": per_state(test_df, bte),
        "liquid_rejected": [list(map(float, b)) for b in bad_tr + bad_te],
        "liquid_max_xtal_frac": {"%.3f,%.3f" % k: float(v)
                                 for k, v in pd.concat([worst_tr, worst_te]).items()},
    }
    # test errors split by class
    for name, p in [("mlp", pte), ("qbar6_threshold", bte)]:
        res[name]["test_crystal"] = acc(p[yte == 1], yte[yte == 1])
        res[name]["test_liquid"] = acc(p[yte == 0], yte[yte == 0])

    # slab: accuracy vs signed distance to the initial interface plane
    d = slab_df["dist"].values
    edges = np.arange(-6.0, 6.01, 0.5)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() == 0:
            continue
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "mlp": acc(psl[m], ysl[m]), "qbar6": acc(bsl[m], ysl[m])})
    res["slab_vs_dist"] = rows

    # second ground truth: side of the *current* interface, located from the
    # layer-averaged q6 profile (independent of any per-atom classifier)
    d_fit, shifts = fitted_interface_distance(slab_df)
    ysl_fit = (d_fit > 0).astype(np.int64)
    rows_fit = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_fit >= lo) & (d_fit < hi)
        if m.sum() == 0:
            continue
        rows_fit.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                         "mlp": acc(psl[m], ysl_fit[m]), "qbar6": acc(bsl[m], ysl_fit[m])})
    res["slab_vs_dist_fitted"] = rows_fit
    res["slab_fitted"] = {"mlp": acc(psl, ysl_fit), "qbar6": acc(bsl, ysl_fit),
                          "interface_shift_mean": float(np.mean(shifts)),
                          "interface_shift_max": float(np.max(shifts))}
    nf = np.abs(d_fit) < 1.0
    res["slab_fitted_near"] = {"mlp": acc(psl[nf], ysl_fit[nf]), "qbar6": acc(bsl[nf], ysl_fit[nf]),
                               "n": int(nf.sum())}
    far = np.abs(d) >= 2.0
    near = np.abs(d) < 1.0
    res["slab_far"] = {"mlp": acc(psl[far], ysl[far]), "qbar6": acc(bsl[far], ysl[far]),
                       "n": int(far.sum())}
    res["slab_near"] = {"mlp": acc(psl[near], ysl[near]), "qbar6": acc(bsl[near], ysl[near]),
                        "n": int(near.sum())}
    # per-slab-T accuracy
    res["slab_per_T"] = []
    for T, g in slab_df.assign(pm=psl, pb=bsl).groupby("T"):
        res["slab_per_T"].append({"T": T, "mlp": acc(g["pm"].values, g["label"].values),
                                  "qbar6": acc(g["pb"].values, g["label"].values),
                                  "pred_xtal_frac": float(g["pm"].mean())})

    res["interface_ablation"] = interface_ablation(train_df, test_df, slab_df, d_fit, model,
                                                   mean, std, t6)
    print(json.dumps({k: res[k] for k in ["mlp", "qbar6_threshold", "slab_far", "slab_near",
                                          "slab_fitted", "slab_fitted_near",
                                          "interface_ablation"]},
                     indent=1))
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    torch.save({"state_dict": model.state_dict(), "mean": mean.tolist(), "std": std.tolist(),
                "features": FEATS}, os.path.join(HERE, "model.pt"))

    make_figures(train_df, test_df, slab_df, psl, bsl, prob_sl, t6, rows, rows_fit)


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(train_df, test_df, slab_df, psl, bsl, prob_sl, t6, rows, rows_fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK})

    # 1. feature space: qbar4 vs qbar6, train and held-out test
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    rng = np.random.default_rng(1)
    for ax, df, title in [(axes[0], train_df, "Training state points"),
                          (axes[1], test_df, "Held-out state points")]:
        sub = df.iloc[rng.choice(len(df), size=min(20000, len(df)), replace=False)]
        for lab, col, name in [(0, ORANGE, "liquid"), (1, BLUE, "crystal")]:
            s = sub[sub["label"] == lab]
            ax.scatter(s["qbar4"], s["qbar6"], s=2, c=col, alpha=0.35, linewidths=0,
                       label=name, rasterized=True)
        ax.axhline(t6, color=MUTED, linewidth=1, linestyle="--")
        ax.text(0.005, t6 + 0.008, "qbar6 threshold %.3f" % t6, color=MUTED, fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("qbar4")
        style(ax)
    axes[0].set_ylabel("qbar6")
    leg = axes[0].legend(loc="lower right", markerscale=5, frameon=False)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "feature_space.png"), dpi=150)
    plt.close(fig)

    # 2. accuracy vs distance from the interface, two ground truths
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rr, title, xl in [
            (axes[0], rows, "Truth = initial region", "distance from initial plane (sigma)"),
            (axes[1], rows_fit, "Truth = side of current interface (layer-averaged q6)",
             "distance from current interface (sigma)")]:
        mid = [(r["lo"] + r["hi"]) / 2 for r in rr]
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.plot(mid, [r["mlp"] for r in rr], color=BLUE, linewidth=2, marker="o",
                markersize=5, label="MLP")
        ax.plot(mid, [r["qbar6"] for r in rr], color=ORANGE, linewidth=2, marker="s",
                markersize=5, label="qbar6 threshold")
        ax.set_xlabel(xl + ", crystal side > 0")
        ax.set_title(title, fontsize=10, loc="left")
        ax.text(-5.8, 0.06, "liquid side", color=MUTED, fontsize=9)
        ax.text(3.4, 0.06, "crystal side", color=MUTED, fontsize=9)
        ax.set_ylim(0, 1.03)
        style(ax)
    axes[0].set_ylabel("per-atom accuracy (0.5 sigma bins)")
    axes[0].legend(frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "interface_accuracy.png"), dpi=150)
    plt.close(fig)

    # 3. crystal-probability profile along z for one slab (T=0.55, last snapshot)
    T0 = 0.55
    m = (slab_df["T"].values == T0) & (slab_df["snap"].values == slab_df["snap"].max())
    s = slab_df[m]
    z = s["z"].values
    a = (4.0 / 0.95) ** (1.0 / 3.0)
    bins = np.linspace(0, 16 * a, 33)  # one (001) layer (a/2) per bin
    c = 0.5 * (bins[1:] + bins[:-1])
    idx = np.minimum(np.digitize(np.mod(z, 16 * a), bins) - 1, len(c) - 1)
    def prof(v):
        return np.array([v[idx == k].mean() if np.any(idx == k) else np.nan
                         for k in range(len(c))])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(c, prof(s["label"].values.astype(float)), color=MUTED, linewidth=1.5,
            drawstyle="steps-mid", label="initial-region label")
    ax.plot(c, prof(psl[m].astype(float)), color=BLUE, linewidth=2, label="MLP crystal fraction")
    ax.plot(c, prof(bsl[m].astype(float)), color=ORANGE, linewidth=1.5, linestyle="--",
            label="qbar6-threshold crystal fraction")
    ax.set_xlabel("z (sigma)")
    ax.set_ylabel("fraction labeled crystal")
    ax.set_title("Slab, T* = %.2f, rho* = 0.95, 150 steps after release" % T0, fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.55))
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "slab_profile.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
