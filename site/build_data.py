#!/usr/bin/env python3
"""Build the data behind site/index.html.

    python3 site/build_data.py

Reads the production summary, the perf CSVs and the classifier results from the
repository, writes compact JSON to site/data/, and inlines the same JSON into
site/index.html between <script type="application/json" id="..."> tags, so the
page works without fetch (file://, GitHub Pages, or a sandbox).
"""
import csv
import json
import os
import re
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PERF = os.path.join(ROOT, "runs", "perf", "results")
OUT = os.path.join(HERE, "data")
HTML = os.path.join(HERE, "index.html")


def read_summary(path):
    """prod_summary.dat: '#' header with column names, '|' separates two column groups."""
    with open(path) as f:
        lines = f.read().splitlines()
    names = lines[0].lstrip("#").replace("|", " ").split()
    meta = lines[1].lstrip("#").split()
    meta = {meta[i]: meta[i + 1] for i in range(0, len(meta) - 1, 2)}
    rows = []
    for ln in lines[2:]:
        if not ln.strip() or ln.startswith("#"):
            continue
        vals = ln.replace("|", " ").split()
        rows.append({k: float(v) for k, v in zip(names, vals)})
    return rows, meta


def production():
    rows, meta = read_summary(os.path.join(ROOT, "runs", "production", "results", "p100", "prod_summary.dat"))
    N = int(meta["nglob"])
    movie_every = 1000                       # movie snapshot every 10 analysis steps (prod_p100.in)
    snap = 13 * N + 40                       # bytes per movie snapshot (src/insitu.h, analysis/plot_io.py)
    keep = rows[::2]
    if keep[-1] is not rows[-1]:
        keep.append(rows[-1])
    GB = 1e9
    out = {
        "N": N, "K": int(meta["K"]), "n_rows_total": len(rows),
        "t": [], "step": [], "nmax_mlp": [], "nmax_thr": [], "ncr_mlp": [], "ncr_thr": [],
        "n2_mlp": [], "gb_insitu": [], "gb_labels": [], "gb_full": [],
    }
    for r in keep:
        step = int(r["step"])
        out["t"].append(round(r["time"], 1))
        out["step"].append(step)
        out["nmax_mlp"].append(int(r["nmax_mlp"]))
        out["nmax_thr"].append(int(r["nmax_thr"]))
        out["ncr_mlp"].append(int(r["n_crystal_mlp"]))
        out["ncr_thr"].append(int(r["n_crystal_thr"]))
        out["n2_mlp"].append(int(r["n2_mlp"]))
        out["gb_insitu"].append(round(r["bytes_insitu"] / GB, 5))
        out["gb_labels"].append(round((r["bytes_insitu"] - (step // movie_every) * snap) / GB, 5))
        out["gb_full"].append(round(r["bytes_fulldump"] / GB, 5))
    return out


def read_csv(name):
    with open(os.path.join(PERF, name)) as f:
        return list(csv.DictReader(f))


def speedup():
    rows = read_csv("speedup_14364484_cellorder.csv")
    acc = defaultdict(list)
    for r in rows:
        acc[(r["config"], int(r["nglob"]))].append((float(r["step_ms"]), float(r["force_ms"])))
    Ns = sorted({n for _, n in acc})
    out = {"N": Ns, "job": 14364484}
    for c in ["cpu1", "cpu10", "cpu20", "gpu1", "gpu2"]:
        out[c] = [round(mean(v[0] for v in acc[(c, n)]), 2) for n in Ns]
        out[c + "_force"] = [round(mean(v[1] for v in acc[(c, n)]), 2) for n in Ns]
    return out


def mean_by(rows, key="config", field="step_ms"):
    acc = defaultdict(list)
    for r in rows:
        acc[r[key]].append(float(r[field]))
    return {k: round(mean(v), 2) for k, v in acc.items()}


def insitu():
    cell = read_csv("insitu_14364486_cellorder.csv")
    sep = read_csv("insitu_sep_14362880.csv")
    pana = read_csv("insitu_14363659.csv")
    sort = read_csv("sortfix_14363770.csv")
    return {
        "cellorder_k100": mean_by(cell),          # off / k100, cell-order kernel
        "separate_node": mean_by(sep),            # off / k100_same / k100_sep, original kernel
        "p_ana": mean_by(pana),                   # off / p1 / p2, original kernel
        "p_ana_t_analysis": mean_by([r for r in pana if r["config"] != "off"], field="t_ana_mean"),
        "p_ana_stall_total": mean_by([r for r in pana if r["config"] != "off"], field="stall_total"),
        "sortfix": mean_by(sort),
    }


def scaling():
    rows = read_csv("scaling_14364485_cellorder.csv")
    acc = defaultdict(list)
    for r in rows:
        acc[(r["config"], int(r["sim_ranks"]))].append(float(r["step_ms"]))
    return {"%s_P%d" % k: round(mean(v), 2) for k, v in sorted(acc.items())}


def classifier():
    with open(os.path.join(ROOT, "train", "results.json")) as f:
        r = json.load(f)
    return {
        "n_train": r["n_train"], "n_test": r["n_test"], "n_slab": r["n_slab"],
        "mlp": {"train": r["mlp"]["train"], "test": r["mlp"]["test"],
                "slab": r["slab_fitted"]["mlp"], "near": r["slab_fitted_near"]["mlp"]},
        "thr": {"train": r["qbar6_threshold"]["train"], "test": r["qbar6_threshold"]["test"],
                "slab": r["slab_fitted"]["qbar6"], "near": r["slab_fitted_near"]["qbar6"],
                "t": r["qbar6_threshold"]["t"]},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    blobs = {
        "data-prod": production(),
        "data-speedup": speedup(),
        "data-insitu": insitu(),
        "data-scaling": scaling(),
        "data-classifier": classifier(),
    }
    total = 0
    for k, v in blobs.items():
        s = json.dumps(v, separators=(",", ":"))
        total += len(s)
        with open(os.path.join(OUT, k.replace("data-", "") + ".json"), "w") as f:
            f.write(s + "\n")
        print("%-16s %7d bytes" % (k, len(s)))
    print("embedded JSON total %d bytes" % total)

    if os.path.exists(HTML):
        html = open(HTML).read()
        for k, v in blobs.items():
            s = json.dumps(v, separators=(",", ":"))
            pat = re.compile(r'(<script type="application/json" id="%s">)(.*?)(</script>)' % re.escape(k), re.S)
            if not pat.search(html):
                print("warning: no <script id=%s> block in index.html" % k)
                continue
            html = pat.sub(lambda m: m.group(1) + s + m.group(3), html)
        with open(HTML, "w") as f:
            f.write(html)
        print("inlined into", os.path.relpath(HTML, ROOT), "(%d bytes)" % len(html))


if __name__ == "__main__":
    main()
