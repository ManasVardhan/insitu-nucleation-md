"""Readers for the in situ output files (formats in src/insitu.h).

    from insitu_read import read_summary, iter_labels, read_movie
    s = read_summary("prod_summary.dat")      # dict of numpy columns
    for step, ids, bits in iter_labels("prod_labels.bin"): ...
    f = read_movie("prod_movie_00010000.bin")  # dict: step, time, box, pos, bits

bits: 1 = MLP crystal, 2 = qbar6 > thr, 4 = in largest MLP cluster,
8 = in largest threshold cluster.
"""
import numpy as np

SUMMARY_COLS = ["step", "time", "n_crystal_mlp", "n_crystal_thr", "nmax_mlp",
                "nmax_thr", "t_analysis", "t_stall", "bytes_insitu", "n2_mlp",
                "n2_thr", "nbig_mlp", "nbig_thr", "t_feat", "t_clus_io",
                "t_recvwait", "t_simhalo", "t_simpack", "bytes_fulldump", "t_wall"]


def read_summary(path):
    rows = []
    with open(path) as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            rows.append([float(x) for x in line.replace("|", " ").split()])
    a = np.array(rows)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # files written before t_wall was added have one column less
    return {c: a[:, i] for i, c in enumerate(SUMMARY_COLS[:a.shape[1]])}


def iter_labels(path):
    with open(path, "rb") as fp:
        while True:
            hdr = np.fromfile(fp, dtype="<i4", count=2)
            if hdr.size < 2:
                return
            step, n = int(hdr[0]), int(hdr[1])
            ids = np.fromfile(fp, dtype="<i4", count=n)
            bits = np.fromfile(fp, dtype="u1", count=n)
            yield step, ids, bits


def read_movie(path):
    with open(path, "rb") as fp:
        nglob, step = np.fromfile(fp, dtype="<i4", count=2)
        time = np.fromfile(fp, dtype="<f8", count=1)[0]
        box = np.fromfile(fp, dtype="<f8", count=3)
        pos = np.fromfile(fp, dtype="<f4", count=3 * nglob).reshape(nglob, 3)
        bits = np.fromfile(fp, dtype="u1", count=nglob)
    return {"step": int(step), "time": time, "box": box, "pos": pos, "bits": bits}
