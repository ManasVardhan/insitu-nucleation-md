"""Solid-like atom fraction and largest solid cluster from pmd XYZ frames.

Quick post hoc check for the Phase 1 smoke test (the in situ classifier is
separate). Uses the ten Wolde-Frenkel criterion: neighbors within rq,
normalized q6 vectors, a bond is solid-like if d6(i,j) > 0.7, an atom is
solid-like with >= 7 such bonds. Clusters: solid atoms within rq.

Usage: python solid_frac.py [--rq 1.5] frame1.xyz [frame2.xyz ...]
Needs numpy and scipy.
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gr import read_xyz


def analyze(x, L, rq):
    n = len(x)
    x = np.mod(x, L)
    tree = cKDTree(x, boxsize=L)
    pairs = tree.query_pairs(rq, output_type="ndarray")
    i, j = np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]]
    d = x[j] - x[i]
    d -= L * np.round(d / L)
    r = np.linalg.norm(d, axis=1)
    theta = np.arccos(np.clip(d[:, 2] / r, -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])
    q = np.zeros((n, 13), complex)
    for k, m in enumerate(range(-6, 7)):
        np.add.at(q[:, k], i, sph_harm(m, 6, phi, theta))
    q /= np.maximum(np.linalg.norm(q, axis=1), 1e-12)[:, None]
    d6 = np.real((q[i] * np.conj(q[j])).sum(1))
    nsolid_bonds = np.bincount(i, weights=(d6 > 0.7), minlength=n)
    solid = nsolid_bonds >= 7
    sb = solid[i] & solid[j]
    g = coo_matrix((np.ones(sb.sum()), (i[sb], j[sb])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab[solid]) if solid.any() else np.array([0])
    return solid.mean(), sizes.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--rq", type=float, default=1.5)
    a = ap.parse_args()
    for f in a.frames:
        x, L = read_xyz(f)
        frac, nmax = analyze(x, L, a.rq)
        print(f"{f}: solid fraction {frac:.4f}  largest cluster {nmax}")


if __name__ == "__main__":
    main()
