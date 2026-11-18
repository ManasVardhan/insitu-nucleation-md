"""Radial distribution function g(r) from extended XYZ frames written by pmd.

Usage: python gr.py [--rmax 4.0] [--out gr.dat] frame1.xyz [frame2.xyz ...]
Prints the first and second peak and the first minimum. numpy only.
"""
import argparse
import re
import numpy as np


def read_xyz(path):
    with open(path) as fh:
        n = int(fh.readline())
        hdr = fh.readline()
        L = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', hdr)[1].split()])
        L = L[[0, 4, 8]]
        a = np.loadtxt(fh, max_rows=n)
    return a[:, 2:5], L


def gr_frame(x, L, rmax, nbin):
    n = len(x)
    hist = np.zeros(nbin)
    for i0 in range(0, n, 512):
        d = x[i0:i0 + 512, None, :] - x[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d * d).sum(-1)).ravel()
        r = r[(r > 0) & (r < rmax)]
        hist += np.histogram(r, bins=nbin, range=(0, rmax))[0]
    edges = np.linspace(0, rmax, nbin + 1)
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = n / L.prod()
    return hist / (n * rho * shell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--rmax", type=float, default=4.0)
    ap.add_argument("--nbin", type=int, default=400)
    ap.add_argument("--out", default="gr.dat")
    a = ap.parse_args()
    g = 0
    for f in a.frames:
        x, L = read_xyz(f)
        g = g + gr_frame(x, L, a.rmax, a.nbin)
    g /= len(a.frames)
    r = (np.arange(a.nbin) + 0.5) * a.rmax / a.nbin
    np.savetxt(a.out, np.c_[r, g], fmt="%.4f", header="r g(r)")
    i1 = np.argmax(g)
    m = i1 + np.argmin(g[i1:i1 + int(1.0 / (r[1] - r[0]))])
    i2 = m + np.argmax(g[m:m + int(1.2 / (r[1] - r[0]))])
    print(f"{len(a.frames)} frames: 1st peak r={r[i1]:.3f} g={g[i1]:.3f}; "
          f"1st min r={r[m]:.3f} g={g[m]:.3f}; 2nd peak r={r[i2]:.3f} g={g[i2]:.3f}")


if __name__ == "__main__":
    main()
