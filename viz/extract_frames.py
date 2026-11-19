"""Reduce in situ movie frames to what the renders need.

    python viz/extract_frames.py <dir with prod_movie_*.bin> <out dir> [slab half-width]

For every frame, writes <out>/f_<step>.npz with
  cpos, cbits : atoms labeled crystalline by either label set (bits 1 or 2)
  spos, sbits : all atoms in a slab |z - L/2| < w through the seed center
  step, time, box
The full frames (13 bytes/atom, 13.6 MB each at N = 1,048,576) can then be
deleted; the reduced set is about a quarter of the size.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from insitu_read import read_movie  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
w = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
os.makedirs(out, exist_ok=True)
for path in sorted(glob.glob(os.path.join(src, "*_movie_*.bin"))):
    f = read_movie(path)
    b, p, box = f["bits"], f["pos"], f["box"]
    c = (b & 3) > 0
    s = np.abs(p[:, 2] - 0.5 * box[2]) < w
    np.savez(os.path.join(out, "f_%08d.npz" % f["step"]), cpos=p[c], cbits=b[c],
             spos=p[s], sbits=b[s], step=f["step"], time=f["time"], box=box)
    print(f["step"], int(c.sum()), int(s.sum()))
