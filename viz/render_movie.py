"""Render the in situ movie frames with OVITO (headless).

    python viz/render_movie.py <frames_small dir> <out dir> [--renderer opengl|tachyon]
           [--size 960x720] [--only A|B] [--steps 1000,50000,...] [--hold 3]

Input: the reduced frames from viz/extract_frames.py. Two views of the same
frames, rendered to <out>/A/*.png and <out>/B/*.png:

  A  crystalline atoms only (MLP label), colored by distance from the seed
     center, inside the box outline; the camera turns slowly around z.
     Atoms that the MLP calls crystalline but that are not in the largest
     cluster are drawn grey.
  B  a slab |z - L/2| < 1.5 through the seed center, all atoms, looking
     down z, colored by the two label sets: both crystalline, MLP only,
     q6-bar threshold only, liquid.

Each frame is rendered --hold times (A: with the camera advancing), so a
97-frame run at 24 fps lasts 97*hold/24 s.
"""
import argparse
import glob
import math
import os

import numpy as np
from ovito.data import DataCollection, Particles, SimulationCell
from ovito.pipeline import Pipeline, StaticSource
from ovito.vis import (OpenGLRenderer, TachyonRenderer, TextLabelOverlay,
                       Viewport)
from ovito.qt_compat import QtCore

TOPLEFT = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop

# colorblind-safe (Okabe-Ito) categories for B
C_BOTH, C_MLP, C_THR, C_LIQ = (0.0, 0.447, 0.698), (0.0, 0.62, 0.451), \
    (0.902, 0.624, 0.0), (0.80, 0.80, 0.80)
C_GREY = (0.62, 0.62, 0.62)

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--renderer", default="opengl")
ap.add_argument("--size", default="1280x720")
ap.add_argument("--only", default="AB")
ap.add_argument("--steps", default="")
ap.add_argument("--hold", type=int, default=3)
ap.add_argument("--slab", type=float, default=1.5)
ap.add_argument("--deg-per-frame", type=float, default=0.6)
args = ap.parse_args()
W, H = (int(x) for x in args.size.split("x"))
files = sorted(glob.glob(os.path.join(args.src, "f_*.npz")))
if args.steps:
    keep = {int(s) for s in args.steps.split(",")}
    files = [f for f in files if int(os.path.basename(f)[2:10]) in keep]


def renderer():
    if args.renderer == "tachyon":
        return TachyonRenderer(shadows=False, ambient_occlusion=True,
                               ambient_occlusion_brightness=0.8, antialiasing_samples=6)
    return OpenGLRenderer(antialiasing_level=3)


def make_dc(box, pos, colors, radius):
    dc = DataCollection()
    cell = SimulationCell(pbc=(True, True, True))
    cell[...] = [[box[0], 0, 0, 0], [0, box[1], 0, 0], [0, 0, box[2], 0]]
    cell.vis.line_width = 0.25
    cell.vis.rendering_color = (0.45, 0.45, 0.45)
    dc.objects.append(cell)
    p = Particles(count=len(pos))
    p.create_property("Position", data=pos.astype(np.float64))
    p.create_property("Color", data=np.asarray(colors, dtype=np.float64))
    p.create_property("Radius", data=np.broadcast_to(np.asarray(radius, dtype=np.float64), (len(pos),)).copy())
    dc.objects.append(p)
    return dc


def label(vp, text, pos, size=0.05, color=(0.12, 0.12, 0.12)):
    o = TextLabelOverlay(text=text, alignment=TOPLEFT, offset_x=pos[0], offset_y=pos[1],
                         font_size=size, text_color=color)
    vp.overlays.append(o)
    return o


def cmap(t):
    """viridis, perceptually uniform and colorblind safe (t in [0, 1])."""
    import matplotlib
    return matplotlib.colormaps["viridis"](np.clip(t, 0, 1))[:, :3]


summary = {}
sfile = os.path.join(args.src, "summary.txt")
if os.path.exists(sfile):
    for line in open(sfile):
        s, nm, nt = line.split()
        summary[int(s)] = (int(nm), int(nt))

t_release = 40.0  # seed released at step 8000 (t = 40)
t_hold = 35.0     # hold at T_q = 0.70 starts at step 7000 (t = 35)


def phase(t):
    if t < 25.0:
        return "melt, T* = 2.0"
    if t < t_hold:
        return "quench to T* = 0.70"
    if t < t_release:
        return "hold, seed frozen"
    return "hold, seed released"


pipe = Pipeline(source=StaticSource(data=DataCollection()))
pipe.add_to_scene()
for view in args.only:
    os.makedirs(os.path.join(args.out, view), exist_ok=True)
k = 0
for f in files:
    d = np.load(f)
    box, step, t = d["box"], int(d["step"]), float(d["time"])
    nm, nt = summary.get(step, (None, None))
    for view in args.only:
        if view == "A":
            m = (d["cbits"] & 1) > 0
            pos = d["cpos"][m]
            r = np.linalg.norm(pos - 0.5 * box, axis=1)
            col = cmap(r / 55.0)
            other = (d["cbits"][m] & 4) == 0
            col[other] = C_GREY
            pipe.source.data = make_dc(box, pos, col, np.where(other, 0.4, 0.5))
        else:
            m = np.abs(d["spos"][:, 2] - 0.5 * box[2]) < args.slab
            pos, b = d["spos"][m], d["sbits"][m]
            col = np.tile(C_LIQ, (len(pos), 1))
            mlp, thr = (b & 1) > 0, (b & 2) > 0
            col[mlp & thr] = C_BOTH
            col[mlp & ~thr] = C_MLP
            col[~mlp & thr] = C_THR
            rad = np.where(mlp | thr, 0.5, 0.42)
            pipe.source.data = make_dc(box, pos, col, rad)
        for h in range(args.hold if not args.steps else 1):
            vp = Viewport()
            if view == "A":
                ang = math.radians(35 + args.deg_per_frame * k)
                c = 0.5 * box
                dist = 2.6 * box[0]
                cam = c + dist * np.array([math.cos(ang), math.sin(ang), 0.6])
                fwd = (c - cam) / np.linalg.norm(c - cam)
                right = np.cross(fwd, [0.0, 0.0, 1.0])
                right /= np.linalg.norm(right)
                # aim left of the box center so the box sits right of the text column
                tgt = c - 0.22 * box[0] * right
                vp.type = Viewport.Type.Perspective
                vp.camera_pos = tuple(cam - 0.22 * box[0] * right)
                vp.camera_dir = tuple(tgt - (cam - 0.22 * box[0] * right))
                vp.fov = math.radians(30)
            else:
                L = box[0]
                vp.type = Viewport.Type.Ortho
                vp.camera_pos = (0.5 * L - 0.36 * L, 0.5 * L, 2.0 * L)
                vp.camera_dir = (0.0, 0.0, -1.0)
                vp.fov = 0.56 * L
            ttl = "Crystalline atoms (MLP label)" if view == "A" else \
                "Slab through the seed, |z - L/2| < %.1f" % args.slab
            label(vp, ttl, (0.015, -0.03), 0.042)
            label(vp, "t = %.0f   (step %d)" % (t, step), (0.015, -0.10), 0.036, (0.3, 0.3, 0.3))
            label(vp, phase(t), (0.015, -0.15), 0.036, (0.3, 0.3, 0.3))
            if nm is not None:
                label(vp, "n_max, MLP: {:,}".format(nm), (0.015, -0.20), 0.036, (0.3, 0.3, 0.3))
                label(vp, "n_max, q6-bar: {:,}".format(nt), (0.015, -0.25), 0.036, (0.3, 0.3, 0.3))
            if view == "B":
                for i, (txt, cc) in enumerate([("both labels crystalline", C_BOTH),
                                               ("MLP only", C_MLP),
                                               ("q6-bar > 0.308 only", C_THR),
                                               ("liquid", (0.6, 0.6, 0.6))]):
                    label(vp, "\u25cf " + txt, (0.015, -0.68 - 0.05 * i), 0.034, cc)
            else:
                for i, txt in enumerate(["color: distance from the seed",
                                         "center, 0 (purple) to 55 (yellow)",
                                         "grey: not in the largest cluster",
                                         "N = 1,048,576, T* = 0.70"]):
                    label(vp, txt, (0.015, -0.68 - 0.05 * i), 0.032, (0.3, 0.3, 0.3))
            name = os.path.join(args.out, view, "%05d.png" % (k + h) if not args.steps
                                else "step_%08d.png" % step)
            vp.render_image(size=(W, H), filename=name, background=(1, 1, 1),
                            renderer=renderer())
    k += 1 if args.steps else args.hold
    print(step, flush=True)
