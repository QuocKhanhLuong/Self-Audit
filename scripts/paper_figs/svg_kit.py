"""Deterministic, print-sized SVG primitives for the paper figures.

The 1008-unit canvas is seven inches wide: 14 units == 7 pt. Text is
outlined from bundled Matplotlib DejaVu fonts; MRI PNGs remain embedded
rasters. All diagram geometry and math stay vector in SVG and PDF.
"""
from __future__ import annotations

import base64
import html
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath

matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

INK = "#203346"
MUTED = "#506579"
BLUE = "#3678A3"
ENC = "#C2DAEB"
TEAL = "#2E7C71"
DEC = "#BBDDD4"
ORANGE = "#B7632D"
ALIGN = "#F4D7BE"
GOLD = "#8C6819"
SSM = "#F3E2AD"
RED = "#A74A50"
PINK = "#F9ECEB"
GREY = "#64717C"
LIGHT = "#F4F7FA"
RULE = "#D8E0E6"
WHITE = "#FFFFFF"


def fmt(v):
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def tint(color, amount):
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    target = 255 if amount > 0 else 0
    return "#" + "".join(f"{round(c + abs(amount) * (target - c)):02x}" for c in rgb)


@dataclass
class Node:
    id: str
    x: float
    y: float
    w: float
    h: float
    dx: float = 0
    dy: float = 0

    @property
    def left(self): return (self.x, self.y + self.h / 2)
    @property
    def right(self): return (self.x + self.w + self.dx, self.y + self.h / 2)
    @property
    def top(self): return (self.x + self.w / 2 + self.dx, self.y - self.dy)
    @property
    def bottom(self): return (self.x + self.w / 2, self.y + self.h)
    @property
    def center(self): return (self.x + self.w / 2, self.y + self.h / 2)


class Figure:
    def __init__(self, slug, height, title, description):
        self.slug, self.w, self.h = slug, 1008, height
        self.title, self.description = title, description
        self.layers = {key: [] for key in ("bg", "edges", "nodes", "text")}
        self.nodes = {}
        self.records = []
        self.labels = []
        self.edges = []
        self.images = []
        self.backgrounds = set()
        self.masks = set()
        self.min_font = 100

    def add(self, svg, layer="nodes"):
        self.layers[layer].append(svg)

    def text(self, x, y, text, size=16, *, color=INK, weight="normal", anchor="middle", layer="text"):
        """Text positioned by visual center (x,y); all glyphs are vector outlines."""
        if size < 14:
            raise ValueError(f"Text below 7 pt at print size: {text}")
        self.min_font = min(self.min_font, size)
        prop = FontProperties(family="DejaVu Sans", weight=weight)
        path = TextPath((0, 0), text, size=size, prop=prop, usetex=False)
        bb = path.get_extents()
        if anchor == "middle": tx = x - (bb.x0 + bb.x1) / 2
        elif anchor == "start": tx = x - bb.x0
        elif anchor == "end": tx = x - bb.x1
        else: raise ValueError(anchor)
        ty = y + (bb.y0 + bb.y1) / 2
        d = []
        for vertices, code in path.iter_segments(curves=True, simplify=False):
            if code == MplPath.CLOSEPOLY:
                d.append("Z")
            else:
                command = {MplPath.MOVETO: "M", MplPath.LINETO: "L", MplPath.CURVE3: "Q", MplPath.CURVE4: "C"}[code]
                d.append(command + " ".join(fmt(v) for v in vertices))
        self.add(f'<g aria-label="{html.escape(text, quote=True)}"><path fill="{color}" '
                 f'transform="translate({fmt(tx)} {fmt(ty)}) scale(1 -1)" d="{" ".join(d)}"/></g>', layer)
        self.labels.append({"text": text, "size": size, "box": [tx + bb.x0, ty - bb.y1, bb.width, bb.height]})
        self.records.append(("text", (x, y, text, size, color, weight, anchor)))

    def lines(self, x, y, lines, size=16, *, leading=None, color=INK, weight="normal", anchor="middle"):
        leading = leading or size * 1.35
        for i, line in enumerate(lines):
            self.text(x, y + (i - (len(lines) - 1) / 2) * leading, line, size, color=color, weight=weight, anchor=anchor)

    def heading(self, title, subtitle=None):
        """Titles belong in the LaTeX caption; the figure carries none."""
        return None

    @property
    def crop_top(self):
        tops = [n.y - getattr(n, "dy", 0) for n in self.nodes.values()] + [l["box"][1] for l in self.labels]
        return max(0, int((min(tops) - 12) // 2 * 2))

    @property
    def eff_h(self):
        return self.h - self.crop_top

    def rect(self, sid, x, y, w, h, *, fill=WHITE, stroke=INK, radius=4, dashed=False, layer="nodes", width=1.6):
        node = Node(sid, x, y, w, h)
        if sid in self.nodes: raise ValueError(f"Duplicate node: {sid}")
        self.nodes[sid] = node
        if layer == "bg": self.backgrounds.add(sid)
        if layer == "text": self.masks.add(sid)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.add(f'<rect id="{sid}" x="{fmt(x)}" y="{fmt(y)}" width="{fmt(w)}" height="{fmt(h)}" '
                 f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash}/>', layer)
        self.records.append(("rect", (sid, x, y, w, h, fill, stroke, radius, dashed)))
        return node

    def box(self, sid, x, y, w, h, label, *, fill=LIGHT, stroke=BLUE, size=16, color=INK, weight="normal", dashed=False):
        node = self.rect(sid, x, y, w, h, fill=fill, stroke=stroke, dashed=dashed)
        self.lines(*node.center, label.split("\n") if isinstance(label, str) else label, size=size, color=color, weight=weight)
        return node

    def cube(self, sid, x, y, w, h, *, fill=ENC, label=None, size=18):
        """A single oblique projection; depth vector is always (+16,-12)."""
        dx, dy = 16, 12
        node = Node(sid, x, y, w, h, dx, dy)
        if sid in self.nodes: raise ValueError(f"Duplicate node: {sid}")
        self.nodes[sid] = node
        faces = [([(x, y), (x + dx, y - dy), (x + w + dx, y - dy), (x + w, y)], tint(fill, .35)),
                 ([(x + w, y), (x + w + dx, y - dy), (x + w + dx, y + h - dy), (x + w, y + h)], tint(fill, -.18)),
                 ([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], fill)]
        body = "".join(f'<polygon points="{" ".join(f"{fmt(a)},{fmt(b)}" for a,b in pts)}" '
                       f'fill="{col}" stroke="{BLUE if fill == ENC else TEAL if fill == DEC else INK}" stroke-width="1.5" stroke-linejoin="round"/>' for pts, col in faces)
        self.add(f'<g id="{sid}" data-depth="16,-12">{body}</g>')
        self.records.append(("cube", (sid, x, y, w, h, fill)))
        if label: self.text(*node.center, label, size, weight="bold")
        return node

    def circle(self, sid, cx, cy, r=14, *, label="+", fill=WHITE, stroke=BLUE, size=20):
        node = Node(sid, cx - r, cy - r, 2*r, 2*r)
        self.nodes[sid] = node
        self.add(f'<circle id="{sid}" cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        self.text(cx, cy, label, size)
        self.records.append(("circle", (sid, cx, cy, r, label, fill, stroke)))
        return node

    def edge(self, points, *, color=BLUE, dashed=False, arrow=True, width=1.8, source=None, target=None):
        """Orthogonal polyline with small rounded bends and tip exactly at endpoint."""
        points = [tuple(p) for p in points]
        for a, b in zip(points, points[1:]):
            if a == b or (a[0] != b[0] and a[1] != b[1]):
                raise ValueError(f"Nonorthogonal/zero edge: {a} to {b}")
        d = [f"M {fmt(points[0][0])} {fmt(points[0][1])}"]
        for i, b in enumerate(points[1:-1], 1):
            a, c = points[i - 1], points[i + 1]
            length1 = abs(b[0]-a[0]) + abs(b[1]-a[1])
            length2 = abs(c[0]-b[0]) + abs(c[1]-b[1])
            r = min(6, length1/2, length2/2)
            p = (b[0] + (a[0]-b[0])*r/length1, b[1] + (a[1]-b[1])*r/length1)
            q = (b[0] + (c[0]-b[0])*r/length2, b[1] + (c[1]-b[1])*r/length2)
            d += [f"L {fmt(p[0])} {fmt(p[1])}", f"Q {fmt(b[0])} {fmt(b[1])} {fmt(q[0])} {fmt(q[1])}"]
        d.append(f"L {fmt(points[-1][0])} {fmt(points[-1][1])}")
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        marker = f' marker-end="url(#{self.slug}-arrow-{color[1:]})"' if arrow else ""
        self.add(f'<path d="{" ".join(d)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}{marker}/>', "edges")
        self.edges.append({"points": points, "color":color, "dashed":dashed, "arrow":arrow, "source":source, "target":target})

    def connect(self, a, b, *, via=(), **kwargs):
        self.edge([a, *via, b], **kwargs)

    def image(self, sid, x, y, size, path):
        path = Path(path)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        self.add(f'<image id="{sid}" x="{x}" y="{y}" width="{size}" height="{size}" '
                 f'href="data:image/png;base64,{data}" preserveAspectRatio="xMidYMid meet"/>')
        node = Node(sid, x, y, size, size)
        self.nodes[sid] = node
        self.images.append({"id":sid, "path":str(path), "x":x, "y":y, "size":size})
        self.records.append(("image", (sid, x, y, size, path)))
        return node

    def rule(self, y):
        if y < 100:
            return
        self.add(f'<path d="M24 {y} H984" stroke="{RULE}" stroke-width="1"/>', "bg")

    def svg(self):
        colors = sorted(set(e["color"] for e in self.edges))
        defs = "".join(f'<marker id="{self.slug}-arrow-{c[1:]}" markerWidth="7" markerHeight="7" '
                       f'refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L7 3.5 L0 7 Z" fill="{c}"/></marker>' for c in colors)
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="7in" height="{self.eff_h / 144:.8f}in" viewBox="0 {self.crop_top} 1008 {self.eff_h}" role="img" '
                f'aria-labelledby="{self.slug}-title {self.slug}-desc">'
                f'<title id="{self.slug}-title">{html.escape(self.title)}</title>'
                f'<desc id="{self.slug}-desc">{html.escape(self.description)}</desc><defs>{defs}</defs>'
                f'<rect y="{self.crop_top}" width="1008" height="{self.eff_h}" fill="white"/>'
                + "".join("".join(self.layers[k]) for k in self.layers) + '</svg>\n')

    def export(self, directory, renderer="rsvg-convert"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        svg = directory / f"{self.slug}.svg"
        svg.write_text(self.svg())
        env = dict(os.environ, SOURCE_DATE_EPOCH="1795564800")
        subprocess.run([renderer, "--format=pdf", "--output", str(svg.with_suffix(".pdf")), str(svg)], check=True, env=env)
        subprocess.run([renderer, "--format=png", "--width=2016", f"--height={self.eff_h*2}",
                        "--output", str(svg.with_suffix(".png")), str(svg)], check=True)
        return {"figure": self.slug, "canvas": [1008, self.eff_h], "crop_top": self.crop_top, "png": [2016, self.eff_h*2],
                "width_inches":7, "minimum_font_pt":self.min_font/2, "labels":len(self.labels),
                "cubes":sum(r[0] == "cube" for r in self.records), "embedded_images":len(self.images)}
