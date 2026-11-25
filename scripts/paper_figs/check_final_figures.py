"""Geometry and artifact checks for the SVG-native publication figures.

Checks measured glyph bounds, visible cube ports, orthogonal connectors,
connector/label collisions, connector/node collisions and collinear overlaps.
Visual inspection is still required: a passing geometry audit is not a render.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

from PIL import Image


def rect(node):
    return (node.x, node.y-node.dy, node.x+node.w+node.dx, node.y+node.h)


def contains(a, b, tol=.2):
    return a[0]-tol <= b[0] and a[1]-tol <= b[1] and a[2]+tol >= b[2] and a[3]+tol >= b[3]


def intersects(a, b, tol=.2):
    return min(a[2], b[2])-max(a[0], b[0]) > tol and min(a[3], b[3])-max(a[1], b[1]) > tol


def line_hits(a, b, box, pad=0):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0-pad, y0-pad, x1+pad, y1+pad
    if a[0] == b[0]:
        lo, hi = sorted((a[1],b[1]))
        return x0+.01 < a[0] < x1-.01 and lo < y1-.01 and hi > y0+.01
    lo, hi = sorted((a[0],b[0]))
    return y0+.01 < a[1] < y1-.01 and lo < x1-.01 and hi > x0+.01


def on_boundary(p, node, tol=.15):
    x,y,w,h,dx,dy = node.x,node.y,node.w,node.h,node.dx,node.dy
    # The actual visible silhouette, not the enclosing rectangle of a cube.
    polygon = [(x,y),(x+dx,y-dy),(x+w+dx,y-dy),(x+w+dx,y+h-dy),(x+w,y+h),(x,y+h)] if dx else [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    for a,b in zip(polygon, polygon[1:]+polygon[:1]):
        length = math.hypot(b[0]-a[0],b[1]-a[1])
        distance = abs((b[0]-a[0])*(a[1]-p[1])-(a[0]-p[0])*(b[1]-a[1])) / length
        if distance <= tol and min(a[0],b[0])-tol <= p[0] <= max(a[0],b[0])+tol and min(a[1],b[1])-tol <= p[1] <= max(a[1],b[1])+tol:
            return True
    return False


def check_geometry(f):
    errors = []
    nodes = {k:n for k,n in f.nodes.items() if k not in f.backgrounds | f.masks}
    boxes = {k:rect(n) for k,n in nodes.items()}
    labels = [(v["text"], (v["box"][0], v["box"][1], v["box"][0]+v["box"][2], v["box"][1]+v["box"][3])) for v in f.labels]
    for name,box in labels:
        if not contains((0,0,f.w,f.h),box): errors.append(f"Label outside page: {name}")
        for sid,b in boxes.items():
            if intersects(box,b) and not contains(b,box):
                errors.append(f"Label partly overlaps {sid}: {name}")
    for (name,a),(other,b) in combinations(labels,2):
        if intersects(a,b): errors.append(f"Overlapping labels: {name} / {other}")
    for (sid,a),(tid,b) in combinations(boxes.items(),2):
        if intersects(a,b) and not contains(a,b) and not contains(b,a):
            errors.append(f"Overlapping nodes: {sid} / {tid}")
    for i,e in enumerate(f.edges):
        for key,p in [("source",e["points"][0]),("target",e["points"][-1])]:
            sid=e.get(key)
            if sid and (sid not in f.nodes or not on_boundary(p,f.nodes[sid])):
                errors.append(f"Edge {i} {key} misses visible face: {sid} at {p}")
        for a,b in zip(e["points"],e["points"][1:]):
            for sid,box in boxes.items():
                if sid not in (e.get("source"),e.get("target")) and line_hits(a,b,box):
                    errors.append(f"Edge {i} crosses node {sid}")
            for name,box in labels:
                if any(contains(rect(f.nodes[s]),box) for s in f.masks): continue
                if line_hits(a,b,box,pad=2): errors.append(f"Edge {i} crosses label: {name}")
    crossings = 0
    for (i,e),(j,g) in combinations(enumerate(f.edges),2):
        for a,b in zip(e["points"],e["points"][1:]):
            for c,d in zip(g["points"],g["points"][1:]):
                vertical = a[0] == b[0]
                same_axis = vertical == (c[0] == d[0])
                if same_axis:
                    fixed,axis = (0,1) if vertical else (1,0)
                    if a[fixed] == c[fixed] and min(max(a[axis],b[axis]),max(c[axis],d[axis])) > max(min(a[axis],b[axis]),min(c[axis],d[axis]))+.01:
                        errors.append(f"Edges {i}/{j} share a stroke")
                else:
                    v0,v1,h0,h1=(a,b,c,d) if vertical else (c,d,a,b)
                    if min(h0[0],h1[0]) < v0[0] < max(h0[0],h1[0]) and min(v0[1],v1[1]) < h0[1] < max(v0[1],v1[1]):
                        crossings += 1
                        errors.append(f"Edges {i}/{j} cross")
    return {"errors":sorted(set(errors)), "crossings":crossings, "nodes":len(nodes), "edges":len(f.edges), "labels":len(labels)}


def check_exports(f, directory):
    path = Path(directory) / f.slug
    root = ET.parse(path.with_suffix(".svg")).getroot()
    assert root.get("width") == "7in"
    assert root.get("viewBox") == f"0 {f.crop_top} 1008 {f.eff_h}"
    assert root.get("role") == "img"
    assert len(root.findall("{http://www.w3.org/2000/svg}title")) == 1
    assert len(list(root.iter("{http://www.w3.org/2000/svg}image"))) == len(f.images)
    with Image.open(path.with_suffix(".png")) as png:
        assert png.size == (2016,f.eff_h*2), png.size
    import pymupdf as fitz
    with fitz.open(path.with_suffix(".pdf")) as pdf:
        assert len(pdf) == 1
        page=pdf[0]
        assert abs(page.rect.width-504) < .01
        assert abs(page.rect.height-f.eff_h/2) < .01
        assert len(page.get_drawings()) > 10, "PDF diagram must remain vector"
    return {"svg_valid":True, "png_2x":True, "pdf_vector":True, "pdf_width_pt":504}
