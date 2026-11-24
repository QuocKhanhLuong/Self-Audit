#!/usr/bin/env python3
"""Independent check of a finished .drawio file: every vertex x/y/width/height and every edge waypoint is on the 10 px grid,
every edge has explicit exit/entry constraints where it has a source/target, and every page has math enabled."""
import re
import sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
root = ET.parse(path).getroot()
total_bad = 0
for d in root.iter("diagram"):
    model = d.find("mxGraphModel")
    cells = list(d.iter("mxCell"))
    bad, nv, ne, imgs = [], 0, 0, 0
    for c in cells:
        g = c.find("mxGeometry")
        if c.get("vertex") == "1":
            nv += 1
            imgs += "image=data:image/png," in (c.get("style") or "")
            for k in ("x", "y", "width", "height"):
                if float(g.get(k)) % 10:
                    bad.append(f"vertex {c.get('id')}.{k}={g.get(k)}")
        elif c.get("edge") == "1":
            ne += 1
            st = c.get("style") or ""
            if "edgeStyle=orthogonalEdgeStyle" not in st or "rounded=0" not in st:
                bad.append(f"edge {c.get('id')} is not orthogonal/rounded=0")
            for side, key in (("source", "exit"), ("target", "entry")):
                if c.get(side) and f"{key}X=" not in st:
                    bad.append(f"edge {c.get('id')} has a {side} but no {key}X/{key}Y")
            for pt in g.iter("mxPoint"):
                if float(pt.get("x")) % 10 or float(pt.get("y")) % 10:
                    bad.append(f"edge {c.get('id')} point ({pt.get('x')},{pt.get('y')}) off grid")
    if model.get("math") != "1":
        bad.append("math typesetting is off")
    print(f"{d.get('name')}: {nv} vertices, {ne} edges, {imgs} embedded images, gridSize={model.get('gridSize')}, math={model.get('math')}, {len(bad)} problems")
    for b in bad:
        print("   ", b)
    total_bad += len(bad)
sys.exit(1 if total_bad else 0)
