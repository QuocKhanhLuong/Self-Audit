"""Tiny draw.io page builder with a built-in layout checker and a matplotlib preview.

Every shape is placed on a 10 px grid and every edge is an explicit orthogonal polyline whose end points are tied to the
source and target with exitX/exitY/entryX/entryY.  The checker in :func:`check_page` enforces the figure rules (grid,
orthogonality, no line through a block or label, minimum gaps, crossing count) so that no render is needed to find
geometry mistakes.
"""
from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

FONT = "Times New Roman"
BLUE, RED, GREY = "#1F4E9A", "#D32F2F", "#757575"
EDGE_STYLES = {
    "data": f"strokeColor={BLUE};strokeWidth=1.5;",
    "skip": f"strokeColor={BLUE};strokeWidth=1.5;dashed=1;dashPattern=6 4;",
    "spacing": f"strokeColor={RED};strokeWidth=1.5;fontColor={RED};",
    "train": f"strokeColor={GREY};strokeWidth=1.2;dashed=1;dashPattern=1 3;",
}
EDGE_BASE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;jettySize=auto;html=1;endArrow=blockThin;endFill=1;endSize=6;"
             f"jumpStyle=arc;jumpSize=8;fontFamily={FONT};fontSize=12;exitPerimeter=0;entryPerimeter=0;")


@dataclass
class Shape:
    id: str
    kind: str  # cube box text icon image region overlay legend
    x: int
    y: int
    w: int
    h: int
    style: str = ""
    value: str = ""
    depth: int = 0


@dataclass
class Edge:
    id: str
    src: str | None
    tgt: str | None
    pts: list[tuple[int, int]]
    kind: str = "data"
    owner_of: str | None = None
    both: bool = False


@dataclass
class Page:
    name: str
    w: int
    h: int
    shapes: list[Shape] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    _n: int = 1

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def shape(self, shape: Shape) -> Shape:
        if any(s.id == shape.id for s in self.shapes):
            raise ValueError(f"duplicate id {shape.id}")
        self.shapes.append(shape)
        return shape

    def get(self, sid: str) -> Shape:
        return next(s for s in self.shapes if s.id == sid)

    # -- shapes ---------------------------------------------------------------------------------------------------
    def cube(self, sid, x, y, w, h, fill, label="", font="#FFFFFF", depth=10, front=None, front_frac=0.4):
        st = (f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.15;size={depth};"
              f"fillColor={fill};strokeColor=#000000;fontSize=14;fontColor={font};fontStyle=1;fontFamily={FONT};")
        s = self.shape(Shape(sid, "cube", x, y, w, h, st, label, depth))
        if front:  # lighter slab on the front face (two-tone encoder/decoder block)
            fw = int(round((w - depth) * front_frac / 10.0)) * 10 or 10
            self.shape(Shape(sid + "_f", "overlay", x, y + depth, fw, h - depth,
                             f"rounded=0;whiteSpace=wrap;html=1;strokeColor=#000000;strokeWidth=1;fillColor={front};", "", 0))
        return s

    def box(self, sid, x, y, w, h, label="", fill="#FFFFFF", stroke="#000000", font="#000000", size=14, dashed=False,
            bold=True, rounded=True, kind="box", stroke_w=1.5, vertical=False):
        st = (f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={stroke_w};"
              f"fontColor={font};fontSize={size};fontFamily={FONT};{'fontStyle=1;' if bold else ''}{'dashed=1;dashPattern=6 4;' if dashed else ''}"
              f"{'horizontal=0;' if vertical else ''}")
        return self.shape(Shape(sid, kind, x, y, w, h, st, label))

    def region(self, sid, x, y, w, h, fill, stroke, label="", dashed=True, width=2):
        st = (f"rounded=0;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={width};{'dashed=1;dashPattern=8 4;' if dashed else ''}"
              f"verticalAlign=top;align=left;spacingLeft=8;fontFamily={FONT};fontSize=14;fontStyle=1;")
        return self.shape(Shape(sid, "region", x, y, w, h, st, label))

    def text(self, sid, x, y, w, h, label, size=14, color="#000000", bold=False, align="center", kind="text"):
        st = (f"text;html=1;align={align};verticalAlign=middle;strokeColor=none;fillColor=none;fontFamily={FONT};fontSize={size};"
              f"fontColor={color};{'fontStyle=1;' if bold else ''}whiteSpace=wrap;")
        return self.shape(Shape(sid, kind, x, y, w, h, st, label))

    def icon(self, sid, x, y, label, fill, size=20):
        st = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontFamily={FONT};fillColor={fill};fontColor=#FFFFFF;fontSize=14;"
              f"strokeColor=none;fontStyle=1;spacing=0;")
        return self.shape(Shape(sid, "icon", x, y, size, size, st, label))

    def image(self, sid, x, y, w, h, png: Path | None, kind="image"):
        if png is not None and Path(png).is_file():
            data = base64.b64encode(Path(png).read_bytes()).decode()
            img = f"data:image/png,{data}"
        else:
            img = _placeholder_png()
        st = f"shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;strokeColor=#000000;strokeWidth=1;image={img};"
        return self.shape(Shape(sid, kind, x, y, w, h, st, ""))

    def edge(self, src, tgt, pts, kind="data", eid=None, owner_of=None, both=False):
        e = Edge(eid or self._id("e"), src, tgt, [tuple(map(int, p)) for p in pts], kind, owner_of, both)
        self.edges.append(e)
        return e

    # -- output ---------------------------------------------------------------------------------------------------
    def to_xml(self, diagram_id: str) -> str:
        out = [f'<diagram id={quoteattr(diagram_id)} name={quoteattr(self.name)}>',
               f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
               f'page="1" pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="1" shadow="0"><root>',
               '<mxCell id="0"/><mxCell id="1" parent="0"/>']
        order = {"region": 0, "overlay": 2}
        for s in sorted(self.shapes, key=lambda s: order.get(s.kind, 1) if s.kind != "overlay" else 2):
            out.append(f'<mxCell id={quoteattr(s.id)} value={quoteattr(s.value)} style={quoteattr(s.style)} vertex="1" parent="1">'
                       f'<mxGeometry x="{s.x}" y="{s.y}" width="{s.w}" height="{s.h}" as="geometry"/></mxCell>')
        for e in self.edges:
            style = EDGE_BASE + EDGE_STYLES[e.kind] + ("startArrow=blockThin;startFill=1;startSize=6;" if e.both else "")
            attrs = ""
            if e.src:
                s = self.get(e.src)
                style += f"exitX={(e.pts[0][0] - s.x) / s.w:.6f};exitY={(e.pts[0][1] - s.y) / s.h:.6f};exitDx=0;exitDy=0;"
                attrs += f" source={quoteattr(e.src)}"
            if e.tgt:
                t = self.get(e.tgt)
                style += f"entryX={(e.pts[-1][0] - t.x) / t.w:.6f};entryY={(e.pts[-1][1] - t.y) / t.h:.6f};entryDx=0;entryDy=0;"
                attrs += f" target={quoteattr(e.tgt)}"
            wp = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in e.pts[1:-1])
            if not e.src:
                src_pt = f'<mxPoint x="{e.pts[0][0]}" y="{e.pts[0][1]}" as="sourcePoint"/>'
            else:
                src_pt = ""
            if not e.tgt:
                tgt_pt = f'<mxPoint x="{e.pts[-1][0]}" y="{e.pts[-1][1]}" as="targetPoint"/>'
            else:
                tgt_pt = ""
            arr = f'<Array as="points">{wp}</Array>' if wp else ""
            out.append(f'<mxCell id={quoteattr(e.id)} value="" style={quoteattr(style)} edge="1" parent="1"{attrs}>'
                       f'<mxGeometry relative="1" as="geometry">{src_pt}{tgt_pt}{arr}</mxGeometry></mxCell>')
        out.append("</root></mxGraphModel></diagram>")
        return "".join(out)


def _placeholder_png() -> str:
    """Neutral grey 8x8 PNG used when the real slices have not been exported yet (clearly not a real slice)."""
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (8, 8), 128).save(buf, "PNG")
    return "data:image/png," + base64.b64encode(buf.getvalue()).decode()


def write_drawio(pages: list[Page], path: Path) -> None:
    body = "".join(p.to_xml(f"page{i}") for i, p in enumerate(pages))
    Path(path).write_text(f'<mxfile host="app.diagrams.net" agent="paper_figs/build_drawio.py" version="24.0.0">{body}</mxfile>\n')


# ----------------------------------------------------------------------------------------------------------------------
# checker
# ----------------------------------------------------------------------------------------------------------------------
def _segments(e: Edge):
    return list(zip(e.pts[:-1], e.pts[1:]))


def _seg_hits_rect(a, b, r, pad=0) -> bool:
    """Open segment (axis-aligned) intersects the interior of rect r=(x0,y0,x1,y1) enlarged by pad."""
    x0, y0, x1, y1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
    (ax, ay), (bx, by) = a, b
    if ax == bx:  # vertical
        lo, hi = sorted((ay, by))
        return x0 < ax < x1 and lo < y1 and hi > y0
    lo, hi = sorted((ax, bx))
    return y0 < ay < y1 and lo < x1 and hi > x0


def _rect(s: Shape):
    return (s.x, s.y, s.x + s.w, s.y + s.h)


def check_page(p: Page, *, min_gap=20, label_clear=10, max_cross=2) -> list[str]:
    """Return a list of rule violations (empty = clean)."""
    bad: list[str] = []
    for s in p.shapes:
        for k, v in (("x", s.x), ("y", s.y), ("w", s.w), ("h", s.h)):
            if v % 10:
                bad.append(f"grid: {s.id}.{k}={v}")
    for e in p.edges:
        for pt in e.pts:
            if pt[0] % 10 or pt[1] % 10:
                bad.append(f"grid: edge {e.id} point {pt}")
        for a, b in _segments(e):
            if a[0] != b[0] and a[1] != b[1]:
                bad.append(f"orthogonal: edge {e.id} segment {a}->{b}")
            if a == b:
                bad.append(f"degenerate: edge {e.id} repeats point {a}")
    blocks = [s for s in p.shapes if s.kind in ("cube", "box", "icon", "image")]
    solid = [s for s in p.shapes if s.kind in ("cube", "box", "icon", "image", "stack", "text")]
    for e in p.edges:
        for ep, sid in ((e.pts[0], e.src), (e.pts[-1], e.tgt)):
            if sid:
                s = p.get(sid)
                x0, y0, x1, y1 = _rect(s)
                on = (x0 <= ep[0] <= x1 and y0 <= ep[1] <= y1) and (ep[0] in (x0, x1) or ep[1] in (y0, y1))
                if not on:
                    bad.append(f"port: edge {e.id} end {ep} is not on the boundary of {sid}")
        for a, b in _segments(e):
            for s in solid:
                if s.id in (e.src, e.tgt):
                    continue
                pad = label_clear if s.kind == "text" and s.id != e.owner_of else 0
                if s.kind == "text" and s.id == e.owner_of:
                    continue
                if _seg_hits_rect(a, b, _rect(s), pad):
                    bad.append(f"through: edge {e.id} {a}->{b} touches {s.id} ({s.value[:20]!r})")
            for sid in (e.src, e.tgt):  # a segment may touch its own block only at the port
                if sid:
                    s = p.get(sid)
                    if (a, b) not in ((e.pts[0], e.pts[1]), (e.pts[-2], e.pts[-1])) and _seg_hits_rect(a, b, _rect(s), 0):
                        bad.append(f"through: edge {e.id} runs inside its own block {sid}")
    for i, s in enumerate(blocks):
        for t in blocks[i + 1:]:
            gx = max(s.x - (t.x + t.w), t.x - (s.x + s.w))
            gy = max(s.y - (t.y + t.h), t.y - (s.y + s.h))
            if max(gx, gy) < min_gap:
                bad.append(f"gap: {s.id} and {t.id} are {max(gx, gy)} px apart (< {min_gap})")
    cross = 0
    for i, e in enumerate(p.edges):
        for f in p.edges[i + 1:]:
            for a, b in _segments(e):
                for c, d in _segments(f):
                    if (a[0] == b[0]) == (c[0] == d[0]):
                        continue
                    (vx, vy0, vy1), (hy, hx0, hx1) = ((a[0], *sorted((a[1], b[1]))), (c[1], *sorted((c[0], d[0])))) if a[0] == b[0] else \
                        ((c[0], *sorted((c[1], d[1]))), (a[1], *sorted((a[0], b[0]))))
                    if hx0 < vx < hx1 and vy0 < hy < vy1:
                        cross += 1
    if cross > max_cross:
        bad.append(f"crossings: {cross} (> {max_cross})")
    p.crossings = cross  # type: ignore[attr-defined]
    return bad


# ----------------------------------------------------------------------------------------------------------------------
# preview (matplotlib, only to look at the layout; the real render is draw.io's)
# ----------------------------------------------------------------------------------------------------------------------
def _fill(style: str, default="#FFFFFF"):
    m = re.search(r"fillColor=(#\w+)", style)
    return m.group(1) if m else default


def preview(p: Page, path: Path, scale=1.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    fig, ax = plt.subplots(figsize=(p.w / 100 * scale, p.h / 100 * scale), dpi=100)
    ax.set_xlim(0, p.w), ax.set_ylim(p.h, 0), ax.set_aspect("equal"), ax.axis("off")
    for s in sorted(p.shapes, key=lambda s: {"region": 0}.get(s.kind, 2 if s.kind == "overlay" else 1)):
        col = _fill(s.style)
        if s.kind == "cube":
            d = s.depth
            ax.add_patch(Rectangle((s.x, s.y + d), s.w - d, s.h - d, fc=col, ec="k", lw=0.8))
            ax.add_patch(Polygon([(s.x, s.y + d), (s.x + d, s.y), (s.x + s.w, s.y), (s.x + s.w - d, s.y + d)], fc=col, ec="k", lw=0.8, alpha=0.8))
            ax.add_patch(Polygon([(s.x + s.w - d, s.y + d), (s.x + s.w, s.y), (s.x + s.w, s.y + s.h - d), (s.x + s.w - d, s.y + s.h)], fc=col, ec="k", lw=0.8, alpha=0.6))
        elif s.kind in ("icon",):
            ax.add_patch(plt.Circle((s.x + s.w / 2, s.y + s.h / 2), s.w / 2, fc=col, ec="none"))
        elif s.kind in ("image", "stack"):
            ax.add_patch(Rectangle((s.x, s.y), s.w, s.h, fc="#9E9E9E", ec="k", lw=0.8))
        elif s.kind == "text":
            pass
        else:
            ax.add_patch(Rectangle((s.x, s.y), s.w, s.h, fc=col if "fillColor=none" not in s.style else "none", ec="k", lw=0.8,
                                   ls="--" if "dashed=1" in s.style else "-", alpha=0.9))
        if s.value and s.kind != "overlay":
            txt = re.sub(r"<br\s*/?>", "\n", s.value)
            txt = re.sub(r"\\\(|\\\)", "", txt).replace("\\mathbf", "").replace("\\,", " ").replace("\\", "")
            ax.text(s.x + s.w / 2, s.y + (s.h / 2 if s.kind != "region" else 10), txt, ha="center", va="center", fontsize=6.5,
                    color="#1b1b1b" if s.kind in ("text", "region") else "#111", clip_on=False)
    colors = {"data": "#1F4E9A", "skip": "#1F4E9A", "spacing": "#D32F2F", "train": "#757575"}
    for e in p.edges:
        xs, ys = zip(*e.pts)
        ax.plot(xs, ys, color=colors[e.kind], lw=1.2, ls="--" if e.kind in ("skip", "train") else "-")
        (x0, y0), (x1, y1) = e.pts[-2], e.pts[-1]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=colors[e.kind], lw=1.2))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
