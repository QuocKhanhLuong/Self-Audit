"""Export the publication figures to an editable, native draw.io companion.

The paper figures are authored with :mod:`svg_kit`, whose records retain the
semantic primitives used to build each drawing.  This adapter deliberately
replays those records as native ``mxCell`` vertices and edges instead of
embedding a rendered SVG.  Images remain images (the five patient slices in
figures 1 and 4); boxes, cubes, circles, labels, and formulas remain editable
draw.io elements.

The native ``shape=cube`` stencil is used for cubes.  Its shading is close to
the SVG three-face projection, but draw.io owns the final face geometry; the
vector SVG/PDF exports remain the visual source of truth.  A ``gridSize=1``
model is emitted because the SVG layouts contain half-unit coordinates.
"""
from __future__ import annotations

import base64
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import quoteattr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drawio_kit import FONT, Page, Shape, write_drawio  # noqa: E402


_MATH_RE = re.compile(r"\$(?!\s)(.+?)(?<!\s)\$", re.DOTALL)
_TOKEN_RE = re.compile(r"([A-Za-z]+|-?\d+(?:\.\d+)?)")
def _fmt(value: float | int) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite coordinate: {value!r}")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _mathify(value: str) -> str:
    """Use draw.io's native MathJax delimiters for SVG-kit ``$...$`` labels."""

    if "$" not in value:
        return value
    return _MATH_RE.sub(lambda match: r"\(" + match.group(1) + r"\)", value)


def _color(value: str | None, default: str = "#203346") -> str:
    if not value or value in ("none", "transparent"):
        return "none" if value == "none" else default
    return value


def _text_metrics(value: str, size: float) -> tuple[float, float]:
    """Fallback bounds when an older Figure does not expose measured labels."""

    lines = value.splitlines() or [""]
    longest = max((len(line) for line in lines), default=1)
    width = max(24.0, min(980.0, longest * max(size * 0.33, 4.0) + 18.0))
    height = max(16.0, max(size * 0.95, 8.0) * max(1, len(lines)) + 8.0)
    return width, height


def _bounds(shape: Shape) -> tuple[float, float, float, float]:
    """Bounds stored for a vertex; cube geometry already encloses all three faces."""

    return shape.x, shape.y, shape.x + shape.w, shape.y + shape.h


def _on_or_inside(point: tuple[float, float], bounds: tuple[float, float, float, float], eps: float = 1e-4) -> bool:
    x, y = point
    x0, y0, x1, y1 = bounds
    return x0 - eps <= x <= x1 + eps and y0 - eps <= y <= y1 + eps


def _native_style(kind: str, *, fill: str = "#FFFFFF", stroke: str = "#203346", radius: float = 4,
                  dashed: bool = False, font_size: float = 7, font_color: str = "#203346",
                  align: str = "center", bold: bool = False, value: str = "", depth: float = 16) -> str:
    dash = "dashed=1;dashPattern=6 5;" if dashed else ""
    if kind == "cube":
        return (
            "shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;"
            f"darkOpacity=0.05;darkOpacity2=0.15;size={_fmt(depth)};"
            f"fillColor={_color(fill)};strokeColor={_color(stroke)};strokeWidth=1.5;"
            f"fontFamily=DejaVu Sans;fontSize={_fmt(font_size)};fontColor={_color(font_color)};"
        )
    if kind == "circle":
        return (
            "ellipse;aspect=fixed;whiteSpace=wrap;html=1;"
            f"fillColor={_color(fill)};strokeColor={_color(stroke)};strokeWidth=1.6;"
            f"fontFamily=DejaVu Sans;fontSize={_fmt(font_size)};fontColor={_color(font_color)};"
        )
    if kind == "image":
        return (
            "shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;"
            f"strokeColor={_color(stroke)};strokeWidth=1;"
        )
    if kind == "text":
        return (
            f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;strokeColor=none;fillColor=none;"
            f"fontFamily=DejaVu Sans;fontSize={_fmt(font_size)};fontColor={_color(font_color)};"
            f"{'fontStyle=1;' if bold else ''}"
        )
    return (
        f"rounded={1 if radius else 0};whiteSpace=wrap;html=1;"
        f"fillColor={_color(fill)};strokeColor={_color(stroke)};strokeWidth=1.4;"
        f"fontFamily=DejaVu Sans;fontSize={_fmt(font_size)};fontColor={_color(font_color)};"
        f"{'fontStyle=1;' if bold else ''}{dash}"
    )


@dataclass
class _EditableEdge:
    id: str
    src: str | None
    tgt: str | None
    pts: list[tuple[float, float]]
    color: str = "#3678A3"
    dashed: bool = False
    arrow: bool = True
    both: bool = False


class _EditablePage(Page):
    """Page with float-safe geometry and the one-unit grid used by SVG records."""

    def to_xml(self, diagram_id: str) -> str:  # noqa: D401
        out = [
            f'<diagram id={quoteattr(diagram_id)} name={quoteattr(self.name)}>',
            (
                f'<mxGraphModel dx="{_fmt(self.w)}" dy="{_fmt(self.h)}" grid="1" gridSize="1" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{_fmt(self.w)}" pageHeight="{_fmt(self.h)}" math="1" shadow="0"><root>'
            ),
            '<mxCell id="0"/>',
            '<mxCell id="1" parent="0"/>',
        ]
        for shape in self.shapes:
            out.append(
                f'<mxCell id={quoteattr(shape.id)} value={quoteattr(shape.value or "")} '
                f'style={quoteattr(shape.style)} vertex="1" parent="1">'
                f'<mxGeometry x="{_fmt(shape.x)}" y="{_fmt(shape.y)}" width="{_fmt(shape.w)}" '
                f'height="{_fmt(shape.h)}" as="geometry"/></mxCell>'
            )
        for edge in self.edges:
            if not isinstance(edge, _EditableEdge):
                raise TypeError(f"editable page contains a non-editable edge: {edge!r}")
            src = self._shape(edge.src) if edge.src else None
            tgt = self._shape(edge.tgt) if edge.tgt else None
            style = (
                "edgeStyle=orthogonalEdgeStyle;rounded=0;jettySize=auto;html=1;"
                f"endArrow={'blockThin' if edge.arrow else 'none'};endFill={'1' if edge.arrow else '0'};"
                "endSize=6;startArrow=none;exitPerimeter=0;entryPerimeter=0;"
                f"strokeColor={_color(edge.color)};strokeWidth=1.4;"
                f"{'dashed=1;dashPattern=6 5;' if edge.dashed else ''}"
                f"fontFamily={FONT};fontSize=7;"
            )
            attrs = ""
            if src:
                x0, y0, x1, y1 = _bounds(src)
                sx = 0.0 if x1 == x0 else (edge.pts[0][0] - x0) / (x1 - x0)
                sy = 0.0 if y1 == y0 else (edge.pts[0][1] - y0) / (y1 - y0)
                style += f"exitX={_fmt(sx)};exitY={_fmt(sy)};exitDx=0;exitDy=0;"
                attrs += f" source={quoteattr(src.id)}"
            if tgt:
                x0, y0, x1, y1 = _bounds(tgt)
                tx = 0.0 if x1 == x0 else (edge.pts[-1][0] - x0) / (x1 - x0)
                ty = 0.0 if y1 == y0 else (edge.pts[-1][1] - y0) / (y1 - y0)
                style += f"entryX={_fmt(tx)};entryY={_fmt(ty)};entryDx=0;entryDy=0;"
                attrs += f" target={quoteattr(tgt.id)}"
            points = "".join(
                f'<mxPoint x="{_fmt(x)}" y="{_fmt(y)}"/>' for x, y in edge.pts[1:-1]
            )
            source_point = "" if src else f'<mxPoint x="{_fmt(edge.pts[0][0])}" y="{_fmt(edge.pts[0][1])}" as="sourcePoint"/>'
            target_point = "" if tgt else f'<mxPoint x="{_fmt(edge.pts[-1][0])}" y="{_fmt(edge.pts[-1][1])}" as="targetPoint"/>'
            arr = f'<Array as="points">{points}</Array>' if points else ""
            out.append(
                f'<mxCell id={quoteattr(edge.id)} value="" style={quoteattr(style)} edge="1" parent="1"{attrs}>'
                f'<mxGeometry relative="1" as="geometry">{source_point}{target_point}{arr}</mxGeometry></mxCell>'
            )
        out.append("</root></mxGraphModel></diagram>")
        return "".join(out)

    def _shape(self, sid: str) -> Shape:
        return next(s for s in self.shapes if s.id == sid)


def _edge_color(edge: Mapping[str, Any]) -> str:
    return str(edge.get("color") or "#3678A3")


def _edge_from_record(index: int, edge: Mapping[str, Any]) -> _EditableEdge:
    pts = [(float(x), float(y)) for x, y in edge.get("points", [])]
    if len(pts) < 2:
        raise ValueError(f"edge {index} has fewer than two points")
    color = _edge_color(edge)
    return _EditableEdge(
        id=f"edge_{index}",
        src=edge.get("source"),
        tgt=edge.get("target"),
        pts=pts,
        color=color,
        dashed=bool(edge.get("dashed", False)),
        arrow=bool(edge.get("arrow", True)),
    )


def _parse_background(fig: Any, page: _EditablePage, used: set[str], record_ids: set[str] | None = None) -> None:
    """Retain the few panel/rule primitives added directly to ``Figure.layers``."""

    record_ids = record_ids or set()
    snippets = fig.layers.get("bg", {}) if hasattr(fig, "layers") else []
    for snippet_index, snippet in enumerate(snippets):
        try:
            root = ET.fromstring(f"<svg>{snippet}</svg>")
        except ET.ParseError:
            continue
        for item_index, elem in enumerate(root):
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "rect":
                if elem.get("id") in record_ids:
                    continue
                sid = _fresh_id(f"bg_rect_{snippet_index}_{item_index}", used)
                fill = _color(elem.get("fill"), "none")
                stroke = _color(elem.get("stroke"), "none")
                style = _native_style(
                    "rect", fill=fill, stroke=stroke, radius=float(elem.get("rx", 0) or 0),
                    font_size=7,
                )
                page.shape(Shape(sid, "region", float(elem.get("x", 0)), float(elem.get("y", 0)),
                                 float(elem.get("width", 0)), float(elem.get("height", 0)), style, ""))
            elif tag == "path":
                points = _simple_path_points(elem.get("d", ""))
                if len(points) >= 2:
                    page.edges.append(_EditableEdge(
                        id=_fresh_id(f"bg_line_{snippet_index}_{item_index}", used),
                        src=None,
                        tgt=None,
                        pts=points,
                        color=_color(elem.get("stroke"), "#D8E0E6"),
                        dashed=False,
                        arrow=False,
                    ))


def _simple_path_points(path: str) -> list[tuple[float, float]]:
    """Parse the horizontal/vertical paths used by ``Figure.rule`` and panels."""

    tokens = _TOKEN_RE.findall(path)
    if not tokens:
        return []
    points: list[tuple[float, float]] = []
    i = 0
    command = ""
    x = y = 0.0
    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            command = token
            i += 1
            continue
        if command in ("M", "L") and i + 1 < len(tokens):
            x, y = float(tokens[i]), float(tokens[i + 1])
            points.append((x, y))
            i += 2
        elif command == "H":
            x = float(token)
            points.append((x, y))
            i += 1
        elif command == "V":
            y = float(token)
            points.append((x, y))
            i += 1
        else:
            return []
    return points


def _fresh_id(base: str, used: set[str]) -> str:
    sid = base
    suffix = 1
    while sid in used:
        sid = f"{base}_{suffix}"
        suffix += 1
    used.add(sid)
    return sid


def _append_records(fig: Any, page: _EditablePage, used: set[str]) -> None:
    records = list(getattr(fig, "records", []))
    # Native visual nodes come first, then native text cells.  This keeps labels
    # above cubes/boxes in draw.io and leaves all of them independently editable.
    node_records = [record for record in records if record and record[0] in {"rect", "cube", "circle", "image"}]
    text_records = [record for record in records if record and record[0] == "text"]
    for kind, values in node_records:
        if kind == "rect":
            sid, x, y, w, h, fill, stroke, radius, dashed = values
            sid = _fresh_id(str(sid), used)
            page.shape(Shape(sid, "box", float(x), float(y), float(w), float(h), _native_style(
                "box", fill=_color(fill), stroke=_color(stroke), radius=float(radius or 0), dashed=bool(dashed),
            ), ""))
        elif kind == "cube":
            sid, x, y, w, h, fill = values
            sid = _fresh_id(str(sid), used)
            # svg_kit cubes use the fixed oblique vector (+16, -12).  Store the
            # complete visible silhouette in the mxGeometry bounds so ports
            # and endpoint validation refer to the same rectangle as the SVG.
            cube_x, cube_y = float(x), float(y) - 12.0
            cube_w, cube_h = float(w) + 16.0, float(h) + 12.0
            page.shape(Shape(sid, "cube", cube_x, cube_y, cube_w, cube_h, _native_style(
                "cube", fill=_color(fill), stroke="#203346", depth=16,
            ), "", 16))
        elif kind == "circle":
            sid, cx, cy, r, _label, fill, stroke = values
            sid = _fresh_id(str(sid), used)
            page.shape(Shape(sid, "circle", float(cx) - float(r), float(cy) - float(r), float(r) * 2,
                             float(r) * 2, _native_style("circle", fill=_color(fill), stroke=_color(stroke)), ""))
        elif kind == "image":
            sid, x, y, size, image_path = values
            sid = _fresh_id(str(sid), used)
            image_path = Path(image_path)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            style = _native_style("image", stroke="#D8E0E6") + f"image=data:image/png,{image_data};"
            page.shape(Shape(sid, "image", float(x), float(y), float(size), float(size), style, ""))
    measured = list(getattr(fig, "labels", []))
    for text_index, (kind, values) in enumerate(text_records):
        x, y, value, size, color, _weight, anchor = values
        value = _mathify(str(value))
        size = float(size)
        if text_index < len(measured) and len(measured[text_index].get("box", ())) == 4:
            # Figure.text stores the exact TextPath glyph bounds.  A small
            # margin leaves a usable draw.io edit box without reflowing labels.
            bx, by, bw, bh = (float(v) for v in measured[text_index]["box"])
            pad = 3.0
            x0, y0, w, h = bx - pad, by - pad, bw + 2 * pad, bh + 2 * pad
        else:
            w, h = _text_metrics(value, size)
            if anchor == "start":
                x0 = float(x)
            elif anchor == "end":
                x0 = float(x) - w
            else:
                x0 = float(x) - w / 2
            y0 = float(y) - h / 2
        align = {"start": "left", "end": "right", "middle": "center"}.get(str(anchor), "center")
        sid = _fresh_id(f"text_{text_index}", used)
        page.shape(Shape(sid, "text", x0, y0, w, h, _native_style(
            "text", font_size=size, font_color=_color(str(color)), align=align, bold=str(_weight) == "bold",
        ), value))


def _validate(pages: Sequence[_EditablePage]) -> None:
    for page in pages:
        ids: set[str] = set()
        shapes = {shape.id: shape for shape in page.shapes}
        for shape in page.shapes:
            if shape.id in ids:
                raise ValueError(f"duplicate draw.io id {shape.id} on {page.name}")
            ids.add(shape.id)
            for value in (shape.x, shape.y, shape.w, shape.h):
                if not math.isfinite(float(value)):
                    raise ValueError(f"non-finite geometry for {shape.id}")
            if shape.w < 0 or shape.h < 0:
                raise ValueError(f"negative geometry for {shape.id}")
            if shape.x < 0 or shape.y < 0 or shape.x + shape.w > page.w or shape.y + shape.h > page.h:
                raise ValueError(f"vertex outside {page.name}: {shape.id} at {(shape.x, shape.y, shape.w, shape.h)}")
        for edge in page.edges:
            if not isinstance(edge, _EditableEdge):
                raise TypeError(f"unexpected edge type on {page.name}: {edge!r}")
            if edge.id in ids:
                raise ValueError(f"duplicate draw.io id {edge.id} on {page.name}")
            ids.add(edge.id)
            if edge.src and edge.src not in shapes:
                raise ValueError(f"edge {edge.id} references missing source {edge.src}")
            if edge.tgt and edge.tgt not in shapes:
                raise ValueError(f"edge {edge.id} references missing target {edge.tgt}")
            if len(edge.pts) < 2:
                raise ValueError(f"edge {edge.id} has fewer than two points")
            for a, b in zip(edge.pts, edge.pts[1:]):
                if a == b:
                    raise ValueError(f"edge {edge.id} has a zero-length segment at {a}")
                if a[0] != b[0] and a[1] != b[1]:
                    raise ValueError(f"edge {edge.id} is not orthogonal: {a}->{b}")
            for point in edge.pts:
                if not all(math.isfinite(float(v)) for v in point):
                    raise ValueError(f"edge {edge.id} has non-finite point {point}")
                if not (0 <= point[0] <= page.w and 0 <= point[1] <= page.h):
                    raise ValueError(f"edge {edge.id} point outside {page.name}: {point}")
            if edge.src and not _on_or_inside(edge.pts[0], _bounds(shapes[edge.src]), eps=1.0):
                raise ValueError(f"edge {edge.id} starts outside {edge.src}: {edge.pts[0]}")
            if edge.tgt and not _on_or_inside(edge.pts[-1], _bounds(shapes[edge.tgt]), eps=1.0):
                raise ValueError(f"edge {edge.id} ends outside {edge.tgt}: {edge.pts[-1]}")


def write_editable(figures: Sequence[Any] | Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Write one editable draw.io page for each SVG-kit ``Figure``.

    ``figures`` may be a sequence or a mapping.  The figure's own slug becomes
    the page name and every node/edge keeps its source coordinates and colors.
    The output is validated before writing; errors indicate a malformed figure
    record or an edge that no longer terminates on its declared node.
    """

    if isinstance(figures, Mapping):
        figures = list(figures.values())
    else:
        figures = list(figures)
    pages: list[_EditablePage] = []
    for figure in figures:
        page = _EditablePage(str(figure.slug), int(figure.w), int(figure.h))
        used: set[str] = set()
        record_ids = {
            str(values[0])
            for kind, values in getattr(figure, "records", [])
            if kind in {"rect", "cube", "circle", "image"} and values
        }
        _parse_background(figure, page, used, record_ids)
        _append_records(figure, page, used)
        for index, edge in enumerate(getattr(figure, "edges", [])):
            record = _edge_from_record(index, edge)
            # Figure.edge IDs are implicit; make references explicit and avoid
            # collisions with semantic node IDs.
            record.id = _fresh_id(record.id, used)
            page.edges.append(record)
        _validate([page])
        pages.append(page)
    _validate(pages)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_drawio(pages, output)
    return {
        "path": str(output),
        "pages": len(pages),
        "vertices": sum(len(page.shapes) for page in pages),
        "edges": sum(len(page.edges) for page in pages),
        "approximation_warning": (
            "Native draw.io shape=cube stencils are editable approximations; "
            "the SVG/PDF exports retain the exact three-face projection."
        ),
    }


__all__ = ["write_editable"]

