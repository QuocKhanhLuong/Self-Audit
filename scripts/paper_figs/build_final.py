#!/usr/bin/env python3
"""Build and check the four SVG-native SASNet paper figures.

Requirements: Python with matplotlib, Pillow and PyMuPDF; rsvg-convert on PATH.
No model import, patient-volume access, checkpoint loading or inference occurs.

    python scripts/paper_figs/build_final.py
    python scripts/paper_figs/build_final.py --review-dir /tmp/sasnet-review

The optional review directory receives independent PDF rasterizations at
144 dpi and 96 dpi (seven-inch display width), for manual visual inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from check_final_figures import check_exports, check_geometry
from fig1_overview import build as fig1
from fig2_physzssm import build as fig2
from fig3_slicealign import build as fig3
from fig4_shiftrecovery import build as fig4

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/paper-figures-20261124"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_all(slices):
    meta = json.loads((slices / "META.json").read_text())
    if meta["patient"] != "patient011":
        raise ValueError("This delivery must preserve the existing patient011 illustration")
    if meta["shift_max_px"] != 8 or meta["slice_indices"] != [0,2,4,6,8]:
        raise ValueError("Unexpected figure-asset identity or shift convention")
    # Fail closed: never replace missing real PNGs with placeholders.
    for kind in ("in", "out", "shift"):
        for k in range(5):
            if not (slices / f"{kind}_{k}.png").is_file():
                raise FileNotFoundError(slices / f"{kind}_{k}.png")
    return [fig1(slices), fig2(), fig3(), fig4(slices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", type=Path, default=REPORT / "slices")
    parser.add_argument("--out", type=Path, default=REPORT / "final")
    parser.add_argument("--review-dir", type=Path)
    args = parser.parse_args()
    renderer = shutil.which("rsvg-convert")
    if not renderer:
        raise SystemExit("Install librsvg (rsvg-convert) to export vector PDF and 2x PNG.")
    figures = build_all(args.slices)
    audits = [check_geometry(f) for f in figures]
    for f,audit in zip(figures,audits):
        if audit["errors"]:
            raise SystemExit(f.slug + ":\n" + "\n".join(audit["errors"]))
    assets = {p.name:sha256(p) for p in sorted(args.slices.iterdir()) if p.suffix in (".json", ".png")}
    results = []
    for f,audit in zip(figures,audits):
        result = f.export(args.out, renderer)
        result["geometry"] = audit
        result["exports"] = check_exports(f,args.out)
        result["sha256"] = {suffix:sha256(args.out / f"{f.slug}.{suffix}") for suffix in ("svg", "pdf", "png")}
        results.append(result)
        print(f"{f.slug}: geometry clean; vector PDF; 2x PNG {f.w*2}×{f.eff_h*2}; minimum {f.min_font/2:g} pt")
        if args.review_dir:
            import pymupdf as fitz
            args.review_dir.mkdir(parents=True,exist_ok=True)
            with fitz.open(args.out / f"{f.slug}.pdf") as pdf:
                for dpi in (144,96):
                    pdf[0].get_pixmap(dpi=dpi,alpha=False).save(args.review_dir / f"{f.slug}_pdf_{dpi}dpi.png")
    from editable_drawio import write_editable
    companion = args.out / "sasnet_figures.drawio"
    drawio_result = write_editable(figures,companion)
    # Verify that generation preserved the source raster assets byte for byte.
    assert assets == {p.name:sha256(p) for p in sorted(args.slices.iterdir()) if p.suffix in (".json", ".png")}
    scripts = sorted(Path(__file__).parent.glob("*.py"))
    source_files = [ROOT / "self_audit/candidate_pg/slicessm_student.py", ROOT / "self_audit/candidate_pg/aniso_student.py"]
    receipt = {"schema":"sasnet.publication_figures.v1", "renderer":"librsvg rsvg-convert",
               "print_width_inches":7, "png_scale_from_viewbox":2,
               "projection_depth_vector":[16,-12], "patient":"patient011",
               "source_assets_sha256":assets,
               "architecture_source_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in source_files},
               "scripts_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in scripts},
               "figures":results, "drawio":drawio_result,
               "visual_review":"See REPORT.md for the manual raster inspection; automated checks do not establish visual quality."}
    (args.out / "BUILD_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
