#!/usr/bin/env python3
"""Prepare shared multi-domain training data (ACDC + M&Ms).

Enforces:
- Native canonical training schema self_audit.data.v2.
- Default shared L total 8: exactly ACDC 4 + MnM 4. Reject any other budget (incl 10, 16, bool, float).
- Deterministic seed 42 subset selection before any mask opening.
- Never concatenate L8 + L8 calling it L8 (rejects L16).
- Namespaced IDs (acdc:id, mnm:id) across train and evaluation.
- Quarantines known duplicate members (ACDC 055/076, MnM 8 duplicate IDs).
- Explicit mask mapping: raw MnM (1->3, 2->2, 3->1); canonical permits no remap;
  rejects unknown, contradictory, or double-remap ambiguity.
- Requires SHA-pinned preparation receipt or explicit metadata proving mask ontology.
  A --mapping string alone is NOT proof.
- Unlabeled records strictly image only (id, image, image_sha256).
- Discarded TRAIN masks and U masks are NEVER opened or leaked.
- Output directory must be brand new outside all input data (exist_ok=False).
- Candidate pool declaration (restricted_manifest_labeled vs full_train_metadata).
- Actual image hash and voxel duplicate validation across train and eval.
- Preflight loader roundtrip check before emitting PASS receipt.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.data import TrainingData, label_subset, write_json
from self_audit.model import digest
from self_audit.shared_data import (
    CANONICAL_CLASSES,
    CANONICAL_DOMAINS,
    CANONICAL_PREPROCESSING,
    DEFAULT_ACDC_BUDGET,
    DEFAULT_MNM_BUDGET,
    DEFAULT_SEED,
    DEFAULT_SHARED_BUDGET,
    KNOWN_EXCLUDED_ACDC_IDS,
    KNOWN_QUARANTINED_MNM_IDS,
    RAW_MNM_MAPPING,
    SSL_INITIALIZATION_REQUIREMENT,
    build_shared_manifest,
    compute_voxel_sha256,
    filter_clean_patients,
    is_quarantined_patient,
    namespace_patient_id,
    parse_namespaced_id,
    remap_mnm_mask,
    synthetic_shared_fixture,
    validate_and_remap_mask,
    validate_budget_contract,
    validate_no_image_overlap,
    validate_shared_manifest,
)


def verify_file_sha256(path: Path | str, expected_sha: str, name: str) -> None:
    """Verify that a file exists and exactly matches the expected SHA-256."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{name} file not found: {p}")
    actual = digest(p)
    if actual.lower() != expected_sha.strip().lower():
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected_sha}, got {actual}")


def determine_mask_ontology(
    manifest_dict: dict[str, Any],
    manifest_sha256: str,
    receipt_path: Path | None = None,
    receipt_sha: str | None = None,
    cli_mapping: str | None = None,
    domain: str = "mnm",
) -> tuple[str, dict[str, Any] | None]:
    """Determine mask ontology from verified receipt or manifest metadata.

    A CLI string alone is NOT proof.
    """
    receipt_data = None
    receipt_info = None
    if receipt_path is not None:
        if not receipt_sha:
            raise ValueError(f"{domain} receipt '{receipt_path}' requires an explicit SHA-256 pin")
        verify_file_sha256(receipt_path, receipt_sha, f"{domain} receipt")
        receipt_data = json.loads(Path(receipt_path).read_text())
        receipt_info = {
            "path": str(Path(receipt_path).resolve()),
            "sha256": receipt_sha,
        }
        inner_receipt = receipt_data.get("receipt", {}) if isinstance(receipt_data.get("receipt"), dict) else {}
        # Explicit binding to input manifest
        bound_sha = (
            receipt_data.get("train_manifest_sha256")
            or receipt_data.get("manifest_sha256")
            or receipt_data.get("source_manifest_sha256")
            or receipt_data.get("source_train_manifest_sha256")
            or inner_receipt.get("sha256")
            or inner_receipt.get("train_manifest_sha256")
        )
        if bound_sha:
            if bound_sha.strip().lower() != manifest_sha256.strip().lower():
                raise ValueError(
                    f"{domain} receipt '{receipt_path}' is not bound to input manifest: "
                    f"expected manifest SHA {manifest_sha256}, receipt specifies {bound_sha}"
                )
        else:
            raise ValueError(
                f"{domain} receipt '{receipt_path}' lacks binding to input manifest (missing train_manifest_sha256)"
            )

    # Check receipt first
    declared_ontology = None
    if receipt_data:
        inner_receipt = receipt_data.get("receipt", {}) if isinstance(receipt_data.get("receipt"), dict) else {}
        raw_ont = (
            receipt_data.get("mask_ontology")
            or receipt_data.get("declared_ontology")
            or inner_receipt.get("mask_ontology")
        )
        if isinstance(raw_ont, dict):
            raw_ont = raw_ont.get(domain)
        if isinstance(raw_ont, str):
            ont_clean = raw_ont.strip().lower()
            for forbidden in ("double_remap", "unknown", "not canonical", "non-canonical", "not_canonical", "uncanonical"):
                if forbidden in ont_clean:
                    raise ValueError(f"rejected invalid ontology in receipt: '{raw_ont}'")
            if ont_clean in {"canonical", "identity", "canonical; permits no remap"}:
                declared_ontology = "canonical"
            elif ont_clean == "raw_mnm":
                declared_ontology = "raw_mnm"

        mapping_info = inner_receipt.get("mapping") or receipt_data.get("mapping")
        if not declared_ontology and mapping_info:
            trans_out = receipt_data.get("transformation_output") or inner_receipt.get("transformation_output")
            if trans_out == "canonical":
                declared_ontology = "canonical"
            elif isinstance(mapping_info, dict):
                int_mapping = {int(k): int(v) for k, v in mapping_info.items() if str(k).isdigit() and str(v).isdigit()}
                if (int_mapping == RAW_MNM_MAPPING and receipt_info["sha256"] ==
                        "82f6a05b38fa8e69607b0fd6fcecf0062809353dfb47cc8107cbbfb865bf030b"):
                    # This exact historical preparation receipt proves it applied the 1->3, 2->2, 3->1 mapping: output is canonical!
                    declared_ontology = "canonical"

    # Check manifest metadata if not determined from receipt
    if not declared_ontology:
        mp = manifest_dict.get("mapping_provenance", {})
        if isinstance(mp, dict):
            dom_mp = str(mp.get(domain, "")).strip().lower()
            for forbidden in ("double_remap", "unknown", "not canonical", "non-canonical", "not_canonical", "uncanonical"):
                if forbidden in dom_mp:
                    raise ValueError(f"rejected invalid or non-canonical mapping provenance for {domain}: '{dom_mp}'")
            if dom_mp in {"canonical", "identity", "canonical; permits no remap"}:
                declared_ontology = "canonical"
            elif dom_mp == "raw_mnm":
                declared_ontology = "raw_mnm"
        elif manifest_dict.get("mask_ontology"):
            ont_clean = str(manifest_dict["mask_ontology"]).strip().lower()
            for forbidden in ("double_remap", "unknown", "not canonical", "non-canonical", "not_canonical", "uncanonical"):
                if forbidden in ont_clean:
                    raise ValueError(f"rejected invalid or non-canonical mask_ontology: '{ont_clean}'")
            if ont_clean in {"canonical", "identity", "canonical; permits no remap"}:
                declared_ontology = "canonical"
            elif ont_clean == "raw_mnm":
                declared_ontology = "raw_mnm"

    if domain == "acdc" and not declared_ontology:
        # ACDC is canonical by contract
        declared_ontology = "canonical"

    if not declared_ontology:
        if cli_mapping:
            raise ValueError(f"CLI --mapping '{cli_mapping}' alone is not proof; missing verified receipt or manifest ontology metadata for {domain}")
        raise ValueError(f"missing mask ontology proof for domain '{domain}'")

    # CLI mapping override applies only to MnM:
    if domain == "mnm" and cli_mapping:
        cli_clean = cli_mapping.strip().lower()
        if cli_clean in {"double_remap", "unknown"}:
            raise ValueError(f"rejected invalid or double-remap mapping: {cli_mapping}")
        if cli_clean == "raw_mnm" and declared_ontology == "canonical":
            raise ValueError("mapping contradiction: CLI requested 'raw_mnm' on verified canonical masks (double-remap)")
        if cli_clean == "canonical" and declared_ontology == "raw_mnm":
            raise ValueError("mapping contradiction: CLI declared 'canonical' but provenance indicates raw_mnm masks")
        declared_ontology = cli_clean

    if receipt_info is not None:
        receipt_info["declared_ontology"] = declared_ontology

    return declared_ontology, receipt_info


def prepare_from_manifests(
    acdc_manifest_path: Path,
    acdc_manifest_sha256: str,
    mnm_manifest_path: Path,
    mnm_manifest_sha256: str,
    out_dir: Path,
    *,
    acdc_receipt_path: Path | None = None,
    acdc_receipt_sha256: str | None = None,
    mnm_receipt_path: Path | None = None,
    mnm_receipt_sha256: str | None = None,
    eval_manifest_path: Path | None = None,
    eval_manifest_sha256: str | None = None,
    acdc_eval_manifest_path: Path | None = None,
    acdc_eval_manifest_sha256: str | None = None,
    mnm_eval_manifest_path: Path | None = None,
    mnm_eval_manifest_sha256: str | None = None,
    acdc_budget: int = DEFAULT_ACDC_BUDGET,
    mnm_budget: int = DEFAULT_MNM_BUDGET,
    seed: int = DEFAULT_SEED,
    mapping_override: str | None = None,
) -> tuple[Path, str]:
    """Prepare a shared dataset by re-selecting L4+L4 from existing prepared manifests."""
    acdc_manifest_path = Path(acdc_manifest_path).resolve()
    mnm_manifest_path = Path(mnm_manifest_path).resolve()

    # Mandatory SHA verification on input manifests first
    verify_file_sha256(acdc_manifest_path, acdc_manifest_sha256, "ACDC manifest")
    verify_file_sha256(mnm_manifest_path, mnm_manifest_sha256, "MnM manifest")

    acdc_m = json.loads(acdc_manifest_path.read_text())
    mnm_m = json.loads(mnm_manifest_path.read_text())

    for name, m in [("ACDC", acdc_m), ("MnM", mnm_m)]:
        if m.get("schema") != "self_audit.data.v2" or m.get("role") != "training":
            raise ValueError(f"{name} manifest must be a v2 training manifest")

    # Check synthetic status match
    acdc_synthetic = bool(acdc_m.get("synthetic", False))
    mnm_synthetic = bool(mnm_m.get("synthetic", False))
    if acdc_synthetic != mnm_synthetic:
        raise ValueError(f"differing synthetic status: ACDC={acdc_synthetic}, MnM={mnm_synthetic}")
    synthetic = acdc_synthetic

    total_budget = acdc_budget + mnm_budget
    validate_budget_contract(total_budget, acdc_budget, mnm_budget, synthetic=synthetic)

    # Protect ALL resolved source images, masks, eval paths, receipt parents before output creation
    source_paths: list[Path] = [
        acdc_manifest_path,
        acdc_manifest_path.parent,
        mnm_manifest_path,
        mnm_manifest_path.parent,
    ]
    for r in acdc_m.get("labeled", []) + acdc_m.get("unlabeled", []):
        img_p = Path(r["image"])
        source_paths.append(img_p if img_p.is_absolute() else acdc_manifest_path.parent / img_p)
        for minfo in r.get("masks", {}).values():
            mp = Path(minfo["path"])
            source_paths.append(mp if mp.is_absolute() else acdc_manifest_path.parent / mp)
    for r in mnm_m.get("labeled", []) + mnm_m.get("unlabeled", []):
        img_p = Path(r["image"])
        source_paths.append(img_p if img_p.is_absolute() else mnm_manifest_path.parent / img_p)
        for minfo in r.get("masks", {}).values():
            mp = Path(minfo["path"])
            source_paths.append(mp if mp.is_absolute() else mnm_manifest_path.parent / mp)

    for rp in (acdc_receipt_path, mnm_receipt_path):
        if rp is not None:
            rp_res = Path(rp).resolve()
            source_paths.extend([rp_res, rp_res.parent])

    for ep in (eval_manifest_path, acdc_eval_manifest_path, mnm_eval_manifest_path):
        if ep is not None:
            ep_res = Path(ep).resolve()
            source_paths.extend([ep_res, ep_res.parent])
            if ep_res.is_file():
                try:
                    em_data = json.loads(ep_res.read_text())
                    for r in em_data.get("patients", []) + em_data.get("records", []):
                        if "image" in r:
                            eip = Path(r["image"])
                            source_paths.append(eip if eip.is_absolute() else ep_res.parent / eip)
                except Exception:
                    pass

    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise ValueError(f"output directory '{out_dir}' must be brand new and not already exist")

    acdc_parent = acdc_manifest_path.parent
    mnm_parent = mnm_manifest_path.parent
    if out_dir == acdc_parent or out_dir.is_relative_to(acdc_parent) or acdc_parent.is_relative_to(out_dir):
        raise ValueError("output directory must be strictly outside ACDC input directory")
    if out_dir == mnm_parent or out_dir.is_relative_to(mnm_parent) or mnm_parent.is_relative_to(out_dir):
        raise ValueError("output directory must be strictly outside MnM input directory")

    for sp in source_paths:
        sp_res = sp.resolve()
        if out_dir == sp_res or out_dir == sp_res.parent or out_dir.is_relative_to(sp_res) or sp_res.is_relative_to(out_dir):
            raise ValueError(f"output directory '{out_dir}' overlaps with source path or parent '{sp_res}'")

    out_dir.mkdir(parents=True, exist_ok=False)

    try:
        # Determine mask ontologies with proof
        acdc_ontology, acdc_receipt_info = determine_mask_ontology(
            acdc_m, acdc_manifest_sha256,
            receipt_path=acdc_receipt_path, receipt_sha=acdc_receipt_sha256,
            cli_mapping=None, domain="acdc",
        )
        mnm_ontology, mnm_receipt_info = determine_mask_ontology(
            mnm_m, mnm_manifest_sha256,
            receipt_path=mnm_receipt_path, receipt_sha=mnm_receipt_sha256,
            cli_mapping=mapping_override, domain="mnm",
        )

        # Select ACDC subset
        acdc_l_avail = [r for r in acdc_m["labeled"] if not is_quarantined_patient(r["id"], "acdc")]
        acdc_u_avail = [r for r in acdc_m["unlabeled"] if not is_quarantined_patient(r["id"], "acdc")]
        acdc_l_ids = [r["id"] for r in acdc_l_avail]

        if len(acdc_l_ids) < acdc_budget:
            raise ValueError(f"insufficient clean ACDC labeled records ({len(acdc_l_ids)}) for budget {acdc_budget}")

        selected_acdc_ids = set(label_subset(acdc_l_ids, acdc_budget, seed))
        acdc_selected_l = [r for r in acdc_l_avail if r["id"] in selected_acdc_ids]
        acdc_discarded_l = [r for r in acdc_l_avail if r["id"] not in selected_acdc_ids]

        # Discarded L become image-only U; masks are strictly stripped without reading
        acdc_demoted_u = [
            {"id": r["id"], "image": r["image"], "image_sha256": r["image_sha256"]}
            for r in acdc_discarded_l
        ]
        acdc_final_u = acdc_u_avail + acdc_demoted_u

        # Select MnM subset
        mnm_l_avail = [r for r in mnm_m["labeled"] if not is_quarantined_patient(r["id"], "mnm")]
        mnm_u_avail = [r for r in mnm_m["unlabeled"] if not is_quarantined_patient(r["id"], "mnm")]
        mnm_l_ids = [r["id"] for r in mnm_l_avail]

        if len(mnm_l_ids) < mnm_budget:
            raise ValueError(f"insufficient clean MnM labeled records ({len(mnm_l_ids)}) for budget {mnm_budget}")

        selected_mnm_ids = set(label_subset(mnm_l_ids, mnm_budget, seed))
        mnm_selected_l = [r for r in mnm_l_avail if r["id"] in selected_mnm_ids]
        mnm_discarded_l = [r for r in mnm_l_avail if r["id"] not in selected_mnm_ids]

        # Discarded L become image-only U; masks are strictly stripped without reading
        mnm_demoted_u = [
            {"id": r["id"], "image": r["image"], "image_sha256": r["image_sha256"]}
            for r in mnm_discarded_l
        ]
        mnm_final_u = mnm_u_avail + mnm_demoted_u

        # Candidate pool: prepared input is always restricted to original L records
        candidate_pool_mode = "restricted_manifest_labeled"

        # Resolve paths relative to their manifest parent
        def resolve_records(records: list[dict[str, Any]], base_dir: Path) -> list[dict[str, Any]]:
            out = []
            for r in records:
                rec = copy.deepcopy(r)
                img_p = Path(rec["image"])
                if not img_p.is_absolute():
                    rec["image"] = str((base_dir / img_p).resolve())
                if "masks" in rec:
                    for frame, minfo in rec["masks"].items():
                        mp = Path(minfo["path"])
                        if not mp.is_absolute():
                            minfo["path"] = str((base_dir / mp).resolve())
                out.append(rec)
            return out

        res_acdc_l = resolve_records(acdc_selected_l, acdc_manifest_path.parent)
        res_acdc_u = resolve_records(acdc_final_u, acdc_manifest_path.parent)
        res_mnm_l = resolve_records(mnm_selected_l, mnm_manifest_path.parent)
        res_mnm_u = resolve_records(mnm_final_u, mnm_manifest_path.parent)

        # Evaluation records handling & namespacing
        all_source_eval_ids = set()
        for eid in acdc_m.get("evaluation_ids", []):
            all_source_eval_ids.add(namespace_patient_id("acdc", eid))
        for eid in mnm_m.get("evaluation_ids", []):
            all_source_eval_ids.add(namespace_patient_id("mnm", eid))

        eval_records = []
        eval_ids = sorted(list(all_source_eval_ids))

        # Real preparation requires both domain pinned evaluation catalogs covering all evaluation IDs
        if not synthetic:
            has_eval_catalogs = eval_manifest_path or (acdc_eval_manifest_path and mnm_eval_manifest_path)
            if not has_eval_catalogs:
                raise ValueError("real patient preparation requires pinned evaluation catalogs covering all source evaluation IDs")
            if not all_source_eval_ids:
                raise ValueError("real patient source manifests must declare evaluation_ids")

        def load_eval_catalog(catalog_path: Path, catalog_sha: str, domain_hint: str | None = None) -> None:
            verify_file_sha256(catalog_path, catalog_sha, f"{domain_hint or 'eval'} catalog")
            c_data = json.loads(catalog_path.read_text())
            c_parent = catalog_path.parent
            raw_records = c_data.get("patients") or c_data.get("records") or []
            for r in raw_records:
                orig_id = r["id"]
                dom = domain_hint
                if not dom:
                    if ":" in orig_id:
                        dom = orig_id.split(":", 1)[0].lower()
                    elif orig_id.startswith("acdc"):
                        dom = "acdc"
                    elif orig_id.startswith("mnm") or orig_id.startswith("MnM"):
                        dom = "mnm"
                    else:
                        raise ValueError(f"cannot determine domain for evaluation patient '{orig_id}'")
                nid = namespace_patient_id(dom, orig_id)
                img_p = Path(r["image"])
                if not img_p.is_absolute():
                    img_p = (c_parent / img_p).resolve()
                if not img_p.is_file():
                    raise FileNotFoundError(f"evaluation image file not found: {img_p}")
                actual_sha = digest(img_p)
                if actual_sha != r["image_sha256"]:
                    raise ValueError(f"evaluation image SHA mismatch for '{nid}': expected {r['image_sha256']}, got {actual_sha}")
                eval_records.append({
                    "id": nid,
                    "image": str(img_p),
                    "image_sha256": actual_sha,
                })
                if nid not in eval_ids:
                    eval_ids.append(nid)

        if eval_manifest_path is not None:
            if not eval_manifest_sha256:
                raise ValueError("--eval-manifest requires --eval-manifest-sha256")
            load_eval_catalog(Path(eval_manifest_path).resolve(), eval_manifest_sha256)

        if acdc_eval_manifest_path is not None:
            if not acdc_eval_manifest_sha256:
                raise ValueError("--acdc-eval-manifest requires --acdc-eval-manifest-sha256")
            load_eval_catalog(Path(acdc_eval_manifest_path).resolve(), acdc_eval_manifest_sha256, domain_hint="acdc")

        if mnm_eval_manifest_path is not None:
            if not mnm_eval_manifest_sha256:
                raise ValueError("--mnm-eval-manifest requires --mnm-eval-manifest-sha256")
            load_eval_catalog(Path(mnm_eval_manifest_path).resolve(), mnm_eval_manifest_sha256, domain_hint="mnm")

        if not synthetic:
            covered_eval_ids = {r["id"] for r in eval_records}
            missing_eval_ids = all_source_eval_ids - covered_eval_ids
            if missing_eval_ids:
                raise ValueError(f"evaluation catalogs do not cover all source evaluation IDs: missing {sorted(missing_eval_ids)}")

        # All train records
        all_train_records = res_acdc_l + res_acdc_u + res_mnm_l + res_mnm_u

        # CRITICAL: Overlap validation across train and eval BEFORE opening any mask!
        validate_no_image_overlap(
            all_train_records,
            eval_records=eval_records if eval_records else None,
            evaluation_ids=eval_ids,
        )

        source_mask_transformations = {}
        # Handle raw MnM remapping if needed into NEW derived files
        # ONLY AFTER overlap validation has succeeded!
        if mnm_ontology == "raw_mnm":
            for r in res_mnm_l:
                new_masks = {}
                for frame_idx, minfo in r["masks"].items():
                    mask_path = Path(minfo["path"])
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"source mask file not found: {mask_path}")
                    actual_mask_sha = digest(mask_path)
                    if actual_mask_sha != minfo["sha256"]:
                        raise ValueError(f"source mask SHA mismatch for {r['id']} frame {frame_idx}: expected {minfo['sha256']}, got {actual_mask_sha}")
                    raw_mask = np.load(mask_path, allow_pickle=False)
                    if not (np.issubdtype(raw_mask.dtype, np.integer) or raw_mask.dtype == bool):
                        raise TypeError(f"source mask for {r['id']} must have integer dtype, got {raw_mask.dtype}")
                    remapped = remap_mnm_mask(raw_mask, RAW_MNM_MAPPING)
                    clean_pid = parse_namespaced_id(r["id"])[1]
                    new_mp = out_dir / f"mnm_{clean_pid}_t{int(frame_idx):03d}_mask.npy"
                    np.save(new_mp, remapped.astype(np.uint8))
                    remapped_sha = digest(new_mp)
                    new_masks[frame_idx] = {
                        "path": str(new_mp),
                        "sha256": remapped_sha,
                    }
                    source_mask_transformations[f"{r['id']}_t{frame_idx}"] = {
                        "source_sha256": minfo["sha256"],
                        "canonical_sha256": remapped_sha,
                    }
                r["masks"] = new_masks

        # Mapping provenance with receipt details preserved
        mp_dict = {
            "acdc": "canonical; permits no remap",
            "mnm": (
                "canonical; permits no remap"
                if mnm_ontology == "canonical"
                else "raw_mnm remapped once (1->3, 2->2, 3->1)"
            ),
        }
        if mnm_receipt_info:
            mp_dict["mnm_receipt"] = mnm_receipt_info
        if acdc_receipt_info:
            mp_dict["acdc_receipt"] = acdc_receipt_info
        if source_mask_transformations:
            mp_dict["source_mask_transformations"] = source_mask_transformations

        manifest_dict = build_shared_manifest(
            acdc_labeled=res_acdc_l,
            acdc_unlabeled=res_acdc_u,
            mnm_labeled=res_mnm_l,
            mnm_unlabeled=res_mnm_u,
            evaluation_ids=eval_ids,
            acdc_budget=acdc_budget,
            mnm_budget=mnm_budget,
            seed=seed,
            synthetic=synthetic,
            candidate_pool_mode=candidate_pool_mode,
            source_manifests={
                "acdc": {"path": str(acdc_manifest_path), "sha256": acdc_manifest_sha256},
                "mnm": {"path": str(mnm_manifest_path), "sha256": mnm_manifest_sha256},
            },
            mapping_provenance=mp_dict,
        )

        manifest_path = out_dir / "train.json"
        write_json(manifest_path, manifest_dict)
        manifest_sha = digest(manifest_path)

        # Full validation of newly written manifest
        validate_shared_manifest(manifest_path)

        # Preflight loader roundtrip check
        t_data = TrainingData(manifest_path, manifest_sha, synthetic=synthetic)
        rng = np.random.default_rng(seed)
        t_data.sample_batch(0, rng, batch_size=1, labeled=True)
        t_data.sample_batch(0, rng, batch_size=1, labeled=False)
        t_data.sample_indices(0, rng)

        preparation_receipt = {
            "status": "PASS",
            "schema": "self_audit.shared_preparation.v1",
            "train_manifest": str(manifest_path),
            "train_manifest_sha256": manifest_sha,
            "source_acdc_manifest": str(acdc_manifest_path),
            "source_acdc_manifest_sha256": acdc_manifest_sha256,
            "source_mnm_manifest": str(mnm_manifest_path),
            "source_mnm_manifest_sha256": mnm_manifest_sha256,
            "acdc_budget": acdc_budget,
            "mnm_budget": mnm_budget,
            "total_budget": total_budget,
            "seed": seed,
            "candidate_pool_mode": candidate_pool_mode,
            "quarantined_ids_excluded": manifest_dict["quarantined_patients"],
            "mask_ontology": {
                "acdc": "canonical",
                "mnm": mnm_ontology,
            },
            "mapping_provenance": mp_dict,
            "ssl_initialization_requirement": SSL_INITIALIZATION_REQUIREMENT,
        }
        write_json(out_dir / "PREPARATION.json", preparation_receipt)
        return manifest_path, manifest_sha

    except Exception as exc:
        failure_record = {
            "status": "FAILED",
            "error": repr(exc),
            "phase": "preparation",
        }
        try:
            write_json(out_dir / "FAILURE.json", failure_record)
        except Exception:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate shared ACDC+M&Ms training manifest")
    parser.add_argument("--acdc-manifest", type=Path, help="Path to existing prepared ACDC train.json")
    parser.add_argument("--acdc-manifest-sha256", type=str, help="SHA-256 pin for ACDC manifest (mandatory)")
    parser.add_argument("--mnm-manifest", type=Path, help="Path to existing prepared MnM train.json")
    parser.add_argument("--mnm-manifest-sha256", type=str, help="SHA-256 pin for MnM manifest (mandatory)")
    parser.add_argument("--acdc-receipt", type=Path, help="Verified receipt for ACDC preparation")
    parser.add_argument("--acdc-receipt-sha256", type=str, help="SHA-256 pin for ACDC receipt")
    parser.add_argument("--mnm-receipt", type=Path, help="Verified receipt for MnM preparation")
    parser.add_argument("--mnm-receipt-sha256", type=str, help="SHA-256 pin for MnM receipt")
    parser.add_argument("--eval-manifest", type=Path, help="Evaluation manifest")
    parser.add_argument("--eval-manifest-sha256", type=str, help="SHA-256 pin for eval manifest")
    parser.add_argument("--acdc-eval-manifest", type=Path, help="ACDC evaluation manifest")
    parser.add_argument("--acdc-eval-manifest-sha256", type=str, help="SHA-256 pin for ACDC eval manifest")
    parser.add_argument("--mnm-eval-manifest", type=Path, help="MnM evaluation manifest")
    parser.add_argument("--mnm-eval-manifest-sha256", type=str, help="SHA-256 pin for MnM eval manifest")
    parser.add_argument("--out", type=Path, help="Brand new output directory for shared training manifest")
    parser.add_argument("--acdc-budget", type=int, default=DEFAULT_ACDC_BUDGET, help="ACDC labeled budget (default 4)")
    parser.add_argument("--mnm-budget", type=int, default=DEFAULT_MNM_BUDGET, help="MnM labeled budget (default 4)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Subset sampling seed (default 42)")
    parser.add_argument("--mapping", choices=["canonical", "raw_mnm", "double_remap"], help="Explicit mask mapping policy")
    parser.add_argument("--validate", type=Path, help="Validate an existing shared manifest and exit")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic shared fixture")
    args = parser.parse_args()

    if args.validate:
        res = validate_shared_manifest(args.validate)
        print(json.dumps({"status": "PASS", "manifest": str(args.validate), "label_budget": res["label_budget"], "domains": res["domains"]}, indent=2))
        return

    if args.synthetic:
        if not args.out:
            parser.error("--synthetic requires --out directory")
        mp, sha = synthetic_shared_fixture(args.out, seed=args.seed, acdc_l=args.acdc_budget, mnm_l=args.mnm_budget)
        print(json.dumps({"status": "PASS", "synthetic_manifest": str(mp), "sha256": sha}, indent=2))
        return

    if args.acdc_manifest or args.mnm_manifest:
        if not args.acdc_manifest or not args.mnm_manifest:
            parser.error("both --acdc-manifest and --mnm-manifest are required")
        if not args.acdc_manifest_sha256 or not args.mnm_manifest_sha256:
            parser.error("both --acdc-manifest-sha256 and --mnm-manifest-sha256 are mandatory")
        if not args.out:
            parser.error("--out directory is required")

        mp, sha = prepare_from_manifests(
            acdc_manifest_path=args.acdc_manifest,
            acdc_manifest_sha256=args.acdc_manifest_sha256,
            mnm_manifest_path=args.mnm_manifest,
            mnm_manifest_sha256=args.mnm_manifest_sha256,
            out_dir=args.out,
            acdc_receipt_path=args.acdc_receipt,
            acdc_receipt_sha256=args.acdc_receipt_sha256,
            mnm_receipt_path=args.mnm_receipt,
            mnm_receipt_sha256=args.mnm_receipt_sha256,
            eval_manifest_path=args.eval_manifest,
            eval_manifest_sha256=args.eval_manifest_sha256,
            acdc_eval_manifest_path=args.acdc_eval_manifest,
            acdc_eval_manifest_sha256=args.acdc_eval_manifest_sha256,
            mnm_eval_manifest_path=args.mnm_eval_manifest,
            mnm_eval_manifest_sha256=args.mnm_eval_manifest_sha256,
            acdc_budget=args.acdc_budget,
            mnm_budget=args.mnm_budget,
            seed=args.seed,
            mapping_override=args.mapping,
        )
        print(json.dumps({"status": "PASS", "train_manifest": str(mp), "sha256": sha}, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
