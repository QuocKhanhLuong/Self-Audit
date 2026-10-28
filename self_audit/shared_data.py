"""Shared multi-domain data management (ACDC + M&Ms).

Enforces:
- Native canonical training schema self_audit.data.v2.
- Default shared L total 8: exactly ACDC 4 + MnM 4. Reject any other budget (incl 10, 16, bool, float).
- Deterministic seed 42 subset selection before any mask opening.
- Explicit namespacing (acdc:id, mnm:id) preventing cross-domain collisions.
- Known duplicate quarantines excluded (ACDC 055/076, MnM 8 duplicate IDs).
- Explicit mask mapping: raw MnM (1->3, 2->2, 3->1); canonical permits no remap;
  rejects unknown, contradictory, or double-remap ambiguity.
- Unlabeled records strictly image only (id, image, image_sha256).
- Image/voxel overlap validation across train and eval before writing (detects actual voxel duplicates).
- Complete and explicit domain_map and domain_patients metadata.
- Preprocessing provenance (.5/99.5, TZYX, [0, 1], native geometry).
- Clear candidate pool declaration (restricted_manifest_labeled vs full_train_metadata).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .data import TrainingData, crop_native, label_subset, write_json
from .model import digest

# Canonical domain names
CANONICAL_DOMAINS = ("acdc", "mnm")
DEFAULT_SHARED_BUDGET = 8
DEFAULT_ACDC_BUDGET = 4
DEFAULT_MNM_BUDGET = 4
DEFAULT_SEED = 42

# Known duplicates from origin/main reports/automatic_pilot_pro4000/dataset_MnM_verified_preparation.json
KNOWN_QUARANTINED_MNM_IDS = frozenset({
    "A7G0P5",
    "A8C9H8",
    "C5Q2Y5",
    "C8J7L5",
    "C8O0P2",
    "E9L4N2",
    "K3R0Y7",
    "Q0Q1Y4",
})

# ACDC reviewed locked split excluded duplicate patients
KNOWN_EXCLUDED_ACDC_IDS = frozenset({
    "patient055",
    "patient076",
    "055",
    "076",
})

# Explicit raw MnM to canonical ontology mapping:
# Raw MnM: 0=BG, 1=LV cavity, 2=LV myocardium, 3=RV cavity, 255=ignore
# Canonical (ACDC standard): 0=BG, 1=RV cavity, 2=LV myocardium, 3=LV cavity, 255=ignore
RAW_MNM_MAPPING: dict[int, int] = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
    255: 255,
}

CANONICAL_CLASSES = frozenset({0, 1, 2, 3, 255})
SSL_INITIALIZATION_REQUIREMENT = "DINO self-supervised LVD1689M initialization required; no GT pretraining"

CANONICAL_PREPROCESSING: dict[str, str] = {
    "normalization": "per-cine image-only .5/99.5 percentiles",
    "array_shape_order": "TZYX",
    "float_range": "[0, 1]",
    "spatial_resizing": "no_resize; native_geometry_only",
    "orientation": "native",
    "valid_mask_policy": "image_geometry_padding_only_no_labels",
}


def compute_voxel_sha256(data_array: np.ndarray) -> str:
    """Hash the raw underlying voxel buffer, invariant to .npy file metadata/headers."""
    if not isinstance(data_array, np.ndarray):
        raise TypeError("data_array must be a numpy ndarray")
    return hashlib.sha256(np.ascontiguousarray(data_array).tobytes()).hexdigest()


def validate_patient_id(patient_id: str) -> None:
    """Ensure patient ID is safe and does not attempt path traversal."""
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError(f"invalid patient ID: {patient_id}")
    pid = patient_id.strip()
    for forbidden in ("..", "/", "\\", "\0", "\n", "\r", "\t"):
        if forbidden in pid:
            raise ValueError(f"potential path traversal or invalid characters in patient ID: '{patient_id}'")


def validate_relative_path(path_str: str) -> None:
    """Ensure path reference does not perform path traversal with '..'."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError(f"invalid path string: {path_str}")
    p = Path(path_str)
    for part in p.parts:
        if part == "..":
            raise ValueError(f"path traversal detected in path: '{path_str}'")


def namespace_patient_id(domain: str, raw_id: str) -> str:
    """Namespace a patient ID with its canonical domain prefix."""
    domain = str(domain).strip().lower()
    raw_id = str(raw_id).strip()
    if domain not in CANONICAL_DOMAINS:
        raise ValueError(f"domain '{domain}' is not canonical; must be one of {CANONICAL_DOMAINS}")
    if not raw_id:
        raise ValueError("nonempty raw_id required for namespacing")
    if ":" in raw_id:
        existing_domain, rest = raw_id.split(":", 1)
        validate_patient_id(rest)
        if existing_domain.lower() != domain:
            raise ValueError(f"ID '{raw_id}' already namespaced with conflicting domain '{existing_domain}'")
        return f"{domain}:{rest}"
    validate_patient_id(raw_id)
    return f"{domain}:{raw_id}"


def parse_namespaced_id(namespaced_id: str) -> tuple[str, str]:
    """Parse namespaced ID into (domain, raw_id)."""
    namespaced_id = str(namespaced_id).strip()
    if ":" in namespaced_id:
        parts = namespaced_id.split(":", 1)
        dom = parts[0].lower()
        if dom in CANONICAL_DOMAINS:
            validate_patient_id(parts[1])
            return dom, parts[1]
        raise ValueError(f"unknown domain prefix '{dom}' in ID '{namespaced_id}'")
    validate_patient_id(namespaced_id)
    if namespaced_id.startswith("acdc_") or namespaced_id.startswith("acdc-"):
        return "acdc", namespaced_id[5:]
    if namespaced_id.startswith("mnm_") or namespaced_id.startswith("mnm-") or namespaced_id.startswith("MnM_"):
        return "mnm", namespaced_id[4:]
    return "", namespaced_id


def is_quarantined_patient(patient_id: str, domain: str | None = None) -> bool:
    """Check if a patient ID is in the known quarantine/exclusion lists."""
    parsed_domain, raw_id = parse_namespaced_id(patient_id) if ":" in patient_id else ("", patient_id)
    eff_domain = domain.lower() if domain else (parsed_domain or None)

    clean_raw = raw_id
    if clean_raw.startswith("acdc_") or clean_raw.startswith("acdc-"):
        clean_raw = clean_raw[5:]
    elif clean_raw.startswith("mnm_") or clean_raw.startswith("mnm-") or clean_raw.startswith("MnM_"):
        clean_raw = clean_raw[4:]
    num_suffix = clean_raw.replace("patient", "")

    if eff_domain == "mnm" or eff_domain is None:
        if clean_raw in KNOWN_QUARANTINED_MNM_IDS:
            return True

    if eff_domain == "acdc" or eff_domain is None:
        if clean_raw in KNOWN_EXCLUDED_ACDC_IDS or num_suffix in KNOWN_EXCLUDED_ACDC_IDS:
            return True

    return False


def filter_clean_patients(patient_ids: list[str], domain: str) -> tuple[list[str], list[str]]:
    """Partition patient IDs into (clean, quarantined)."""
    clean, quarantined = [], []
    for pid in patient_ids:
        if is_quarantined_patient(pid, domain):
            quarantined.append(pid)
        else:
            clean.append(pid)
    return clean, quarantined


def remap_mnm_mask(mask: np.ndarray, mapping: dict[int, int] = RAW_MNM_MAPPING) -> np.ndarray:
    """Remap raw MnM mask labels to canonical ontology.

    1 (LV) -> 3 (LV canonical)
    2 (MYO) -> 2 (MYO canonical)
    3 (RV) -> 1 (RV canonical)
    0 -> 0, 255 -> 255.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy ndarray")
    if not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == bool):
        raise TypeError(f"mask must have integer dtype, got {mask.dtype}")
    unique_vals = set(np.unique(mask).tolist())
    allowed_keys = set(mapping.keys())
    if not unique_vals.issubset(allowed_keys):
        invalid = unique_vals - allowed_keys
        raise ValueError(f"mask contains values {invalid} not present in mapping {mapping}")

    out = np.zeros_like(mask, dtype=np.uint8)
    for src, dst in mapping.items():
        out[mask == src] = dst
    return out


def validate_and_remap_mask(
    mask: np.ndarray,
    domain: str,
    mapping_mode: str,
    provenance_receipt: dict[str, Any] | None = None,
) -> np.ndarray:
    """Validate mask ontology and apply mapping strictly once.

    mapping_mode:
      - 'raw_mnm': expects raw MnM ontology, maps 1->3, 2->2, 3->1.
      - 'canonical': explicitly canonical; permits NO remap.
      - Any double-remap or ambiguous mode raises ValueError.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy ndarray")
    if not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == bool):
        raise TypeError(f"mask must have integer dtype, got {mask.dtype}")

    domain = domain.lower()
    mapping_mode = mapping_mode.strip().lower()

    if mapping_mode in {"double_remap", "remap_again", "unknown"}:
        raise ValueError(f"rejected ambiguous or double-remap mapping_mode: {mapping_mode}")

    if provenance_receipt is not None:
        declared_mapping = provenance_receipt.get("mapping") or provenance_receipt.get("mask_ontology")
        if mapping_mode == "canonical" and declared_mapping == "raw_mnm":
            raise ValueError("mapping contradiction: mapping_mode 'canonical' conflicts with receipt 'raw_mnm'")
        if mapping_mode == "raw_mnm" and declared_mapping == "canonical":
            raise ValueError("mapping contradiction: mapping_mode 'raw_mnm' conflicts with receipt 'canonical'")

    if mapping_mode in {"canonical", "identity", "none"}:
        unique_vals = set(np.unique(mask).tolist())
        if not unique_vals.issubset(CANONICAL_CLASSES):
            raise ValueError(f"canonical mask contains invalid classes: {unique_vals - CANONICAL_CLASSES}")
        return mask.astype(np.uint8).copy()

    if mapping_mode == "raw_mnm":
        if domain != "mnm":
            raise ValueError(f"raw_mnm mapping requested for non-MnM domain: '{domain}'")
        return remap_mnm_mask(mask, RAW_MNM_MAPPING)

    raise ValueError(f"unknown mask mapping_mode: '{mapping_mode}'; must be 'canonical' or 'raw_mnm'")


def select_shared_train_labels(
    acdc_train_ids: list[str],
    mnm_train_ids: list[str],
    acdc_budget: int = DEFAULT_ACDC_BUDGET,
    mnm_budget: int = DEFAULT_MNM_BUDGET,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    """Deterministic selection of shared labeled patients before any mask opening."""
    acdc_clean, _ = filter_clean_patients(acdc_train_ids, "acdc")
    mnm_clean, _ = filter_clean_patients(mnm_train_ids, "mnm")

    if len(acdc_clean) < acdc_budget:
        raise ValueError(f"eligible ACDC train count ({len(acdc_clean)}) < declared budget ({acdc_budget})")
    if len(mnm_clean) < mnm_budget:
        raise ValueError(f"eligible MnM train count ({len(mnm_clean)}) < declared budget ({mnm_budget})")

    acdc_selected = label_subset(acdc_clean, acdc_budget, seed)
    mnm_selected = label_subset(mnm_clean, mnm_budget, seed)
    return acdc_selected, mnm_selected


def validate_no_image_overlap(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]] | None = None,
    evaluation_ids: list[str] | None = None,
    base_dir: Path | None = None,
) -> None:
    """Validate that train and evaluation sets have no image/voxel overlap and no duplicate train images."""
    train_file_hashes: dict[str, str] = {}
    train_voxel_hashes: dict[str, str] = {}
    train_pids: set[str] = set()

    for r in train_records:
        pid = r["id"]
        validate_patient_id(pid)
        sha = r["image_sha256"]
        if pid in train_pids:
            raise ValueError(f"duplicate patient ID in training set: '{pid}'")
        train_pids.add(pid)
        if sha in train_file_hashes:
            raise ValueError(f"duplicate TRAIN image content across patient IDs: '{pid}' matches '{train_file_hashes[sha]}'")
        train_file_hashes[sha] = pid

        # If image file is accessible, compute and check voxel_sha256
        img_path = Path(r["image"])
        if not img_path.is_absolute() and base_dir is not None:
            img_path = base_dir / img_path
        if img_path.is_file():
            actual_sha = digest(img_path)
            if actual_sha != sha:
                raise ValueError(f"TRAIN image sha256 mismatch for '{pid}': expected {sha}, got {actual_sha}")
            arr = np.load(img_path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 4 or not np.isfinite(arr).all():
                raise ValueError(f"TRAIN image array for '{pid}' must be finite normalized TZYX")
            v_sha = compute_voxel_sha256(arr)
            if v_sha in train_voxel_hashes:
                raise ValueError(f"duplicate TRAIN image voxels across patient IDs: '{pid}' matches '{train_voxel_hashes[v_sha]}'")
            train_voxel_hashes[v_sha] = pid
        elif base_dir is not None or img_path.is_absolute():
            raise FileNotFoundError(f"TRAIN image file not found: {img_path}")

    if evaluation_ids:
        overlap_ids = train_pids & set(evaluation_ids)
        if overlap_ids:
            raise ValueError(f"TRAIN/evaluation patient overlap: {sorted(overlap_ids)}")

        # Check for quarantined IDs in evaluation
        for eid in evaluation_ids:
            validate_patient_id(eid)
            if is_quarantined_patient(eid):
                raise ValueError(f"quarantined patient '{eid}' found in evaluation_ids")

    if eval_records:
        if evaluation_ids:
            eval_pids_provided = {r["id"] for r in eval_records}
            missing_ids = set(evaluation_ids) - eval_pids_provided
            if missing_ids:
                raise ValueError(f"evaluation records missing declared evaluation IDs: {sorted(missing_ids)}")

        for r in eval_records:
            pid = r["id"]
            validate_patient_id(pid)
            if is_quarantined_patient(pid):
                raise ValueError(f"quarantined patient '{pid}' found in evaluation records")
            sha = r["image_sha256"]
            if sha in train_file_hashes:
                raise ValueError(f"TRAIN/evaluation image overlap: eval patient '{pid}' has identical image to train patient '{train_file_hashes[sha]}'")

            img_path = Path(r["image"])
            if not img_path.is_absolute() and base_dir is not None:
                img_path = base_dir / img_path
            if img_path.is_file():
                actual_sha = digest(img_path)
                if actual_sha != sha:
                    raise ValueError(f"evaluation image sha256 mismatch for '{pid}': expected {sha}, got {actual_sha}")
                arr = np.load(img_path, mmap_mode="r", allow_pickle=False)
                if arr.ndim != 4 or not np.isfinite(arr).all():
                    raise ValueError(f"evaluation image array for '{pid}' must be finite normalized TZYX")
                v_sha = compute_voxel_sha256(arr)
                if v_sha in train_voxel_hashes:
                    raise ValueError(f"TRAIN/evaluation voxel overlap: eval patient '{pid}' has identical voxels to train patient '{train_voxel_hashes[v_sha]}'")
            elif base_dir is not None or img_path.is_absolute():
                raise FileNotFoundError(f"evaluation image file not found: {img_path}")


def validate_budget_contract(
    label_budget: Any,
    acdc_budget: Any,
    mnm_budget: Any,
    synthetic: bool = False,
) -> None:
    """Enforce strict integer typing and patient 4+4=8 budget contract."""
    if type(label_budget) is not int or isinstance(label_budget, bool):
        raise ValueError(f"label_budget must be int, got {type(label_budget).__name__}")
    if type(acdc_budget) is not int or isinstance(acdc_budget, bool):
        raise ValueError(f"acdc_budget must be int, got {type(acdc_budget).__name__}")
    if type(mnm_budget) is not int or isinstance(mnm_budget, bool):
        raise ValueError(f"mnm_budget must be int, got {type(mnm_budget).__name__}")

    if not synthetic:
        if label_budget != DEFAULT_SHARED_BUDGET or acdc_budget != DEFAULT_ACDC_BUDGET or mnm_budget != DEFAULT_MNM_BUDGET:
            raise ValueError(f"real patient shared budget must be exactly 4+4=8, reject any other budget (got acdc={acdc_budget}, mnm={mnm_budget}, total={label_budget})")
    else:
        if acdc_budget + mnm_budget != label_budget:
            raise ValueError(f"acdc_budget ({acdc_budget}) + mnm_budget ({mnm_budget}) != label_budget ({label_budget})")


def build_shared_manifest(
    *,
    acdc_labeled: list[dict[str, Any]],
    acdc_unlabeled: list[dict[str, Any]],
    mnm_labeled: list[dict[str, Any]],
    mnm_unlabeled: list[dict[str, Any]],
    evaluation_ids: list[str] | None = None,
    acdc_budget: int = DEFAULT_ACDC_BUDGET,
    mnm_budget: int = DEFAULT_MNM_BUDGET,
    seed: int = DEFAULT_SEED,
    synthetic: bool = False,
    candidate_pool_mode: str = "restricted_manifest_labeled",
    mapping_provenance: dict[str, Any] | None = None,
    quarantined_ids: list[str] | None = None,
    source_manifests: dict[str, dict[str, str]] | None = None,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a canonical self_audit.data.v2 shared training manifest."""
    total_budget = acdc_budget + mnm_budget
    validate_budget_contract(total_budget, acdc_budget, mnm_budget, synthetic=synthetic)

    if len(acdc_labeled) != acdc_budget:
        raise ValueError(f"ACDC labeled count ({len(acdc_labeled)}) does not match declared budget ({acdc_budget})")
    if len(mnm_labeled) != mnm_budget:
        raise ValueError(f"MnM labeled count ({len(mnm_labeled)}) does not match declared budget ({mnm_budget})")

    # Standardize records to namespaced IDs
    def namespace_records(records: list[dict[str, Any]], domain: str, role: str) -> list[dict[str, Any]]:
        out = []
        for r in records:
            orig_id = r["id"]
            nid = namespace_patient_id(domain, orig_id)
            if is_quarantined_patient(orig_id, domain):
                raise ValueError(f"quarantined patient '{orig_id}' cannot be included in active {domain} {role}")
            if role == "unlabeled":
                if set(r.keys()) != {"id", "image", "image_sha256"}:
                    raise ValueError(f"unlabeled record for '{orig_id}' must contain image only, never reference paths")
                rec = {"id": nid, "image": r["image"], "image_sha256": r["image_sha256"]}
            else:
                if set(r.keys()) != {"id", "image", "image_sha256", "masks"} or not r["masks"]:
                    raise ValueError(f"invalid labeled record for '{orig_id}'")
                rec = {"id": nid, "image": r["image"], "image_sha256": r["image_sha256"], "masks": copy.deepcopy(r["masks"])}
            out.append(rec)
        return out

    ns_acdc_l = namespace_records(acdc_labeled, "acdc", "labeled")
    ns_acdc_u = namespace_records(acdc_unlabeled, "acdc", "unlabeled")
    ns_mnm_l = namespace_records(mnm_labeled, "mnm", "labeled")
    ns_mnm_u = namespace_records(mnm_unlabeled, "mnm", "unlabeled")

    all_labeled = ns_acdc_l + ns_mnm_l
    all_unlabeled = ns_acdc_u + ns_mnm_u

    # Evaluation IDs must be namespaced
    eval_ids = []
    for eid in (evaluation_ids or []):
        eid_str = str(eid).strip()
        if ":" not in eid_str:
            raise ValueError(f"evaluation ID '{eid_str}' must be namespaced with 'acdc:' or 'mnm:'")
        edom = eid_str.split(":", 1)[0].lower()
        if edom not in CANONICAL_DOMAINS:
            raise ValueError(f"evaluation ID '{eid_str}' has invalid domain prefix '{edom}'")
        if is_quarantined_patient(eid_str):
            raise ValueError(f"quarantined patient '{eid_str}' cannot be included in evaluation_ids")
        eval_ids.append(eid_str)

    # Overlap validation
    validate_no_image_overlap(all_labeled + all_unlabeled, evaluation_ids=eval_ids, base_dir=base_dir)

    # Candidate pool declaration
    is_restricted = candidate_pool_mode != "full_train_metadata"
    candidate_pool_info = {
        "mode": candidate_pool_mode,
        "restricted_candidate_pool": is_restricted,
        "sampling_basis": (
            "reselected_from_existing_prepared_masks; not_full_train_sampling"
            if is_restricted else
            "deterministic_seed42_from_eligible_train_metadata_before_mask_opening"
        ),
    }

    # Quarantines
    declared_quarantines = list(quarantined_ids or [])
    for qid in sorted(KNOWN_QUARANTINED_MNM_IDS):
        nid = f"mnm:{qid}"
        if nid not in declared_quarantines:
            declared_quarantines.append(nid)
    for qid in sorted(KNOWN_EXCLUDED_ACDC_IDS):
        nid = f"acdc:{qid}"
        if nid not in declared_quarantines:
            declared_quarantines.append(nid)

    # Build domain_map
    domain_map = {}
    for r in ns_acdc_l + ns_acdc_u:
        domain_map[r["id"]] = "acdc"
    for r in ns_mnm_l + ns_mnm_u:
        domain_map[r["id"]] = "mnm"

    manifest: dict[str, Any] = {
        "schema": "self_audit.data.v2",
        "role": "training",
        "shared": True,
        "synthetic": bool(synthetic),
        "label_budget": total_budget,
        "label_seed": seed,
        "domains": ["acdc", "mnm"],
        "domain_budgets": {"acdc": acdc_budget, "mnm": mnm_budget},
        "domain_patients": {
            "acdc": [r["id"] for r in ns_acdc_l + ns_acdc_u],
            "mnm": [r["id"] for r in ns_mnm_l + ns_mnm_u],
        },
        "domain_map": domain_map,
        "labeled": all_labeled,
        "unlabeled": all_unlabeled,
        "evaluation_ids": eval_ids,
        "candidate_pool": candidate_pool_info,
        "quarantined_patients": sorted(declared_quarantines),
        "ssl_initialization_requirement": SSL_INITIALIZATION_REQUIREMENT,
        "preprocessing": copy.deepcopy(CANONICAL_PREPROCESSING),
        "lineage": {
            "acdc_labeled_count": acdc_budget,
            "mnm_labeled_count": mnm_budget,
            "exact_lineage": f"{acdc_budget}+{mnm_budget}",
            "seed": seed,
        },
        "mapping_provenance": mapping_provenance or {
            "acdc": "canonical; permits no remap",
            "mnm": "prepared v2 canonical; permits no remap",
        },
    }
    if source_manifests:
        manifest["source_manifests"] = copy.deepcopy(source_manifests)
    return manifest


def validate_shared_manifest(manifest: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Strict acceptance validator for a shared training manifest."""
    manifest_path = None
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).resolve()
        m = json.loads(manifest_path.read_text())
    elif isinstance(manifest, dict):
        m = manifest
    else:
        raise TypeError("manifest must be a dict or Path/str")

    if m.get("schema") != "self_audit.data.v2" or m.get("role") != "training":
        raise ValueError("manifest must have schema 'self_audit.data.v2' and role 'training'")

    if m.get("shared") is not True:
        raise ValueError("manifest must declare 'shared': true")

    if m.get("domains") != ["acdc", "mnm"]:
        raise ValueError("manifest domains must be exactly ['acdc', 'mnm'] with canonical lowercase names and no aliases")

    labeled = m.get("labeled", [])
    unlabeled = m.get("unlabeled", [])
    eval_ids = m.get("evaluation_ids", [])
    label_budget = m.get("label_budget")
    synthetic = bool(m.get("synthetic", False))

    db = m.get("domain_budgets")
    if not isinstance(db, dict) or set(db.keys()) != {"acdc", "mnm"}:
        raise ValueError("manifest must declare 'domain_budgets' with exact keys {'acdc', 'mnm'}")

    validate_budget_contract(label_budget, db.get("acdc"), db.get("mnm"), synthetic=synthetic)

    if label_budget != len(labeled):
        raise ValueError(f"label_budget ({label_budget}) != len(labeled) ({len(labeled)})")

    lids = [r["id"] for r in labeled]
    uids = [r["id"] for r in unlabeled]
    all_train_ids = lids + uids

    if len(all_train_ids) != len(set(all_train_ids)):
        raise ValueError("duplicate patient IDs in training set")

    # Domain patients completeness
    dp = m.get("domain_patients")
    if not isinstance(dp, dict) or set(dp.keys()) != {"acdc", "mnm"}:
        raise ValueError("manifest must declare 'domain_patients' with exact keys {'acdc', 'mnm'}")
    dp_acdc = set(dp["acdc"])
    dp_mnm = set(dp["mnm"])
    if dp_acdc & dp_mnm:
        raise ValueError("domain_patients sets must be disjoint")
    if dp_acdc | dp_mnm != set(all_train_ids):
        raise ValueError("domain_patients must cover all labeled and unlabeled IDs completely")

    # Domain map completeness & consistency
    dmap = m.get("domain_map")
    if not isinstance(dmap, dict) or set(dmap.keys()) != set(all_train_ids):
        raise ValueError("manifest must declare 'domain_map' covering every train patient ID")

    for pid in all_train_ids:
        dom = dmap[pid]
        if dom not in CANONICAL_DOMAINS:
            raise ValueError(f"domain_map[{pid}]='{dom}' is not a canonical domain")
        exp_dom = "acdc" if pid in dp_acdc else "mnm"
        if dom != exp_dom:
            raise ValueError(f"domain_map[{pid}]='{dom}' conflicts with domain_patients '{exp_dom}'")

        if ":" not in pid:
            raise ValueError(f"patient ID '{pid}' lacks canonical 'domain:id' namespace")
        p_dom, _ = pid.split(":", 1)
        if p_dom != dom:
            raise ValueError(f"patient ID '{pid}' namespace '{p_dom}' conflicts with domain_map '{dom}'")

    # Check evaluation IDs namespacing
    for eid in eval_ids:
        if ":" not in eid or eid.split(":", 1)[0] not in CANONICAL_DOMAINS:
            raise ValueError(f"evaluation ID '{eid}' must be namespaced with 'acdc:' or 'mnm:'")

    # Per-domain labeled counts
    acdc_l_count = sum(1 for pid in lids if dmap[pid] == "acdc")
    mnm_l_count = sum(1 for pid in lids if dmap[pid] == "mnm")
    if acdc_l_count != db["acdc"]:
        raise ValueError(f"ACDC labeled count ({acdc_l_count}) does not match budget ({db['acdc']})")
    if mnm_l_count != db["mnm"]:
        raise ValueError(f"MnM labeled count ({mnm_l_count}) does not match budget ({db['mnm']})")

    # U records strictly image only
    for r in unlabeled:
        validate_patient_id(r["id"])
        validate_relative_path(r["image"])
        if set(r.keys()) != {"id", "image", "image_sha256"}:
            raise ValueError(f"unlabeled record '{r.get('id')}' contains forbidden keys (must be image only)")

    # L records validation
    for r in labeled:
        validate_patient_id(r["id"])
        validate_relative_path(r["image"])
        if set(r.keys()) != {"id", "image", "image_sha256", "masks"} or not r["masks"]:
            raise ValueError(f"invalid labeled record '{r.get('id')}'")
        for minfo in r["masks"].values():
            validate_relative_path(minfo["path"])

    # Quarantines
    for pid in all_train_ids + eval_ids:
        validate_patient_id(pid)
        if is_quarantined_patient(pid):
            raise ValueError(f"quarantined patient '{pid}' found in manifest")

    # Candidate pool declaration BEFORE any array I/O
    cp = m.get("candidate_pool")
    if not isinstance(cp, dict) or "mode" not in cp or "restricted_candidate_pool" not in cp:
        raise ValueError("manifest must declare 'candidate_pool' with 'mode' and 'restricted_candidate_pool'")
    if cp["mode"] == "restricted_manifest_labeled" and not cp["restricted_candidate_pool"]:
        raise ValueError("restricted manifest candidate pool must declare restricted_candidate_pool=True")

    # Mapping validation BEFORE any array I/O
    mp = m.get("mapping_provenance")
    if not isinstance(mp, dict) or "acdc" not in mp or "mnm" not in mp:
        raise ValueError("manifest must declare 'mapping_provenance' containing 'acdc' and 'mnm'")
    for dom in ("acdc", "mnm"):
        desc = str(mp[dom]).strip().lower()
        for forbidden in ("double_remap", "unknown", "not canonical", "non-canonical", "not_canonical", "uncanonical"):
            if forbidden in desc:
                raise ValueError(f"rejected invalid mapping provenance for {dom}: {desc}")
        is_canonical = "canonical" in desc or "identity" in desc
        is_raw_mnm = "raw_mnm" in desc
        if dom == "acdc" and not is_canonical:
            raise ValueError(f"ACDC mapping provenance must declare canonical, got: {desc}")
        if dom == "mnm" and not (is_canonical or is_raw_mnm):
            raise ValueError(f"MnM mapping provenance must declare canonical or raw_mnm, got: {desc}")

    # Canonical Preprocessing metadata validation BEFORE any array I/O
    prep = m.get("preprocessing")
    if not isinstance(prep, dict) or prep != CANONICAL_PREPROCESSING:
        raise ValueError("manifest missing or invalid canonical preprocessing metadata")

    # SSL initialization requirement BEFORE any array I/O
    if m.get("ssl_initialization_requirement") != SSL_INITIALIZATION_REQUIREMENT:
        raise ValueError("manifest missing or invalid ssl_initialization_requirement")

    # Lineage metadata BEFORE any array I/O
    lineage = m.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("exact_lineage") != f"{db['acdc']}+{db['mnm']}":
        raise ValueError("manifest missing or invalid lineage metadata")

    # Source manifests validation if present
    if "source_manifests" in m:
        sm = m["source_manifests"]
        if not isinstance(sm, dict) or set(sm.keys()) != {"acdc", "mnm"}:
            raise ValueError("source_manifests must have exact keys {'acdc', 'mnm'}")
        for dom in ("acdc", "mnm"):
            s_entry = sm[dom]
            if not isinstance(s_entry, dict) or "path" not in s_entry or "sha256" not in s_entry:
                raise ValueError(f"source_manifests[{dom}] must declare 'path' and 'sha256'")

    # NOW, and ONLY NOW: Image overlap validation (which does array I/O)
    manifest_dir = manifest_path.parent if manifest_path is not None else None
    validate_no_image_overlap(labeled + unlabeled, evaluation_ids=eval_ids, base_dir=manifest_dir)

    return m


def synthetic_shared_fixture(
    folder: Path | str,
    seed: int = DEFAULT_SEED,
    acdc_l: int = DEFAULT_ACDC_BUDGET,
    acdc_u: int = 4,
    mnm_l: int = DEFAULT_MNM_BUDGET,
    mnm_u: int = 4,
    size: int = 32,
) -> tuple[Path, str]:
    """Manufactured multi-domain shared dataset fixture. Never cardiac accuracy evidence."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]

    def create_patients(domain: str, n_l: int, n_u: int) -> tuple[list[dict], list[dict]]:
        labeled, unlabeled = [], []
        total = n_l + n_u
        for j in range(total):
            pid = f"{domain}:patient{j+1:03d}"
            cy = size * (0.5 + rng.uniform(-0.08, 0.08))
            cx = size * (0.5 + rng.uniform(-0.08, 0.08))
            radius = np.sqrt((yy - cy)**2 + (xx - cx)**2)
            y = np.zeros((size, size), np.uint8)
            y[radius < size * 0.27] = 2  # MYO
            y[radius < size * 0.18] = 3  # LV
            y[(yy - cy)**2 + (xx - (cx - size * 0.29))**2 < (size * 0.12)**2] = 1  # RV

            # Domain-specific intensity shift to simulate distinct scanners
            domain_offset = 0.05 if domain == "acdc" else -0.05
            # Ensure unique voxel content across all patients
            pid_seed = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
            pid_rng = np.random.default_rng(pid_seed)
            unique_noise = pid_rng.normal(0, 0.01, y.shape).astype(np.float32)

            image_raw = np.clip(np.choose(y, [0.1, 0.8, 0.45, 0.9]) + domain_offset + unique_noise, 0, 1)
            image_tzyx = image_raw.astype(np.float32)[None, None]  # (T=1, Z=1, Y, X)

            ip = folder / f"{domain}_p{j+1:03d}.npy"
            np.save(ip, image_tzyx)
            r = {"id": pid, "image": ip.name, "image_sha256": digest(ip)}

            if j < n_l:
                mp = folder / f"{domain}_p{j+1:03d}_mask.npy"
                np.save(mp, y[None])  # (Z=1, Y, X)
                r["masks"] = {"0": {"path": mp.name, "sha256": digest(mp)}}
                labeled.append(r)
            else:
                unlabeled.append(r)
        return labeled, unlabeled

    acdc_labeled, acdc_unlabeled = create_patients("acdc", acdc_l, acdc_u)
    mnm_labeled, mnm_unlabeled = create_patients("mnm", mnm_l, mnm_u)

    manifest_dict = build_shared_manifest(
        acdc_labeled=acdc_labeled,
        acdc_unlabeled=acdc_unlabeled,
        mnm_labeled=mnm_labeled,
        mnm_unlabeled=mnm_unlabeled,
        evaluation_ids=["acdc:eval_001", "mnm:eval_001"],
        acdc_budget=acdc_l,
        mnm_budget=mnm_l,
        seed=seed,
        synthetic=True,
        base_dir=folder,
    )

    manifest_path = folder / "train.json"
    write_json(manifest_path, manifest_dict)
    manifest_sha = digest(manifest_path)
    return manifest_path, manifest_sha
