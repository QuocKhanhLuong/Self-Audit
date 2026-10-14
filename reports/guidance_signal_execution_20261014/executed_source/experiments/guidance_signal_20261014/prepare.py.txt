"""Prepare private Q/h/Z evidence from an ORIGINAL frozen TRAIN cache, offline.

No inference, pretrained downloads, reference arrays, student predictions, or
reference-dependent tolerances are used. Failure leaves an unfrozen FAILED.json
receipt in a fresh output directory; it can never masquerade as a signal bank.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from .evidence import (INVARIANT_ATOL, LINEAR_TOLERANCE, MAX_Z_ERROR, Q_ATOL, R0_ATOL,
                       ReproductionError, reconstruct)
from .runtime import (EXPECTED_SOURCE_SHA256, SNAPSHOT_SHA, TEACHER_CONFIG_SHA256,
                      TEACHER_REVISION, TEACHER_SHA256, load_runtime, sha256)

ROLE = "guidance_signal_bank_v1"
ORIGINAL_RECIPE_BY_BANK = {
    # Original private bytes, not the sanitized publication's hashes.
    "04fb2d30df6c0e0ed235dcf851a491895661b7fd4763e6b89ce95ca6811accae":
        "bed149d73bb0527c1dbcea8ff72e9b28e093149974dcb0c5893fd43d53a6092b",
}
ARRAY_FIELDS = {"Q", "M", "h", "Z", "valid", "r0", "V", "raw_votes", "masses",
                "strength", "z_error_bound", "source_support_data", "source_support_indices",
                "source_support_indptr", "source_support_shape"}


def _pid(pid):
    if not isinstance(pid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", pid) or ".." in pid:
        raise ValueError("unsafe patient ID")
    return pid


def _not_sanitized(value):
    text = json.dumps(value)
    if re.search(r"\$\{|<REDACTED|<SANITIZED|\[REDACTED\]|__REDACTED__", text, re.I):
        raise ValueError("sanitized/placeholder artifact refused; ORIGINAL payloads required")


def guard_fresh_output(output, *input_roots):
    """Reject writes inside original/locked evidence, including via symlinks."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("output must be fresh: " + str(output))
    resolved = output.resolve()
    for root in input_roots:
        root = Path(root).resolve()
        if resolved.is_relative_to(root) or root.is_relative_to(resolved):
            raise ValueError("output overlaps an original input root")
    for ancestor in resolved.parents:
        if any((ancestor/name).exists() for name in
               ("FROZEN.json", "PREDICTIONS_LOCKED.json", "LOCKED.json", "PUBLICATION.json")):
            raise ValueError("output is inside frozen/locked evidence")
    return output


def verify_original(bank, recipe_path, expected_bank_sha):
    runtime = load_runtime()
    bank, recipe_path = Path(bank), Path(recipe_path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_bank_sha or ""):
        raise ValueError("explicit expected ORIGINAL bank manifest SHA256 required")
    if expected_bank_sha not in ORIGINAL_RECIPE_BY_BANK:
        raise ValueError("bank SHA256 is not the pinned original publication identity; no identity override")
    if (bank/"FROZEN.json").is_symlink() or sha256(bank/"FROZEN.json") != expected_bank_sha:
        raise ValueError("original bank manifest SHA256 mismatch")
    manifest = runtime.io.verify(bank, "guidance_bank")
    recipe = runtime.io.read_json(recipe_path)
    _not_sanitized(manifest)
    _not_sanitized(recipe)
    if sha256(recipe_path) != ORIGINAL_RECIPE_BY_BANK[expected_bank_sha]:
        raise ValueError("original published bank/recipe identity mismatch")
    if manifest.get("split") != "train":
        raise ValueError("signal preparation accepts TRAIN banks only; TEST excluded")
    if recipe.get("schema") != "self_audit.full_guidance_recipe.v1":
        raise ValueError("original recipe schema mismatch")
    for item in (manifest, recipe):
        if item.get("source_sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError("original artifact source identity mismatch: only pinned executed snapshot "
                             + SNAPSHOT_SHA + " is supported; older banks need their own immutable runtime")
    cfg = runtime.core.GuidanceConfig(**manifest["config"])
    canonical_config = json.loads(json.dumps(cfg.dictionary()))
    if (manifest["config"] != canonical_config or recipe.get("config") != canonical_config
            or runtime.io.read_json(bank/"config.json") != canonical_config):
        raise ValueError("recipe/bank config mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", recipe.get("config_sha256", "")):
        raise ValueError("missing original config identity")
    splits = runtime.io.read_splits(bank/"split.json")
    if manifest.get("locked_splits") != splits or recipe.get("splits") != splits:
        raise ValueError("recipe/bank locked patient split mismatch")
    ids = manifest.get("patients")
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or any(_pid(pid) not in splits["train"] for pid in ids)
            or set(ids) & (set(splits["val"]) | set(splits["test"]))):
        raise ValueError("original bank patients must be a nonempty TRAIN subset")
    if {name[:-4] for name in manifest["files"] if name.endswith(".npz")} != set(ids):
        raise ValueError("missing/unexpected original patient payloads")
    if set(manifest.get("image_records", {})) != set(ids):
        raise ValueError("missing original native image records")
    if set(manifest.get("patient_reports", {})) != set(ids):
        raise ValueError("missing original permission reports")
    teacher = manifest.get("teacher", {})
    if (teacher.get("frozen") is not True or teacher.get("segmentation_finetune") is not False
            or teacher.get("checkpoint_sha256") != TEACHER_SHA256
            or teacher.get("config_sha256") != TEACHER_CONFIG_SHA256
            or teacher.get("model_revision") != TEACHER_REVISION
            or teacher.get("adapter_source_sha256") != EXPECTED_SOURCE_SHA256["teacher.py"]
            or recipe.get("expected_teacher_sha256") != TEACHER_SHA256
            or recipe.get("expected_model_revision") != TEACHER_REVISION
            or recipe.get("teacher_config_sha256") != TEACHER_CONFIG_SHA256):
        raise ValueError("recipe/bank pinned frozen teacher identity mismatch")
    policy = manifest.get("orientation_policy", "cardinal_v1")
    if policy not in ("cardinal_v1", "world_left_v2") or recipe.get("orientation_policy", "cardinal_v1") != policy:
        raise ValueError("recipe/bank orientation policy mismatch")
    if manifest.get("seed") != recipe.get("seed"):
        raise ValueError("recipe/bank seed mismatch")
    # Read only split JSON, never raw cine or reference files. For official test
    # exports the bank uses the effective split written beside the original recipe.
    original_split = Path(recipe["split_manifest"])
    if sha256(original_split) != recipe.get("split_sha256"):
        raise ValueError("original recipe split digest mismatch")
    if recipe.get("official_acdc_test_unlabeled"):
        effective_path = recipe_path.parent/"EFFECTIVE_SPLIT.json"
        effective = runtime.io.read_json(effective_path)
        if (sha256(effective_path) != manifest.get("source_split_sha256")
                or runtime.io.read_splits(effective_path) != splits
                or effective.get("provenance", {}).get("source_sha256") != recipe["split_sha256"]):
            raise ValueError("effective split identity mismatch")
        base = runtime.io.read_splits(original_split)
        if any(base[key] != splits[key] for key in ("train", "val")):
            raise ValueError("original/effective TRAIN or VAL split mismatch")
    elif (manifest.get("source_split_sha256") != recipe["split_sha256"]
          or runtime.io.read_splits(original_split) != splits):
        raise ValueError("original split identity mismatch")
    return manifest, recipe, cfg


def geometry_arguments(runtime, manifest, pid, image, small):
    """Revalidate the RECORDED geometry; never substitute a new cardinal rule."""
    meta = manifest["image_records"][pid]
    if list(reversed(meta["shape_xyzt"])) != list(image.shape):
        raise ValueError("cached native image/native geometry shape mismatch")
    if "patient_left_axis" not in meta or not re.fullmatch(r"[0-9a-f]{64}", meta.get("image_sha256", "")):
        raise ValueError("incomplete native image geometry identity")
    policy = manifest.get("orientation_policy", "cardinal_v1")
    if policy == "cardinal_v1":
        return {}, dict(policy=policy, patient_left_axis=meta["patient_left_axis"],
                        native_hw=list(image.shape[-2:]), bank_hw=list(small.shape[-2:]),
                        image_sha256=meta["image_sha256"])
    recorded = meta.get("world_left_geometry")
    if (not isinstance(recorded, dict) or recorded.get("policy") != "world_left_v2"
            or recorded.get("image_sha256") != meta["image_sha256"]):
        raise ValueError("missing recorded world-left geometry")
    geometry = runtime.experimental_geometry.world_left_geometry(
        meta["affine"], image.shape[-2:], small.shape[-2:],
        sform_code=recorded["sform_code"], qform_code=recorded["qform_code"])
    # The original reader may abstain for uncoded headers. Preserve its zero
    # vector, including on images with a seemingly interpretable affine.
    for key, value in geometry.items():
        actual = recorded.get(key)
        if isinstance(value, (list, float)):
            if actual is None or not np.allclose(actual, value, rtol=0, atol=1e-10):
                raise ValueError("recorded native-to-bank geometry mismatch: " + key)
        elif actual != value:
            raise ValueError("recorded world geometry mismatch: " + key)
    return {"left_vector_yx": recorded["left_vector_yx"]}, recorded


def scalar_audit(arrays, original, sources):
    """Compact support/contrast audit; class names are hypotheses, not truth."""
    def quantiles(value):
        value = np.asarray(value)
        return np.quantile(value, [0., .01, .25, .5, .75, .99, 1.]).tolist() if value.size else None
    q, r0 = original["q"], original["r0"]
    votes, h, valid = arrays["raw_votes"], arrays["h"], arrays["valid"]
    argmax = q.argmax(-1)
    entropy = -(q*np.log(np.maximum(q, 1e-8))).sum(-1)/np.log(4)
    classes = {}
    for cls, name in enumerate(("BG", "RV", "MYO", "LV")):
        others = [c for c in range(4) if c != cls]
        class_sources = [source for source in sources if source["class_id"] == cls]
        groups = [source["repeated_mask_group"] for source in class_sources]
        group_counts = {}
        for group in groups:
            group_counts[group] = group_counts.get(group, 0)+1
        classes[name] = dict(class_id=cls,
            raw_vote_pixels=int((votes[..., cls] > 0).sum()),
            raw_vote_mass=float(votes[..., cls].sum(dtype=np.float64)),
            R0_above_uniform_pixels=int((r0[..., cls] > .25).sum()),
            R0_positive_winning_contrast_pixels=int((r0[..., cls] > r0[..., others].max(-1)).sum()),
            Q_above_uniform_pixels=int((q[..., cls] > .25).sum()),
            Q_positive_winning_contrast_pixels=int((q[..., cls] > q[..., others].max(-1)).sum()),
            Q_argmax_pixels=int((argmax == cls).sum()),
            original_hard_accepted_pixels=int((original["labels"] == cls).sum()),
            Q_class_probability_quantiles=quantiles(q[..., cls]),
            h_on_Q_argmax_quantiles=quantiles(h[argmax == cls]),
            source_count=len(class_sources), source_mask_group_count=len(group_counts),
            repeated_source_mask_group_count=sum(count > 1 for count in group_counts.values()))
    return dict(classes=classes, quantile_levels=[0., .01, .25, .5, .75, .99, 1.],
        Q_confidence_quantiles=quantiles(q.max(-1)), Q_normalized_entropy_quantiles=quantiles(entropy),
        h_quantiles=quantiles(h), h_on_valid_quantiles=quantiles(h[valid]),
        raw_class_conflict_pixels=int(((votes > 0).sum(-1) > 1).sum()),
        valid_pixels=int(valid.sum()), invalid_pixels=int((~valid).sum()),
        original_UNKNOWN_pixels=int((original["labels"] == 255).sum()),
        source_count=len(sources), source_mask_group_count=len({s["repeated_mask_group"] for s in sources}),
        argmax_ties="first_class_index; strict_winning_contrast_reported_separately",
        evidence_independence="not_asserted", interpretation="support_not_named_anatomical_accuracy")


def prepare(bank, recipe, output, *, patients=None, expected_bank_sha, min_h=1e-6):
    bank, recipe, output = Path(bank), Path(recipe), Path(output)
    guard_fresh_output(output, bank, recipe.parent)
    if not np.isfinite(min_h) or not 0 < min_h < 1:
        raise ValueError("min_h must be finite in (0,1)")
    manifest, original_recipe, cfg = verify_original(bank, recipe, expected_bank_sha)
    runtime = load_runtime()
    ids = list(manifest["patients"]) if patients is None else list(patients)
    if (not ids or len(ids) != len(set(ids))
            or any(_pid(pid) not in manifest["patients"] for pid in ids)):
        raise ValueError("selected patients must be unique members of original TRAIN bank")
    recipe_hash = sha256(recipe)
    original_permission_counts = [sum(report["class_pixels"][cls] > 0
        for report in manifest["patient_reports"].values()) for cls in range(4)]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    reports, ledgers, geometry = {}, {}, {}
    try:
        for pid in ids:
            cached = runtime.io.load_bank_patient(bank, pid)
            image, features = cached["image"], cached["features"]
            if image.dtype != np.float32 or features.shape != (*image.shape[:2], cfg.grid, cfg.grid, cfg.feature_dim):
                raise ValueError("original native image/cache dtype or feature shape mismatch")
            small = F.interpolate(torch.from_numpy(image.copy()).flatten(0, 1)[:, None],
                                  (cfg.grid, cfg.grid), mode="bilinear", align_corners=False)
            small = small[:, 0].reshape(*image.shape[:2], cfg.grid, cfg.grid).numpy()
            args, geometry[pid] = geometry_arguments(runtime, manifest, pid, image, small)
            arrays, report, sources = reconstruct(small, features,
                manifest["image_records"][pid]["patient_left_axis"], cfg,
                patient_id=pid, min_h=min_h, cached_r0=cached["r0"], cached_q=cached["q"],
                cached_reliability=cached["reliability"], cached_labels=cached["labels"], **args)
            ledger_path = output/(pid + ".sources.json")
            runtime.io.write_json(ledger_path, dict(patient=pid, sources=sources,
                evidence_independence="not_asserted_multiscale_votes_can_repeat",
                support="initial s*source_vote/mass; not propagated; CSR node by source",
                original_bank_payload_sha256=manifest["files"][pid+".npz"]))
            os.chmod(ledger_path, 0o600)
            ledgers[pid] = sha256(ledger_path)
            np.savez_compressed(output/(pid+".npz"), **arrays)
            os.chmod(output/(pid+".npz"), 0o600)
            report.update(feature_cache_dtype=str(features.dtype),
                          ledger_sha256=ledgers[pid], native_image_sha256=manifest["image_records"][pid]["image_sha256"],
                          scalar_audit=scalar_audit(arrays, cached, sources))
            reports[pid] = report
        # Recheck external input hashes after all computation, before sealing.
        runtime.io.verify(bank, "guidance_bank")
        if sha256(bank/"FROZEN.json") != expected_bank_sha or sha256(recipe) != recipe_hash:
            raise ValueError("original inputs changed during signal preparation")
        load_runtime()  # Recheck all 11 source hashes before sealing.
        runtime.io.write_json(output/"REPORT.json", dict(status="PASS_REPRODUCTION", patients=reports,
                              original_permission_patient_counts=original_permission_counts,
                              original_allowed_classes=[n >= 2 for n in original_permission_counts],
                              gt_read=False, reference_tuning=False, inference_run=False))
        return runtime.io.freeze(output, dict(role=ROLE, split="train", patients=ids,
            locked_splits=manifest["locked_splits"], original_bank_manifest_sha256=expected_bank_sha,
            recipe_sha256=recipe_hash, snapshot_sha=SNAPSHOT_SHA,
            source_sha256=dict(EXPECTED_SOURCE_SHA256),
            experiment_source_sha256={name: sha256(Path(__file__).with_name(name))
                                      for name in ("runtime.py", "evidence.py", "prepare.py")},
            config=manifest["config"], teacher=manifest["teacher"],
            source_split_sha256=manifest["source_split_sha256"],
            orientation_policy=manifest.get("orientation_policy", "cardinal_v1"),
            native_to_bank_geometry=geometry, ledger_sha256=ledgers, patient_reports=reports,
            original_permission_patient_counts=original_permission_counts,
            original_allowed_classes=[n >= 2 for n in original_permission_counts],
            min_h=float(min_h), max_Z_error=MAX_Z_ERROR, r0_atol=R0_ATOL, q_atol=Q_ATOL,
            invalid_Z="NaN; consumers must use valid mask", unknown=255,
            arrays_axis_order="TZYX[class]", status="PASS_REPRODUCTION", gt_read=False,
            numerical_bounds="iterative_truncation_exact_arithmetic_no_interval_roundoff_or_semantic_certificate",
            inference_run=False, quality_certified=False, private_payload=True,
            evidence_independence="not_asserted_multiscale_votes_can_repeat"))
    except Exception as exc:
        runtime.io.write_json(output/"FAILED.json", dict(status="FAIL_REPRODUCTION" if isinstance(exc, ReproductionError)
                              else "FAIL_IDENTITY_OR_PAYLOAD", error=str(exc),
                              diagnostics=getattr(exc, "diagnostics", None),
                              completed_patients=list(reports), original_bank_manifest_sha256=expected_bank_sha,
                              frozen=False, gt_read=False))
        raise


def _signal_manifest(signal_dir, *, verify_all=True):
    runtime = load_runtime()
    path = Path(signal_dir)/"FROZEN.json"
    if path.is_symlink():
        raise ValueError("signal manifest symlink refused")
    manifest = runtime.io.verify(signal_dir, ROLE) if verify_all else runtime.io.read_json(path)
    if (manifest.get("schema") != "self_audit.stable_guidance.v1" or manifest.get("role") != ROLE
            or manifest.get("split") != "train" or manifest.get("status") != "PASS_REPRODUCTION"
            or manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256
            or manifest.get("snapshot_sha") != SNAPSHOT_SHA):
        raise ValueError("signal identity/reproduction contract mismatch")
    bank_sha = manifest.get("original_bank_manifest_sha256")
    if (bank_sha not in ORIGINAL_RECIPE_BY_BANK
            or manifest.get("recipe_sha256") != ORIGINAL_RECIPE_BY_BANK[bank_sha]):
        raise ValueError("signal original bank/recipe publication identity mismatch")
    expected_experiment = {name: sha256(Path(__file__).with_name(name))
                           for name in ("runtime.py", "evidence.py", "prepare.py")}
    if manifest.get("experiment_source_sha256") != expected_experiment:
        raise ValueError("signal experiment source identity mismatch")
    if (manifest.get("r0_atol") != R0_ATOL or manifest.get("q_atol") != Q_ATOL
            or manifest.get("max_Z_error") != MAX_Z_ERROR
            or not np.isfinite(manifest.get("min_h", np.nan)) or not 0 < manifest["min_h"] < 1):
        raise ValueError("signal numerical tolerances differ from fixed contract")
    ids = manifest.get("patients", [])
    if not ids or len(ids) != len(set(ids)) or any(_pid(pid) not in manifest["locked_splits"]["train"] for pid in ids):
        raise ValueError("signal patients must be original TRAIN")
    if set(ids) & (set(manifest["locked_splits"]["val"]) | set(manifest["locked_splits"]["test"])):
        raise ValueError("signal patient split leakage")
    if set(manifest.get("patient_reports", {})) != set(ids) or set(manifest.get("ledger_sha256", {})) != set(ids):
        raise ValueError("missing signal patient reports/ledger hashes")
    for pid in ids:
        report = manifest["patient_reports"][pid]
        reproduction = report.get("reproduction", {})
        if (reproduction.get("status") != "PASS_REPRODUCTION"
                or reproduction.get("r0_atol") != R0_ATOL or reproduction.get("q_atol") != Q_ATOL
                or reproduction.get("rtol") != 0 or reproduction.get("hard_label_mismatches") != 0):
            raise ValueError("signal reproduction/gate receipt missing or invalid")
        for field, limit in (("r0_max_error", R0_ATOL), ("q_max_error", Q_ATOL),
                             ("original_q_max_error", Q_ATOL), ("reliability_max_error", Q_ATOL),
                             ("decomposition_original_q_max_error", Q_ATOL)):
            value = reproduction.get(field)
            if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= limit:
                raise ValueError("signal reproduction error exceeds fixed limit: " + field)
        certificate = report.get("linear_solver", {})
        bound = certificate.get("error_bound", np.inf)
        if (certificate.get("converged") is not True or not np.isfinite(bound)
                or not 0 <= bound <= LINEAR_TOLERANCE
                or certificate.get("tolerance") != LINEAR_TOLERANCE
                or report.get("max_Z_error_allowed") != MAX_Z_ERROR):
            raise ValueError("uncertified signal linear solve")
    return manifest


def load_signal_bank(signal_dir, original_bank):
    """Verify the private artifact and its exact original bank binding."""
    manifest = _signal_manifest(Path(signal_dir))
    runtime = load_runtime()
    original = runtime.io.verify(original_bank, "guidance_bank")
    if (sha256(Path(original_bank)/"FROZEN.json") != manifest["original_bank_manifest_sha256"]
            or original.get("split") != "train" or original.get("source_sha256") != EXPECTED_SOURCE_SHA256
            or original.get("locked_splits") != manifest["locked_splits"]
            or any(pid not in original.get("patients", []) for pid in manifest["patients"])):
        raise ValueError("signal/original bank identity mismatch")
    _not_sanitized(original)
    for key in ("config", "teacher", "source_split_sha256"):
        if original.get(key) != manifest.get(key):
            raise ValueError("signal/original " + key + " mismatch")
    if original.get("orientation_policy", "cardinal_v1") != manifest.get("orientation_policy"):
        raise ValueError("signal/original geometry policy mismatch")
    # Array comparisons independently reject a forged or stale PASS marker.
    # Raw evidence replay itself is certified by prepare's exact code/source
    # pins and full original reproduction receipts, not by a semantic claim.
    for pid in manifest["patients"]:
        arrays = load_signal_patient(signal_dir, pid)
        cached = runtime.io.load_bank_patient(original_bank, pid)
        if (arrays["Q"].shape != cached["q"].shape
                or not np.allclose(arrays["Q"], cached["q"], atol=Q_ATOL, rtol=0)
                or not np.allclose(arrays["r0"], cached["r0"], atol=R0_ATOL, rtol=0)):
            raise ValueError("signal/original Q or R0 reproduction mismatch")
    return manifest


def load_signal_patient(signal_dir, pid):
    """Verify sealed payload and validate NaN/mass/CSR invariants on every load."""
    pid, signal_dir = _pid(pid), Path(signal_dir)
    manifest = _signal_manifest(signal_dir, verify_all=False)
    if pid not in manifest["patients"]:
        raise ValueError("patient outside sealed signal TRAIN subset")
    ledger_path = signal_dir/(pid+".sources.json")
    if sha256(ledger_path) != manifest["ledger_sha256"][pid]:
        raise ValueError("source ledger identity mismatch")
    payload_path = signal_dir/(pid+".npz")
    if (payload_path.is_symlink() or ledger_path.is_symlink()
            or sha256(payload_path) != manifest["files"].get(pid+".npz")
            or sha256(ledger_path) != manifest["files"].get(pid+".sources.json")):
        raise ValueError("signal patient payload hash mismatch")
    with np.load(payload_path, allow_pickle=False) as payload:
        if set(payload.files) != ARRAY_FIELDS:
            raise ValueError("unexpected signal array fields")
        arrays = {name: payload[name] for name in payload.files}
    h, valid = arrays["h"], arrays["valid"]
    if h.ndim != 4 or valid.dtype != np.bool_ or valid.shape != h.shape:
        raise ValueError("signal grid/valid shape mismatch")
    for name in ("Q", "M", "Z", "r0", "V", "raw_votes"):
        if arrays[name].shape != (*h.shape, 4):
            raise ValueError("signal class shape mismatch: " + name)
    for name in ("masses", "strength", "z_error_bound"):
        if arrays[name].shape != h.shape:
            raise ValueError("signal scalar shape mismatch: " + name)
    for name in ARRAY_FIELDS - {"Z", "z_error_bound"}:
        if not np.isfinite(arrays[name]).all():
            raise ValueError("nonfinite signal: " + name)
    if (not np.isfinite(arrays["Z"][valid]).all() or not np.isnan(arrays["Z"][~valid]).all()
            or np.isnan(arrays["z_error_bound"]).any() or (arrays["z_error_bound"] < 0).any()
            or not np.array_equal(valid, (h > manifest["min_h"]) & (arrays["z_error_bound"] <= manifest["max_Z_error"]))
            or (h < 0).any() or (h > 1+INVARIANT_ATOL).any()
            or (arrays["M"] < 0).any()
            or not np.allclose(arrays["M"].sum(-1), h, rtol=0, atol=INVARIANT_ATOL)
            or not np.allclose(arrays["Q"], (1-h[..., None])*.25+arrays["M"], rtol=0, atol=INVARIANT_ATOL)
            or not np.allclose(arrays["Z"][valid]*h[valid, None], arrays["M"][valid], rtol=0, atol=INVARIANT_ATOL)):
        raise ValueError("invalid signal mass/NaN contract")
    if (any((arrays[name] < 0).any() for name in ("r0", "V", "raw_votes", "masses", "strength"))
            or (arrays["strength"] > 1).any()
            or not np.allclose(arrays["raw_votes"].sum(-1), arrays["masses"], rtol=1e-6, atol=1e-6)
            or not np.allclose(arrays["strength"], 1-np.exp(-arrays["masses"]), rtol=0, atol=R0_ATOL)
            or not np.allclose(arrays["V"]*arrays["masses"][..., None], arrays["raw_votes"], rtol=1e-7, atol=1e-7)
            or not np.allclose(arrays["r0"], arrays["strength"][..., None]*arrays["V"]
                               +(1-arrays["strength"][..., None])*.25, rtol=0, atol=R0_ATOL)):
        raise ValueError("invalid raw evidence/strength contract")
    shape = arrays["source_support_shape"]
    sources = json.loads(ledger_path.read_text())["sources"]
    indptr, indices, data = (arrays["source_support_"+name] for name in ("indptr", "indices", "data"))
    if (shape.shape != (2,) or not np.array_equal(shape, [h.size, len(sources)])
            or indptr.shape != (h.size+1,) or indptr[0] != 0 or indptr[-1] != len(data)
            or indices.shape != data.shape or (np.diff(indptr) < 0).any()
            or (indices < 0).any() or (indices >= len(sources)).any() or (data < 0).any()):
        raise ValueError("invalid source support CSR")
    if (len({s["source_id"] for s in sources}) != len(sources)
            or any(s.get("independent_evidence") is not False or s.get("class_id") not in range(4)
                   for s in sources)):
        raise ValueError("invalid source identities/independence declaration")
    support = sparse.csr_matrix((data, indices, indptr), shape=tuple(shape))
    for cls in range(4):
        columns = [i for i, source in enumerate(sources) if source["class_id"] == cls]
        actual = np.asarray(support[:, columns].sum(1)).reshape(h.shape)
        expected = arrays["strength"]*arrays["V"][..., cls]
        if not np.allclose(actual, expected, rtol=1e-6, atol=INVARIANT_ATOL):
            raise ValueError("source support numerators differ from raw class evidence")
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patients", nargs="+", help="explicit original TRAIN IDs, space or comma separated")
    parser.add_argument("--expected-bank-sha", required=True)
    parser.add_argument("--min-h", type=float, default=1e-6)
    args = vars(parser.parse_args(argv))
    if args["patients"]:
        args["patients"] = [pid for part in args["patients"] for pid in part.split(",")]
    try:
        manifest = prepare(**args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(dict(status=manifest["status"], patients=manifest["patients"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
