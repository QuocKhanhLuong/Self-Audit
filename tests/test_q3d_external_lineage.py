import copy
import hashlib
import json
from pathlib import Path

import pytest

from self_audit.candidate_q3d.lineage import load_external_lineage, validate_external_view


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _view(source, k):
    result = copy.deepcopy(source)
    eligible = source["labeled"]
    selected = {row["id"] for row in sorted(eligible, key=lambda r: _sha("42:" + r["id"]))[:k]}
    result["labeled"] = [row for row in eligible if row["id"] in selected]
    result["unlabeled"] += [{key: value for key, value in row.items() if key != "masks"}
                            for row in eligible if row["id"] not in selected]
    result["label_budget"] = k
    return result


@pytest.fixture
def evidence(tmp_path):
    source = {"schema": "self_audit.data.v2", "role": "training", "synthetic": False,
              "dataset": "acdc", "label_budget": 10, "label_seed": 42,
              "evaluation_ids": ["dev"], "class_mapping": ["BG", "RV", "MYO", "LV"],
              "labeled": [{"id": f"p{i}", "image": f"old/p{i}.npy", "image_sha256": _sha(f"image{i}"),
                           "masks": {"0": {"path": f"old/m{i}.npy", "sha256": _sha(f"mask{i}")}}}
                          for i in range(10)],
              "unlabeled": [{"id": "u", "image": "old/u.npy", "image_sha256": _sha("u")}]}
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    reference = _view(source, 8)
    reference["fewshot"] = {"source_sha256": source_sha,
                            "selection": "sha256(seed:patient_id) over source labeled IDs",
                            "eligible_labeled_ids": [row["id"] for row in source["labeled"]],
                            "selected_ids": sorted(row["id"] for row in reference["labeled"]),
                            "k": 8, "seed": 42}
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    spec = {"reference_path": str(reference_path),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "source_path": str(source_path), "source_sha256": source_sha}
    return source, spec


def test_external_selector_restores_proof_without_rewriting_views(evidence, tmp_path):
    source, spec = evidence
    context = load_external_lineage(spec, dataset="acdc", seed=42)
    selected = []
    for k in (5, 8, 10):
        view = _view(source, k)
        for row in view["labeled"] + view["unlabeled"]:
            row["image"] = row["image"].replace("old/", "relocated/")
        view["unlabeled"].reverse()
        before = copy.deepcopy(view)
        proof = validate_external_view(view, k=k, context=context)
        selected.append(set(proof["selected_ids"]))
        assert view == before and "fewshot" not in view
        assert proof["external_evidence"]["source"]["sha256"] == spec["source_sha256"]
    assert selected[0] < selected[1] < selected[2]


@pytest.mark.parametrize("field", ["reference_sha256", "source_sha256"])
def test_external_lineage_rejects_wrong_file_pin(evidence, field):
    _, spec = evidence
    spec[field] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_external_lineage(spec, dataset="acdc", seed=42)


@pytest.mark.parametrize("mutation,match", [
    ("image", "image hash"), ("mask", "mask hash"),
    ("cohort", "TRAIN patient cohort"), ("selection", "labeled IDs"),
    ("u_mask", "U record exposes masks"), ("split", "evaluation_ids"),
])
def test_external_lineage_rejects_changed_identity(evidence, mutation, match):
    source, spec = evidence
    context = load_external_lineage(spec, dataset="acdc", seed=42)
    view = _view(source, 5)
    if mutation == "image":
        view["labeled"][0]["image_sha256"] = "1" * 64
    elif mutation == "mask":
        view["labeled"][0]["masks"]["0"]["sha256"] = "1" * 64
    elif mutation == "cohort":
        view["unlabeled"].pop()
    elif mutation == "selection":
        other = next(row for row in source["labeled"] if row["id"] not in {x["id"] for x in view["labeled"]})
        removed = view["labeled"].pop()
        view["labeled"].append(copy.deepcopy(other))
        view["unlabeled"] = [row for row in view["unlabeled"] if row["id"] != other["id"]]
        view["unlabeled"].append({key:value for key,value in removed.items() if key != "masks"})
    elif mutation == "u_mask":
        view["unlabeled"][0]["masks"] = copy.deepcopy(source["labeled"][0]["masks"])
    else:
        view["evaluation_ids"] = ["different_dev"]
    with pytest.raises(ValueError, match=match):
        validate_external_view(view, k=5, context=context)


def test_campaign_nested_validator_accepts_explicit_evidence(evidence, tmp_path, monkeypatch):
    from scripts import q3d_campaign as campaign
    source, spec = evidence
    roles = {}
    for k in (5, 8, 10):
        path = tmp_path / f"k{k}.json"
        path.write_text(json.dumps(_view(source, k)))
        roles[f"k{k}:train"] = str(path)
        if k == 8:
            monkeypatch.setattr(campaign, "K8_LOCKED_TRAIN_MANIFEST_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    before = {path: Path(path).read_bytes() for path in roles.values()}
    audit = campaign._validate_nested_train_lineage({"acdc": roles}, datasets=("acdc",),
        selected_ks=(5, 8, 10), seed=42, lineage_reference=spec)
    assert audit["status"] == "PASS"
    assert all(row["selector_validation"] == "REPRODUCED_FROM_HASH_PINNED_SOURCE" for row in audit["records"])
    assert before == {path: Path(path).read_bytes() for path in roles.values()}
    with pytest.raises(ValueError, match="missing few-shot lineage"):
        campaign._validate_nested_train_lineage({"acdc": roles}, datasets=("acdc",), selected_ks=(5, 8, 10), seed=42)
