"""SSL4MIS-split campaign: lock, data views, sampler isolation, teacher-KD target mode, config diff (CPU, synthetic arrays)."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from self_audit.candidate_pg.config import validate_config  # noqa: E402
from self_audit.candidate_pg.data import PGSampler, open_pg_data  # noqa: E402

import ssl4mis_build_lock as lockmod  # noqa: E402
import w2s_make_configs_ssl4mis as mk  # noqa: E402

LISTS = ROOT / "configs/candidate_pg/ssl4mis_lists"
COMMIT = "06df6047a59aba9988ced8331998b5957ecb356b"
L7 = ["patient021", "patient038", "patient049", "patient050", "patient058", "patient099", "patient100"]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_lock_reproduces_ssl4mis_sets_and_labeled_rule() -> None:
    lock = lockmod.build(LISTS, COMMIT)
    assert lockmod.LABELED_SLICES["7"] == 136
    assert lock["labeled"]["patients"] == L7 and len(lock["labeled"]["volumes"]) == 14
    sets = [set(lock[k]["patients"]) for k in ("labeled", "unlabeled", "val", "test")]
    assert [len(s) for s in sets] == [7, 63, 10, 20]
    assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4))
    assert set().union(*sets) == {f"patient{i:03d}" for i in range(1, 101)}
    # the first 136 slices are exactly those volumes
    slices = (LISTS / "train_slices.list").read_text().split()
    assert {lockmod.volume_of(s) for s in slices[:136]} == set(lock["labeled"]["volumes"])
    committed = json.loads((ROOT / "configs/candidate_pg/ssl4mis_split_lock.json").read_text())
    assert committed == lock and committed["source"]["list_sha256"]["val.list"] == _sha(LISTS / "val.list")


def _synthetic_prepared(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny but structurally exact prepared directory (100 patients, 2-frame stacks)."""
    lock = lockmod.build(LISTS, COMMIT)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    out = tmp_path / "prepared"
    (out / "arrays").mkdir(parents=True)
    shape = (2, 16, 16)
    geom = {"shape": list(shape), "spacing_zyx": [10.0, 1.4, 1.4], "affine_zyx": [[0, 0, -1.4, 0], [0, 1.4, 0, 0], [10.0, 0, 0, 0], [0, 0, 0, 1]], "context": "full_volume"}
    rng = np.random.default_rng(0)
    roles = {"train_l": lock["labeled"]["patients"], "train_u": lock["unlabeled"]["patients"], "val": lock["val"]["patients"], "test": lock["test"]["patients"]}
    recs = {}
    for pid in sum(roles.values(), []):
        image = out / "arrays" / f"{pid}_edes.npy"
        np.save(image, rng.random((2, *shape), dtype=np.float32))
        rec = {"id": pid, "image": str(image), "image_sha256": _sha(image)}
        if pid not in roles["train_u"]:
            masks = {}
            for t in (0, 1):
                m = out / "arrays" / f"{pid}_t{t}_mask.npy"
                np.save(m, rng.integers(0, 4, shape).astype(np.uint8))
                masks[str(t)] = {"path": str(m), "sha256": _sha(m), "source_path": "x", "source_sha256": "0" * 64, "source_frame": t, "transpose": [2, 1, 0]}
            rec.update({"phases": {"ED": 0, "ES": 1}, "masks": masks})
        recs[pid] = rec
    classes = {"0": "BG", "1": "RV", "2": "MYO", "3": "LV"}

    def base(role, ids):
        return {"schema": "self_audit.ssl4mis.data.v1", "role": role, "dataset": "acdc", "synthetic": False, "class_mapping": classes,
                "geometry": {"by_patient": {p: geom for p in ids}}}

    train = base("train", roles["train_l"] + roles["train_u"])
    train.update({"labeled": [recs[p] for p in sorted(roles["train_l"])],
                  "unlabeled": [{k: recs[p][k] for k in ("id", "image", "image_sha256")} for p in sorted(roles["train_u"])],
                  "patients": sorted(roles["train_u"])})
    refs = {}
    for role, manifest in (("train", train), *((r, {**base(r, roles[r]), "patients": [recs[p] for p in sorted(roles[r])]}) for r in ("val", "test"))):
        path = out / f"{role}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        refs[role] = {"path": str(path), "sha256": _sha(path)}
    receipt = {"schema": "self_audit.ssl4mis.materialization.v1", "status": "READY", "lock": {"path": str(lock_path), "sha256": _sha(lock_path)},
               "manifests": refs, "split_ids": roles}
    (out / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return out, lock_path


@pytest.fixture()
def prepared(tmp_path):
    return _synthetic_prepared(tmp_path)


def _config(prepared, lock_path, arm="S2main"):
    return mk.build(arm, prepared, lock_path, 0.99)


def test_u_sampler_only_yields_ed_es_of_u63_and_never_val_or_test(prepared) -> None:
    out, lock_path = prepared
    cfg = _config(out, lock_path)
    bundle = open_pg_data(cfg)
    lock = json.loads(lock_path.read_text())
    l7, u63 = set(lock["labeled"]["patients"]), set(lock["unlabeled"]["patients"])
    held_out = set(lock["val"]["patients"]) | set(lock["test"]["patients"])
    assert len(bundle.train.labeled_keys()) == 14 and len(bundle.train.unlabeled_keys()) == 126
    sampler = PGSampler(bundle, seed=42)
    touched = set()
    for _ in range(300):
        lk, uk, uk2 = sampler.next_step()
        assert lk[0] in l7 and uk[0] in u63 and uk2[0] in u63 and uk[0] != uk2[0]
        assert {lk[1], uk[1], uk2[1]} <= {0, 1}  # ED/ES only: the stored stack has exactly two frames
        touched |= {lk[0], uk[0], uk2[0]}
    assert touched <= l7 | u63 and not touched & held_out
    with pytest.raises(ValueError):
        bundle.train.load((sorted(held_out)[0], 0), labeled=True)
    with pytest.raises(ValueError):
        bundle.train.load((sorted(l7)[0], 0), labeled=False)  # L7 frames are not available as U
    assert len(bundle.val.labeled_keys()) == 20 and len(bundle.test.labeled_keys()) == 40
    with pytest.raises(ValueError):
        bundle.val.load(bundle.val.labeled_keys()[0], labeled=False)


def test_tampered_image_or_wrong_receipt_pin_is_rejected(prepared) -> None:
    out, lock_path = prepared
    cfg = _config(out, lock_path)
    victim = json.loads((out / "val.json").read_text())["patients"][0]["image"]
    arr = np.load(victim)
    arr[0, 0, 0, 0] = 0.5 if arr[0, 0, 0, 0] != 0.5 else 0.25
    np.save(victim, arr)
    with pytest.raises(ValueError, match="SHA256"):
        open_pg_data(cfg)
    bad = copy.deepcopy(cfg)
    bad["data"]["manifests"]["test"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        open_pg_data(bad)


def test_student_arm_configs_differ_only_in_target_mode_and_cbkd(prepared) -> None:
    out, lock_path = prepared

    def flat(x, p=""):
        if isinstance(x, dict):
            r = {}
            for k, v in x.items():
                r.update(flat(v, f"{p}/{k}"))
            return r
        return {p: json.dumps(x)}

    cfgs = {arm: flat(_config(out, lock_path, arm)) for arm in ("S0", "S1kd", "S2arb", "S2main")}
    reference = cfgs["S0"]
    for arm, cfg in cfgs.items():
        diff = {k for k in set(cfg) | set(reference) if cfg.get(k) != reference.get(k)}
        assert all(k == "/pg/target_mode" or k.startswith("/cbkd/") for k in diff), (arm, sorted(diff))
    assert json.loads(cfgs["S1kd"]["/pg/target_mode"]) == "teacher" and json.loads(cfgs["S2main"]["/cbkd/variant"]) == "cbkd_sa_gated"
    assert json.loads(cfgs["S2main"]["/pg/target_mode"]) == "w2s_arbiter"
    assert json.loads(reference["/training/selection/split"]) == "VAL10" and json.loads(reference["/training/eval_splits"]) == ["train_labeled", "VAL10"]
    assert json.loads(reference["/data/k_labeled"]) == 7


def test_teacher_target_mode_builds_packet_from_teacher_only_at_fixed_095(monkeypatch) -> None:
    from self_audit.candidate_pg import w2s_step
    from test_candidate_pg_w2s import _Student, _Teacher, _config as w2s_cfg, _step

    captured = []
    real = w2s_step._packet_loss
    monkeypatch.setattr(w2s_step, "_packet_loss", lambda l, y, w, v: (captured.append((y.clone(), w.clone())), real(l, y, w, v))[1])
    cfg = w2s_cfg("self")
    cfg["pg"]["target_mode"] = "teacher"
    # the student strongly says class 2; the teacher (class 1, p ~ 1) must still decide the target
    out = _step(cfg, _Student(bias=(0, 0, 6.0, 0)), _Teacher(cls=1, strength=6.0), update=10)
    assert out["diagnostics"]["target_mode"] == "teacher" and "rule_fraction" not in out["diagnostics"]
    assert captured and all((y == 1).all() and (w == 1).all() for y, w in captured)
    captured.clear()
    _step(cfg, _Student(bias=(0, 0, 6.0, 0)), _Teacher(cls=1, strength=2.0), update=10)  # teacher p ~ .71 < .95
    assert captured and all((w == 0).all() for _, w in captured)
    with pytest.raises(TypeError):
        _step(cfg, _Student(), None, update=10)  # teacher mode needs the frozen teacher


def test_old_split_configs_still_validate() -> None:
    validate_config(json.loads((ROOT / "configs/candidate_pg/pg_k8_base.json").read_text()))
    for path in sorted((ROOT / "configs/candidate_pg/w2s").glob("w2s_*.json")):
        validate_config(json.loads(path.read_text()))


def test_a4sa_config_differs_from_old_a4sa_only_in_split_selection_and_teacher_threshold(prepared) -> None:
    import cbkd_sa_make_config as old

    out, lock_path = prepared

    def flat(x, p=""):
        if isinstance(x, dict):
            r = {}
            for k, v in x.items():
                r.update(flat(v, f"{p}/{k}"))
            return r
        return {p: json.dumps(x)}

    new = flat(_config(out, lock_path, "A4SA"))
    base = flat(old.build(out, ROOT / "reports/pg-server-20261116/PG_K8_CONFIG.json"))
    diff = sorted(k for k in set(new) | set(base) if new.get(k) != base.get(k))
    allowed = ("/data/", "/training/selection", "/training/eval_splits", "/teacher/select", "/teacher/semi/threshold/clip",
               "/cinema_init/hf_revision")  # hf_revision only pins the CineMA download; S1 is reused, never retrained
    assert all(k.startswith(allowed) for k in diff), [k for k in diff if not k.startswith(allowed)]
    assert json.loads(new["/cbkd/variant"]) == "cbkd_sa" and json.loads(new["/pg/enabled"]) is True and "/pg/target_mode" not in new  # default fusion target, as the old A4-SA recipe
    assert json.loads(new["/teacher/semi/threshold/clip"]) == [0.95, 0.95]
    assert [k for k in diff if k.startswith("/teacher/semi/threshold/clip")]
