import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from self_audit import modal_data_prep as prep
from self_audit.asset_staging import atomic_json, sha256


def _write_fixture(tmp_path, *, placeholder=True, placeholder_pid=None):
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    raw.mkdir()
    prepared.mkdir()
    train_ids = list(prep.L8) + [f"u{i:02d}" for i in range(72)]
    val_ids = [f"v{i:02d}" for i in range(18)]
    split = {"train": train_ids, "val": val_ids}
    references = set(prep.L8 + val_ids)
    cine_header_affine = np.diag([1.25, 1.5, 5.0, 1.0])
    labeled, unlabeled, dev_rows, historical_val_rows = [], [], [], []
    placeholder_pid = placeholder_pid or val_ids[0]
    placeholder_affine = np.asarray(prep.HISTORICAL_FRAME_PLACEHOLDER_AFFINE)

    for index, pid in enumerate(train_ids + val_ids):
        folder = raw / pid
        folder.mkdir()
        cine_values = (np.arange(36, dtype=np.int16).reshape(2, 3, 2, 3) + index * 40)
        cine_image = nib.Nifti1Image(cine_values, cine_header_affine)
        # Match the recovered raw cine header: no coded form, so nibabel uses
        # its NIfTI base affine with the source voxel spacings.
        cine_image.set_sform(None, code=0)
        cine_image.set_qform(None, code=0)
        cine_path = folder / f"{pid}_4d.nii"
        nib.save(cine_image, cine_path)
        loaded_cine = nib.load(cine_path)

        image_path = prepared / f"{pid}.npy"
        image = np.linspace(0, 1, 36, dtype=np.float32).reshape((3, 2, 3, 2))
        np.save(image_path, image, allow_pickle=False)
        rec = {"id": pid, "image": str(image_path), "image_sha256": sha256(image_path)}

        if pid in references:
            masks = {}
            for frame in (0, 2):
                frame_affine = loaded_cine.affine
                if placeholder and pid == placeholder_pid and frame == 0:
                    frame_affine = placeholder_affine
                frame_image = nib.Nifti1Image(cine_values[..., frame], frame_affine)
                frame_path = folder / f"{pid}_frame{frame + 1:02d}.nii"
                nib.save(frame_image, frame_path)

                mask_path = prepared / f"{pid}_t{frame:03d}_mask.npy"
                mask = np.full((2, 3, 2), frame % 4, dtype=np.uint8)
                np.save(mask_path, mask, allow_pickle=False)
                masks[str(frame)] = {"path": str(mask_path), "sha256": sha256(mask_path)}
            rec["masks"] = masks

        if pid in prep.L8:
            labeled.append(rec)
        elif pid in train_ids:
            unlabeled.append(rec)
        else:
            dev_rows.append(rec)
            hist = copy.deepcopy(rec)
            hist.update(
                affine=loaded_cine.affine.tolist(),
                axes=["T", "Z", "Y", "X"],
                spatial_units="mm",
                geometry_source_image={"path": str(cine_path), "sha256": sha256(cine_path)},
            )
            historical_val_rows.append(hist)

    train_manifest = {"schema": "fixture.train.v1", "labeled": labeled, "unlabeled": unlabeled}
    development = {"schema": "fixture.development.v1", "patients": dev_rows}
    train_path = prepared / "train.json"
    development_path = prepared / "development.json"
    atomic_json(train_path, train_manifest)
    atomic_json(development_path, development)
    historical_train_path = tmp_path / "historical_train.json"
    atomic_json(historical_train_path, train_manifest)
    historical_val_path = tmp_path / "historical_val.json"
    atomic_json(historical_val_path, {
        "schema": "fixture.historical_val.v1",
        "patients": historical_val_rows,
    })
    return {
        "raw": raw,
        "prepared": prepared,
        "split": split,
        "historical_train": historical_train_path,
        "historical_val": historical_val_path,
        "placeholder_pid": placeholder_pid,
    }


def _pin_fixture_history(monkeypatch, fixture):
    monkeypatch.setattr(prep, "HISTORICAL_TRAIN_SHA", sha256(fixture["historical_train"]))
    monkeypatch.setattr(prep, "HISTORICAL_VAL_SHA", sha256(fixture["historical_val"]))
    compare = prep.manifest_comparison
    monkeypatch.setattr(
        prep,
        "manifest_comparison",
        lambda path: compare(path, fixture["historical_train"]),
    )


def _validate(fixture, *, compatibility=False):
    return prep.validate_prepared(
        fixture["raw"],
        fixture["prepared"],
        fixture["split"],
        fixture["historical_val"],
        allow_historical_placeholder_geometry=compatibility,
    )


def test_placeholder_geometry_requires_opt_in_and_emits_bounded_receipt(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _pin_fixture_history(monkeypatch, fixture)
    opened = []
    original_load = nib.load

    def tracked_load(path, *args, **kwargs):
        opened.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", tracked_load)

    with pytest.raises(ValueError, match="compatibility is opt-in"):
        _validate(fixture)

    opened.clear()
    receipt = _validate(fixture, compatibility=True)
    raw_frame_ids = {
        path.parent.name for path in opened if "_frame" in path.name
    }
    assert raw_frame_ids == set(prep.L8 + [f"v{i:02d}" for i in range(18)])
    assert all("_gt" not in path.name for path in opened)
    train = json.loads((fixture["prepared"] / "train.json").read_text())
    assert all("masks" not in row for row in train["unlabeled"])

    compatibility = receipt["geometry_compatibility"]
    assert compatibility["mode"] == "historical_placeholder_with_exact_voxel_bridge"
    assert compatibility["placeholder_frame_count"] == 1
    assert compatibility["historical_train_semantic_proof"]["proven"] is True
    assert compatibility["all_18_val_image_mask_raw_affine_hashes_match"] is True
    assert "not physical-space evidence" in compatibility["placeholder_world_coordinate_limitation"]

    row = next(item for item in receipt["geometry"] if item["id"] == fixture["placeholder_pid"])
    bridge = next(item for item in row["frames"] if item["frame"] == 0)
    assert bridge["frame_affine"] == [list(r) for r in prep.HISTORICAL_FRAME_PLACEHOLDER_AFFINE]
    assert bridge["frame_forms"]["sform"]["code"] == 2
    assert bridge["frame_forms"]["qform"]["code"] == 0
    assert bridge["cine_forms"]["sform"]["code"] == 0
    assert bridge["cine_forms"]["qform"]["code"] == 0
    assert bridge["exact_array_bridge"] is True
    assert bridge["world_coordinate_limitation"]
    val_manifest = json.loads((fixture["prepared"] / "val18_fresh_native.json").read_text())
    val_row = next(item for item in val_manifest["patients"] if item["id"] == fixture["placeholder_pid"])
    assert val_row["affine"] == row["affine"]  # raw cine affine stays authoritative
    assert val_row["frame_geometry_bridge"][0]["frame_affine"] == bridge["frame_affine"]


@pytest.mark.parametrize("case", ["arbitrary_affine", "mismatched_voxels"])
def test_opt_in_rejects_nonmatching_affine_or_voxel_bridge(tmp_path, monkeypatch, case):
    fixture = _write_fixture(tmp_path)
    _pin_fixture_history(monkeypatch, fixture)
    raw_frame = fixture["raw"] / fixture["placeholder_pid"] / f"{fixture['placeholder_pid']}_frame01.nii"
    frame = nib.load(raw_frame)
    data = np.asarray(frame.dataobj).copy()

    if case == "arbitrary_affine":
        affine = np.asarray(prep.HISTORICAL_FRAME_PLACEHOLDER_AFFINE).copy()
        affine[0, 0] = -2.0
        replacement = nib.Nifti1Image(data, affine)
        nib.save(replacement, raw_frame)
        with pytest.raises(ValueError, match="outside the exact historical placeholder signature"):
            _validate(fixture, compatibility=True)
    else:
        data[0, 0, 0] += 1
        replacement = nib.Nifti1Image(data, prep.HISTORICAL_FRAME_PLACEHOLDER_AFFINE)
        nib.save(replacement, raw_frame)
        with pytest.raises(ValueError, match="raw frame voxels differ"):
            _validate(fixture, compatibility=True)


def test_opt_in_rejects_cine_with_altered_coded_form_even_when_voxels_match(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path, placeholder_pid=prep.L8[0])
    _pin_fixture_history(monkeypatch, fixture)
    cine_path = fixture["raw"] / fixture["placeholder_pid"] / f"{fixture['placeholder_pid']}_4d.nii"
    original_cine = nib.load(cine_path)
    cine_voxels = np.asarray(original_cine.dataobj).copy()
    assert np.array_equal(
        np.asarray(nib.load(fixture["raw"] / fixture["placeholder_pid"] /
                            f"{fixture['placeholder_pid']}_frame01.nii").dataobj),
        cine_voxels[..., 0],
    )
    coded_cine = nib.Nifti1Image(cine_voxels, original_cine.affine)
    assert int(coded_cine.header["sform_code"]) == 2
    assert int(coded_cine.header["qform_code"]) == 0
    nib.save(coded_cine, cine_path)

    with pytest.raises(ValueError, match="outside the exact historical placeholder signature"):
        _validate(fixture, compatibility=True)


@pytest.mark.parametrize("proof_state", ["missing", "changed"])
def test_opt_in_rejects_missing_or_changed_historical_train_proof(tmp_path, monkeypatch, proof_state):
    fixture = _write_fixture(tmp_path)
    _pin_fixture_history(monkeypatch, fixture)
    if proof_state == "missing":
        monkeypatch.setattr(prep, "manifest_comparison", lambda path: {
            "actual_sha256": sha256(path),
            "expected_sha256": prep.HISTORICAL_TRAIN_SHA,
            "exact_bytes": False,
            "path_only_witness": None,
        })
    else:
        path = fixture["prepared"] / "train.json"
        manifest = json.loads(path.read_text())
        manifest["semantic_marker"] = "changed historical payload"
        atomic_json(path, manifest)

    with pytest.raises(ValueError, match="requires an exact historical TRAIN semantic proof"):
        _validate(fixture, compatibility=True)


def test_opt_in_rejects_historical_val_hash_or_affine_mismatch(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    _pin_fixture_history(monkeypatch, fixture)
    path = fixture["historical_val"]
    historical = json.loads(path.read_text())
    historical["patients"][0]["affine"][0][3] += 0.25
    atomic_json(path, historical)
    monkeypatch.setattr(prep, "HISTORICAL_VAL_SHA", sha256(path))

    with pytest.raises(AssertionError):
        _validate(fixture, compatibility=True)
