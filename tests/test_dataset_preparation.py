import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from prepare_mnms import prepare as prepare_mnms
from prepare_acdc import prepare as prepare_acdc
from self_audit.data import TrainingData
from self_audit.model import digest
from self_audit.preparation import affine_in_mm, native_geometry_zyx, parse_class_map, remap_mask, spatial_unit


def test_original_mnms_source_map_is_explicit_and_remapped():
    source = parse_class_map('{"0":"background","1":"LV","2":"MYO","3":"RV"}')
    source_mask = np.array([[[0, 1], [2, 3]]], dtype=np.int16)
    assert remap_mask(source_mask, source).tolist() == [[[0, 3], [2, 1]]]
    with pytest.raises(ValueError, match="source class map"):
        parse_class_map('{"0":"background","1":"LV"}')


def test_mnms_index_preparation_reads_only_selected_and_development_masks(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    root.mkdir()
    registry = {}
    reads = []

    class Image:
        def __init__(self, data):
            self.dataobj = data
            self.shape = data.shape
            self.affine = np.diag([1.1, 1.2, 5.0, 1.0])
            self.header = types.SimpleNamespace(get_xyzt_units=lambda: ("mm", None))

    rng = np.random.default_rng(3)
    rows = []
    for idx, split in enumerate(("train", "train", "train", "val", "test"), 1):
        pid = f"P{idx:03d}"
        image = rng.normal(size=(4, 5, 2, 3)).astype(np.float32)
        ip = root / f"{pid}.nii.gz"
        ip.write_bytes(f"image-{pid}".encode())
        registry[str(ip.resolve())] = Image(image)
        fields = {"patient_id": pid, "split": split, "image": ip.name,
                  "ed_frame": "0", "es_frame": "2", "axis_order": "XYZT",
                  "mask_axis_order": "XYZ", "mask_ed": "", "mask_es": "", "vendor": "vendor_A"}
        if split in {"train", "val"}:
            for phase in ("ed", "es"):
                mp = root / f"{pid}_{phase}.nii.gz"
                mask = np.zeros((4, 5, 2), dtype=np.int16)
                mask[1:3, 1:3, :] = 1       # source LV
                mask[2:4, 2:4, :] = 3       # source RV
                mp.write_bytes(f"mask-{pid}-{phase}".encode())
                registry[str(mp.resolve())] = Image(mask)
                fields[f"mask_{phase}"] = mp.name
        rows.append(fields)
    index = tmp_path / "mnms.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def load(path):
        resolved = str(Path(path).resolve())
        reads.append(resolved)
        return registry[resolved]

    monkeypatch.setitem(sys.modules, "nibabel", types.SimpleNamespace(load=load))
    out = tmp_path / "prepared"
    prepare_mnms(index, root, out, labels=2, seed=9,
                 source_class_map='{"0":"background","1":"LV","2":"MYO","3":"RV"}',
                 frame_base=0, evaluation=True, dataset_release="original_2020")
    train = json.loads((out / "train.json").read_text())
    development = json.loads((out / "development.json").read_text())
    assert len(train["labeled"]) == 2
    assert all(set(record) == {"id", "image", "image_sha256", "masks"} for record in train["labeled"])
    assert all(set(record) == {"id", "image", "image_sha256"} for record in train["unlabeled"])
    assert development["patients"][0]["id"] == "P004"
    assert development["patient_metadata"]["P004"]["vendor"] == "vendor_A"
    assert set(train["patient_metadata"]) == {"P001", "P002", "P003"}
    assert not any("P005" in path for path in reads)
    remapped = np.load(out / train["labeled"][0]["masks"]["0"]["path"])
    assert 3 in remapped and 1 in remapped
    assert train["class_mapping"] == {"0": "background", "1": "RV", "2": "MYO", "3": "LV", "255": "ignore"}


def test_training_data_accepts_fully_labeled_cohort(tmp_path):
    image = np.zeros((1, 1, 4, 4), dtype=np.float32)
    image_b = np.ones((1, 1, 4, 4), dtype=np.float32)
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    ip, ip_b, mp = tmp_path / "p.npy", tmp_path / "p_b.npy", tmp_path / "m.npy"
    np.save(ip, image)
    np.save(ip_b, image_b)
    np.save(mp, mask)
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({
        "schema": "self_audit.data.v2", "role": "training", "synthetic": True,
        "label_budget": 2,
        "labeled": [{"id": "a", "image": ip.name, "image_sha256": digest(ip),
                     "masks": {"0": {"path": mp.name, "sha256": digest(mp)}}},
                    {"id": "b", "image": ip_b.name, "image_sha256": digest(ip_b),
                     "masks": {"0": {"path": mp.name, "sha256": digest(mp)}}}],
        "unlabeled": [], "evaluation_ids": []}))
    data = TrainingData(manifest, digest(manifest), synthetic=True)
    assert data.uids == [] and data.lids == ["a", "b"]


def test_geometry_units_are_explicit_and_converted_to_mm():
    class Image:
        def __init__(self, units):
            self.header = types.SimpleNamespace(get_xyzt_units=lambda: (units, None))

    image = Image("m")
    assert spatial_unit(image) == "m"
    affine_m = np.diag([0.001, 0.002, 0.005, 1.0])
    geometry = native_geometry_zyx(affine_in_mm(affine_m, spatial_unit(image)), (2, 4, 5))
    assert geometry["spacing_zyx_mm"] == [5.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="unknown spatial unit"):
        spatial_unit(Image("furlong"))
    with pytest.raises(ValueError, match="sheared"):
        native_geometry_zyx(np.array([[1., .1, 0., 0.], [0., 1., 0., 0.],
                                      [0., 0., 1., 0.], [0., 0., 0., 1.]]), (2, 4, 5))


def _write_nifti(nib, path, data, affine, *, spatial_units="mm"):
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=np.float64))
    image.header.set_xyzt_units(xyz=spatial_units, t="sec")
    nib.save(image, str(path))


def _real_acdc_fixture(nib, root):
    affine = np.diag([1.5, 2.0, 4.0, 1.0])
    shape = (4, 5, 2, 2)  # source NIfTI order X,Y,Z,T
    for index, split in enumerate(("train", "train", "val", "test"), 1):
        pid = f"patient{index:03d}"
        folder = root / pid
        folder.mkdir()
        cine = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index
        cine[0, 0, 0, 0] = -10.0 * index
        cine[-1, -1, -1, -1] = 1000.0 + 10.0 * index
        _write_nifti(nib, folder / f"{pid}_4d.nii.gz", cine, affine)
        for frame in (0, 1):
            frame_image = cine[..., frame]
            mask = np.zeros(frame_image.shape, dtype=np.uint8)
            mask[1:3, 1:4, :] = 1  # canonical RV
            mask[2:4, 2:4, :] = 3  # canonical LV
            _write_nifti(nib, folder / f"{pid}_frame{frame + 1:02d}.nii.gz", frame_image, affine)
            _write_nifti(nib, folder / f"{pid}_frame{frame + 1:02d}_gt.nii.gz", mask, affine)
    split = root.parent / "acdc_split.json"
    split.write_text(json.dumps({
        "train": ["patient001", "patient002"],
        "val": ["patient003"],
        "test": ["patient004"],
    }))
    return split


def _real_mnms_fixture(nib, root):
    affine = np.diag([1.5, 2.0, 4.0, 1.0])
    shape = (4, 5, 2, 2)  # source NIfTI order X,Y,Z,T
    rows = []
    for index, split in enumerate(("train", "train", "val", "test"), 1):
        pid = f"M{index:03d}"
        cine = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + index
        cine[0, 0, 0, 0] = -10.0 * index
        cine[-1, -1, -1, -1] = 1000.0 + 10.0 * index
        image_path = root / f"{pid}.nii.gz"
        _write_nifti(nib, image_path, cine, affine)
        row = {
            "patient_id": pid,
            "split": split,
            "image": image_path.name,
            "ed_frame": "0",
            "es_frame": "1",
            "axis_order": "XYZT",
            "mask_axis_order": "XYZ",
            "mask_ed": "",
            "mask_es": "",
        }
        if split in {"train", "val"}:
            for phase in ("ed", "es"):
                mask = np.zeros(shape[:3], dtype=np.uint8)
                mask[1:3, 1:4, :] = 1  # original M&Ms LV
                mask[2:4, 2:4, :] = 3  # original M&Ms RV
                mask_path = root / f"{pid}_{phase}.nii.gz"
                _write_nifti(nib, mask_path, mask, affine)
                row[f"mask_{phase}"] = mask_path.name
        rows.append(row)
    index = root.parent / "mnms_index.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return index


@pytest.mark.parametrize("dataset", ("acdc", "mnms"))
def test_real_nifti_preparation_roundtrip_and_split_isolation(dataset, tmp_path):
    """Exercise actual nibabel read/write, native geometry, and two phases.

    This is a parser/preparation smoke test over manufactured NIfTI files.  It
    is not patient, clinical, or GPU evidence.
    """
    nib = pytest.importorskip("nibabel")
    root = tmp_path / dataset
    root.mkdir()
    if dataset == "acdc":
        split = _real_acdc_fixture(nib, root)
        out = tmp_path / "prepared_acdc"
        prepare_acdc(root, split, out, budget=2, seed=17, evaluation=True)
        expected_ids = {"patient001", "patient002"}
        held_out = "patient004"
    else:
        index = _real_mnms_fixture(nib, root)
        out = tmp_path / "prepared_mnms"
        prepare_mnms(
            index, root, out, labels=2, seed=17,
            source_class_map='{"0":"background","1":"LV","2":"MYO","3":"RV"}',
            frame_base=0, evaluation=True, dataset_release="original_2020",
        )
        expected_ids = {"M001", "M002"}
        held_out = "M004"

    manifest = json.loads((out / "train.json").read_text())
    prep = json.loads((out / "PREPARATION.json").read_text())
    data = TrainingData(out / "train.json", digest(out / "train.json"), synthetic=False)
    assert set(data.lids) == expected_ids
    assert data.uids == []
    for patient_id in expected_ids:
        geometry = data.geometry_for(patient_id, required=True)
        assert geometry["spacing_zyx_mm"] == [4.0, 2.0, 1.5]
        record = next(r for r in manifest["labeled"] if r["id"] == patient_id)
        assert sorted(map(int, record["masks"])) == [0, 1]
        for mask_record in record["masks"].values():
            mask = np.load(out / mask_record["path"])
            assert mask.shape == (2, 5, 4)  # cached Z,Y,X
            assert set(np.unique(mask)).issubset({0, 1, 2, 3, 255})
        if dataset == "mnms":
            mask = np.load(out / next(iter(record["masks"].values()))["path"])
            assert 1 in mask and 3 in mask  # RV/LV after source remapping
    assert held_out not in prep["raw_images"]
    assert held_out not in prep["read_mask_patient_ids"]
    assert prep["unlabeled_mask_reads"] == 0


def test_real_nifti_units_and_non_grid_mask_fail_closed(tmp_path):
    """Header units are converted explicitly; unknown/sheared files fail."""
    nib = pytest.importorskip("nibabel")
    affine_m = np.diag([0.0015, 0.002, 0.004, 1.0])
    image_path = tmp_path / "units_m.nii.gz"
    _write_nifti(nib, image_path, np.zeros((4, 5, 2, 2), dtype=np.float32), affine_m,
                 spatial_units="meter")
    loaded = nib.load(str(image_path))
    assert spatial_unit(loaded) == "m"
    geometry = native_geometry_zyx(affine_in_mm(loaded.affine, spatial_unit(loaded)), (2, 5, 4))
    np.testing.assert_allclose(geometry["spacing_zyx_mm"], [4.0, 2.0, 1.5], rtol=1e-6, atol=1e-6)

    unknown_path = tmp_path / "units_unknown.nii.gz"
    unknown = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    unknown.header["xyzt_units"] = 0
    nib.save(unknown, str(unknown_path))
    with pytest.raises(ValueError, match="unknown spatial unit"):
        spatial_unit(nib.load(str(unknown_path)))

    # A mask with a non-grid affine is not silently resampled or accepted.
    root = tmp_path / "mnms_non_grid"
    root.mkdir()
    affine = np.diag([1.5, 2., 4., 1.])
    image = root / "M001.nii.gz"
    image_two = root / "M002.nii.gz"
    _write_nifti(nib, image, np.zeros((4, 5, 2, 2), dtype=np.float32), affine)
    _write_nifti(nib, image_two, np.ones((4, 5, 2, 2), dtype=np.float32), affine)
    mask = root / "M001_ed.nii.gz"
    sheared = affine.copy()
    sheared[0, 1] = 0.25
    _write_nifti(nib, mask, np.zeros((4, 5, 2), dtype=np.uint8), sheared)
    index = tmp_path / "mnms_non_grid.csv"
    fields = [
        "patient_id", "split", "image", "ed_frame", "es_frame",
        "axis_order", "mask_axis_order", "mask_ed", "mask_es",
    ]
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "patient_id": "M001", "split": "train", "image": image.name,
            "ed_frame": "0", "es_frame": "1", "axis_order": "XYZT",
            "mask_axis_order": "XYZ", "mask_ed": mask.name, "mask_es": mask.name,
        })
        writer.writerow({
            "patient_id": "M002", "split": "train", "image": image_two.name,
            "ed_frame": "0", "es_frame": "1", "axis_order": "XYZT",
            "mask_axis_order": "XYZ", "mask_ed": mask.name, "mask_es": mask.name,
        })
    with pytest.raises(ValueError, match="affine"):
        prepare_mnms(
            index, root, tmp_path / "prepared_non_grid", labels=2, seed=1,
            source_class_map='{"0":"background","1":"LV","2":"MYO","3":"RV"}',
            frame_base=0, dataset_release="original_2020",
        )
