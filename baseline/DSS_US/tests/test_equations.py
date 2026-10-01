"""Independent equation tests; synthetic parameter choices are not CAMUS recipes."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from dss_us.affinity import positive_feature_affinity, ssd_affinity, mi_affinity, combine_affinities
from dss_us.spectral import normalized_laplacian
from dss_us.step1 import oversegment
from dss_us.step2 import semantic_cluster
from dss_us.segment_features import combine_segment_features
from dss_us.preprocessing import imagenet_tensor
from dss_us.features import DinoKeys
from shared_benchmark.native_protocol import ProtocolBlocked


def test_published_positive_affinity_and_laplacian():
    features = np.array([[1., 0.], [-1., 0.], [0., 1.]])
    assert np.array_equal(positive_feature_affinity(features, normalize=False), np.eye(3))
    weights = np.array([[1., 1.], [1., 1.]])
    laplacian = normalized_laplacian(weights)
    assert np.allclose(laplacian, [[.5, -.5], [-.5, .5]])
    assert np.allclose(np.linalg.eigvalsh(laplacian), [0, 1])


def test_ssd_and_paper_mi_definition_not_repaired():
    assert np.allclose(ssd_affinity(np.array([[0], [2]]), delta=.5), [[1, np.exp(-2)], [np.exp(-2), 1]])
    patches = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    result = mi_affinity(patches, delta=1, bins=2, intensity_range=(0, 2))
    assert np.allclose(np.diag(result), np.e)  # Hx+Hy / Hxy = 2 for identical nonconstant patches
    assert np.allclose(result[0, 1], 1)
    with pytest.raises(ValueError, match="zero joint entropy"):
        mi_affinity(np.zeros((2, 4)), delta=1, bins=2, intensity_range=(0, 2))


def test_no_silent_directed_graph_or_degree_repair():
    with pytest.raises(ValueError, match="symmetrization"):
        normalized_laplacian(np.array([[1., 2.], [0., 1.]]))
    with pytest.raises(ValueError, match="nonpositive"):
        normalized_laplacian(np.zeros((2, 2)))


def test_two_stage_shapes_fit_inventory_and_repeat():
    weights = np.full((16, 16), .01)
    weights[:8, :8] += 1
    weights[8:, 8:] += 1
    settings = dict(dimensions=2, discard_first=False, normalize_rows=False, clusters=2,
                    seed=1, n_init=10, max_iter=300, tolerance=.0001, algorithm="lloyd")
    partition, _ = oversegment(weights, (4, 4), **settings)
    again, _ = oversegment(weights, (4, 4), **settings)
    assert np.array_equal(partition, again) and len(np.unique(partition)) == 2
    features = np.array([[0., 0.], [1., 1.], [.01, .01], [.99, .99]])
    segments = [(partition, list(np.unique(partition))), (partition, list(np.unique(partition)))]
    outputs, fitted = semantic_cluster(features, segments, clusters=2, seed=1, n_init=10,
                                      max_iter=300, tolerance=.0001, algorithm="lloyd")
    assert all(o.shape == (4, 4) for o in outputs)
    assert len(fitted["segment_assignment"]) == 4


def test_explicit_embedding_and_imagenet_input():
    features = np.ones((2, 3))
    assert np.array_equal(combine_segment_features(features, features * 2, features * 3,
                                                  c_mask=.5, c_position=1), features * 5)
    tensor = imagenet_tensor(np.zeros((8, 8, 3), np.uint8))
    assert tensor.shape == (3, 8, 8) and torch.isfinite(tensor).all()


def test_key_hook_excludes_cls_and_flattens_heads():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(4, 12, bias=False)
            self.num_heads = 2
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].attn = Attention()
        def get_last_selfattention(self, image):
            self.blocks[-1].attn.qkv(torch.ones(1, 5, 4))
    provider = DinoKeys.__new__(DinoKeys)
    provider.model, provider.device = Model(), "cpu"
    keys, grid = provider.keys(torch.zeros(1, 3, 16, 16), crop_to_patch_multiple=False)
    assert grid == (2, 2) and keys.shape == (1, 4, 4)


def test_step2_matching_primitive_covers_both_resolved_branches():
    spec = importlib.util.spec_from_file_location("step2_evaluator", BASE / "evaluation/track_b/step2.py")
    step2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step2)
    predicted = np.array([[8, 8], [3, 3]])
    gt = np.array([[1, 1], [0, 0]])
    reordered, mapping = step2.semantic_match(predicted, gt)
    assert np.array_equal(reordered, gt) and mapping[8] == 1
    # Unequal counts now use the evidenced exclusive majority vote (no longer a missing branch).
    reordered, mapping = step2.semantic_match(np.array([[0, 1, 2]]), np.array([[0, 1, 1]]))
    assert mapping == {0: 0, 1: 1} and reordered.tolist() == [[0, 1, 0]]


def test_step2_evidence_records_defined_source_branch_and_stays_blocked():
    import json
    spec = json.loads((BASE / "evaluation/track_b/spec.json").read_text())
    assert "eval_utils.py:202" in spec["step_II"]["unequal_count_source"]
    assert spec["step_II"]["status"] == "UNRESOLVED/BLOCKED_PROTOCOL" and spec["step_II"]["unresolved"]
    for path in sorted((BASE / "config/native").glob("step2_*.yaml")):
        config = json.loads(path.read_text())
        gates = config["gates"]["native_track_b"]
        assert gates and not any("missing" in gate for gate in gates)
        assert config["evidence"]["step2_evaluator"]["status"] == "OFFICIAL_IMPLEMENTATION_DETAIL"


def test_pinned_reference_defines_majority_vote_exclusive():
    """Ties the ledger claim to the pinned unlicensed source (read as text, never imported)."""
    import subprocess
    reference = ROOT / ".scratch/dss-us"
    if not (reference / ".git").exists():
        pytest.skip("run scripts/pin_native_references.py for the source-evidence check")
    head = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    assert head == "d4ac44c60df18b921c590796f6994a4c8ac0726c"
    lines = (reference / "evaluation/eval_utils.py").read_text().splitlines()
    assert lines[201].startswith("def majority_vote_exclusive(")
    assert "majority_vote_exclusive(" in lines[171]
