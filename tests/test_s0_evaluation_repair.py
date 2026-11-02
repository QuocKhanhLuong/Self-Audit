import ast
import pytest
from scripts.resume_server_s0_repaired_eval import OLD, repaired_bytes


@pytest.mark.parametrize("provenance,expected", [(None, "current"), ("native cine", "current"), ({"parent_manifest_sha256": "parent"}, "parent")])
def test_repair_changes_only_provenance_metadata(provenance, expected):
    original = ('def evaluate(manifest_data):\n    metrics = {"dice": 0.8}\n'
                '    actual_manifest_sha = "current"\n' + OLD + '    return metrics\n').encode()
    fixed = repaired_bytes(original)
    ast.parse(fixed)
    namespace = {}
    exec(compile(fixed, "<reviewed-fixture>", "exec"), namespace)
    assert namespace["evaluate"]({"geometry_provenance": provenance}) == {
        "dice": 0.8, "evaluation_source_manifest_sha256": expected,
    }
    assert original.decode().count(OLD) == 1


@pytest.mark.parametrize("original", [b"unrecognized source", (OLD + OLD).encode()])
def test_repair_refuses_unknown_or_duplicate_sites(original):
    with pytest.raises(ValueError, match="exactly one"):
        repaired_bytes(original)
