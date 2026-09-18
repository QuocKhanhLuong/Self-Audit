from __future__ import annotations

import copy

import pytest

from shared_benchmark.firewall import FirewallError
from shared_benchmark.spatial import resolve_source_path

from helpers import discovered_projection, write_image


def test_source_locator_cannot_escape_the_declared_image_only_root(tmp_path):
    write_image(tmp_path, "patient001.npy", seed=1)
    _, manifest = discovered_projection(tmp_path)
    record = copy.deepcopy(manifest["records"][0])
    record["source"]["locator"] = "../outside.npy"
    with pytest.raises(FirewallError, match="escapes declared image-only root"):
        resolve_source_path(record, tmp_path)
