import copy

from self_audit.candidate_fuse.data import write_synthetic_manifests
from self_audit.candidate_pg.config import default_config
from self_audit.candidate_pg.data import PGSampler, open_pg_data


def _config(tmp_path):
    refs = write_synthetic_manifests(tmp_path / "fuse")
    cfg = default_config(synthetic=True)
    cfg["data"]["manifests"] = {
        "train": {"path": refs["train"]["path"], "sha256": refs["train"]["sha256"]},
        "val": {"path": refs["development"]["path"], "sha256": refs["development"]["sha256"]},
        "test": {"path": refs["test"]["path"], "sha256": refs["test"]["sha256"]},
    }
    return cfg


def test_pg_opens_fuse_roles_and_sampler_pairs_different_patients(tmp_path):
    data = open_pg_data(_config(tmp_path))
    sampler = PGSampler(data, seed=42)
    samples = [sampler.next_step() for _ in range(sampler.epoch_size)]
    assert len(samples) == 2
    assert all(u1[0] != u2[0] for _, u1, u2 in samples)
    assert all(data.train.load(key, labeled=False).target is None for _, u1, u2 in samples for key in (u1, u2))


def test_pg_sampler_restore_is_known_answer(tmp_path):
    data = open_pg_data(_config(tmp_path))
    first = PGSampler(data, seed=42)
    first.next_step()
    state = copy.deepcopy(first.state_dict())
    expected = [first.next_step() for _ in range(3)]
    restored = PGSampler(data, seed=42)
    restored.load_state_dict(state)
    assert [restored.next_step() for _ in range(3)] == expected
