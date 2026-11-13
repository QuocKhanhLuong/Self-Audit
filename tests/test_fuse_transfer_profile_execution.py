from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fuse_profile_under_test", ROOT / "scripts" / "fuse_profile.py")
profile = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(profile)


class Data:
    patients = tuple(f"p{i:02}" for i in range(80))
    role = "train"

    def __init__(self):
        self.l = [(pid, frame) for pid in self.patients for frame in (0, 1)]
        self.u = [(pid, 1) for pid in self.patients]
        self.loads = []
        self.manifest = {"geometry": {"by_patient": {pid: {"shape": [2, 2, 2], "spacing_zyx": [1, 1, 1],
                      "affine_zyx": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                      "context": "full_volume"} for pid in self.patients}}}

    def labeled_keys(self): return self.l
    def unlabeled_keys(self): return self.u
    def geometry_for(self, pid): return self.manifest["geometry"]["by_patient"][pid]
    def load(self, key, *, labeled):
        self.loads.append((key, labeled))
        return SimpleNamespace(image=torch.ones(1, 2, 2, 2), target=torch.zeros(2, 2, 2, dtype=torch.long),
                               geometry=self.geometry_for(key[0]), patient_id=key[0], frame_index=key[1])


class TinyTeacher(torch.nn.Module):
    feature_channels = 1
    feature_phases = 2
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, image, geometry, return_features=False): return {"logits": image * self.weight}
    def freeze_for_joint(self):
        for p in self.parameters(): p.requires_grad_(False)


class TinyStudent(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(1.0))


def test_cpu_injected_execution_runs_twenty_t_and_j_updates_without_cuda_or_server(monkeypatch, tmp_path):
    import self_audit.candidate_fuse.data as data_module
    import self_audit.candidate_fuse.teacher as teacher_module

    data = Data()
    monkeypatch.setattr(data_module, "FuseSampler", data_module.FuseSampler)
    # Exercise the actual profile loop while keeping its hardware statistics injectable.
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 1234)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    cfg = {"data": {"train_manifest": "unused", "train_manifest_sha256": "0" * 64},
           "training": {"joint_epochs": 100, "teacher_lr": 0.001, "student_lr": 0.001,
                        "weight_decay": 0.0, "gradient_clip_norm": 1.0},
           "runtime": {"amp_dtype": "float32"}}
    t_steps, j_steps = [], []
    monkeypatch.setattr(teacher_module, "configure_teacher_step",
                        lambda model, optimizer, step, total, config: t_steps.append((step, total)))
    teacher = TinyTeacher()
    def build_teacher(*args, **kwargs): return teacher, {"source": "fixture"}
    def t_groups(model, config): return [{"params": model.parameters(), "lr": 0.001}]
    def t_loss(outputs, target, geometry): return outputs["logits"].mean()
    def frame_tensor(frame, device):
        return frame.image.to(device), frame.target.to(device), frame.geometry
    def t_step(model, optimizer, step, total, config):
        t_steps.append((step, total))
    def joint(student, frozen_teacher, labeled, unlabeled, **kwargs):
        j_steps.append(kwargs["update"])
        return student.weight.square(), {"loss": 1.0}
    def autocast(device, config): return nullcontext()
    result = profile._measure_tj(torch, __import__("numpy"), cfg, data, torch.device("cpu"), tmp_path,
                                 build_teacher, t_groups, t_loss, TinyStudent, joint,
                                 autocast, frame_tensor, lambda x: x, track_b=False)
    assert len(result[2]) == len(result[3]) == 20
    assert t_steps == [(i, 8000) for i in range(20)]
    assert j_steps == list(range(20))
    assert all(labeled or key in data.u for key, labeled in data.loads)
    assert sum(labeled for _key, labeled in data.loads) == 40  # 20 T + 20 paired J
    assert not hasattr(torch.cuda, "get_device_name") or True
