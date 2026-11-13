from types import SimpleNamespace
from pathlib import Path

import torch
from torch import nn
import pytest

from self_audit.losses import supervised_loss
from self_audit.candidate_fuse.config import default_config
from self_audit.candidate_fuse.engine import _checkpoint_due, _lambda_u, _soft_ce, joint_loss


_GEOMETRY = {"shape": (2, 4, 4), "spacing_zyx": (1., 1., 1.),
             "affine_zyx": torch.eye(4).tolist(), "context": "full_volume"}


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.called_no_grad = False
        self.shapes = []

    def forward(self, image, geometry, *, return_features=True):
        self.called_no_grad = not torch.is_grad_enabled()
        self.shapes.append(tuple(image.shape[-3:]))
        logits = self.bias * torch.ones((1, 4, *image.shape[-3:]), device=image.device)
        features = tuple(torch.zeros((1, 1, *image.shape[-3:]), device=image.device) for _ in range(5))
        return {"region_logits": logits, "features": features if return_features else None}


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.2))
        self.shapes = []
        self.keep_values = []

    def forward(self, image, geometry, *, teacher_features=None, keep_mask=None):
        self.shapes.append(tuple(image.shape[-3:]))
        self.keep_values.append(float(keep_mask.detach().item()))
        logits = self.bias * torch.ones((1, 4, *image.shape[-3:]), device=image.device)
        return {"logits": logits, "mimic_loss": self.bias * 0 + 0.25,
                "gate_means": [0.] * 5}


def _frame(target=None):
    return SimpleNamespace(image=torch.full((1, 1, 2, 4, 4), 0.5), target=target,
                           geometry=dict(_GEOMETRY), patient_id="synthetic", frame_index=0)


def test_joint_loss_exact_coefficients_ramp_and_frozen_teacher():
    cfg = default_config(synthetic=True)
    cfg["training"]["mimic_weight"] = 0.1
    teacher, student = _Teacher().eval(), _Student()
    target = torch.zeros((1, 2, 4, 4), dtype=torch.long)
    target[:, :, 1:3, 1:3] = 1
    labeled, unlabeled = _frame(target), _frame()
    total = 10
    # The locked ramp is zero before the first completed update, then reaches 1.
    assert _lambda_u(0, total, 0.1) == 0.
    assert _lambda_u(1, total, 0.1) == 1.
    loss, stats = joint_loss(student, teacher, labeled, unlabeled, update=1,
                             total_updates=total, config=cfg,
                             generator=torch.Generator().manual_seed(2))
    expected = (supervised_loss(torch.full((1, 4, 2, 4, 4), 0.2), target)
                + _soft_ce(torch.full((1, 4, 2, 4, 4), 0.2),
                           torch.full((1, 4, 2, 4, 4), 0.3))
                + 0.1 * (0.25 + 0.25))
    torch.testing.assert_close(loss, expected)
    assert stats["lambda_u"] == 1.
    assert teacher.called_no_grad
    loss.backward()
    assert student.bias.grad is not None
    assert teacher.bias.grad is None


def test_joint_loss_does_not_average_supervised_and_unsupervised_terms():
    cfg = default_config(synthetic=True)
    teacher, student = _Teacher().eval(), _Student()
    target = torch.zeros((1, 2, 4, 4), dtype=torch.long)
    labeled, unlabeled = _frame(target), _frame()
    loss, _ = joint_loss(student, teacher, labeled, unlabeled, update=1,
                         total_updates=10, config=cfg,
                         generator=torch.Generator().manual_seed(11))
    ce_dice = supervised_loss(torch.full((1, 4, 2, 4, 4), 0.2), target)
    soft = _soft_ce(torch.full((1, 4, 2, 4, 4), 0.2),
                    torch.full((1, 4, 2, 4, 4), 0.3))
    torch.testing.assert_close(loss, ce_dice + soft + 0.05)


def test_joint_loss_accepts_native_shape_pair_and_samples_independent_masks():
    cfg=default_config(synthetic=True)
    teacher,student=_Teacher().eval(),_Student()
    target=torch.zeros((1,2,4,4),dtype=torch.long)
    labeled=_frame(target)
    unlabeled=SimpleNamespace(image=torch.full((1,1,3,6,5),0.5),target=None,
        geometry={"shape":(3,6,5),"spacing_zyx":(2.,.7,1.3),
                  "affine_zyx":torch.diag(torch.tensor([2.,.7,1.3,1.])).tolist(),"context":"full_volume"},
        patient_id="other",frame_index=1)
    loss,stats=joint_loss(student,teacher,labeled,unlabeled,update=9,total_updates=10,
        config=cfg,generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert teacher.shapes==[(2,4,4),(3,6,5)]
    assert student.shapes==teacher.shapes
    assert student.keep_values==[0.,1.]
    assert stats["keep_fraction_l"]==0. and stats["keep_fraction_u"]==1.


def test_comparison_exports_are_sha_bound_and_deployment_selects_one(tmp_path):
    from self_audit.candidate_fuse.engine import _export_receipts
    from self_audit.candidate_fuse.student import FuseStudent

    class ExportTeacher(nn.Module):
        def __init__(self):
            super().__init__(); self.model=nn.Conv3d(1,1,1)
            self.feature_channels=(1,)*5
            self.feature_phases=(((1,1,1),(0.,0.,0.)),)*5

    cfg=default_config(synthetic=True)
    teacher=ExportTeacher()
    student=FuseStudent(cfg,teacher_channels=teacher.feature_channels,
                        teacher_phases=teacher.feature_phases)
    decision={"mode":"fusion","rule":"student_if_student_fg_dice_ge_fusion_minus_0.01"}
    receipts,clean,deployment=_export_receipts(tmp_path,teacher,student,cfg,
        {"config_sha256":"test"},{"fusion":{"fg_dice":0.6}},decision)
    assert clean is True
    assert set(receipts)=={"teacher","student","fusion"}
    assert deployment["mode"]=="fusion"
    assert deployment["path"]==receipts["fusion"]["path"]
    assert deployment["sha256"]==receipts["fusion"]["sha256"]


def test_production_checkpoint_cadence_and_final_boundary():
    assert not _checkpoint_due(19,8000,synthetic=False)
    assert _checkpoint_due(20,8000,synthetic=False)
    assert not _checkpoint_due(7999,8000,synthetic=False)
    assert _checkpoint_due(8000,8000,synthetic=False)
    assert _checkpoint_due(1,4,synthetic=True)


def test_resume_keeps_committed_joint_pair_cursor(tmp_path, monkeypatch):
    from self_audit.candidate_fuse import data as fuse_data
    from self_audit.candidate_fuse import engine

    class FakeTeacher(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.tensor(0.5))
            self.feature_channels=(1,)*5; self.feature_phases=(((1,1,1),(0.,0.,0.)),)*5
            self.frozen=False
        def forward(self, *args, **kwargs): return {"loss": self.weight.square()}
        def freeze_for_joint(self): self.requires_grad_(False); self.eval(); self.frozen=True
        def train(self, mode=True):
            if self.frozen and mode: mode=False
            return super().train(mode)

    class FakeStudent(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.weight=nn.Parameter(torch.tensor(0.1))

    manifest_dir=tmp_path/"manifests"
    manifests=fuse_data.write_synthetic_manifests(manifest_dir)
    cfg=default_config(synthetic=True)
    for label,name in (("train","train"),("val","development"),("test","test")):
        cfg["data"][f"{label}_manifest"]=manifests[name]["path"]
        cfg["data"][f"{label}_manifest_sha256"]=manifests[name]["sha256"]
    built_teachers=[]
    def make_teacher(*args,**kwargs):
        teacher=FakeTeacher(); built_teachers.append(teacher); return teacher,{"synthetic":True}
    monkeypatch.setattr(engine,"build_teacher",make_teacher)
    monkeypatch.setattr(engine,"FuseStudent",FakeStudent)
    monkeypatch.setattr(engine,"teacher_parameter_groups",lambda t,c:[{"name":"heads","params":list(t.parameters()),"lr":1e-3,"base_lr":1e-3},{"name":"encoder","params":[],"lr":1e-4,"base_lr":1e-4}])
    monkeypatch.setattr(engine,"configure_teacher_step",lambda *a,**k:{})
    monkeypatch.setattr(engine,"teacher_loss",lambda outputs,target,geometry: outputs["loss"])
    monkeypatch.setattr(engine,"evaluate_modes",lambda teacher,student,data,**kw:
                        {m:{"fg_dice":0.5,"per_class":{},"patient_count":1} for m in kw["modes"]})
    monkeypatch.setattr(engine,"_attach_gate_means",lambda *a,**k:None)
    monkeypatch.setattr(engine,"_export_receipts",lambda *a,**k:({
        mode:{"path":f"/tmp/{mode}.pt","sha256":"a"*64,"mode":mode}
        for mode in ("teacher","student","fusion")},True,
        {"mode":"student","path":"/tmp/student.pt","sha256":"a"*64,"rule":"locked"}))
    loaded_u=[]
    original_load=fuse_data.FuseData.load
    def tracking_load(self,key,*,labeled):
        if not labeled: loaded_u.append(tuple(key))
        return original_load(self,key,labeled=labeled)
    monkeypatch.setattr(fuse_data.FuseData,"load",tracking_load)
    calls=0
    def interrupted_loss(student,teacher,labeled,unlabeled,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError("simulated interruption")
        return student.weight.square(),{"loss":float(student.weight.detach()),"lambda_u":0.,"gate_means_l":[0.]*5,"gate_means_u":[0.]*5}
    monkeypatch.setattr(engine,"joint_loss",interrupted_loss)
    run_dir=tmp_path/"run"
    with pytest.raises(RuntimeError,match="simulated interruption"):
        engine.run_training(cfg,run_dir,max_updates=2)
    assert len(loaded_u)==2  # second attempted pair failed before optimizer commit
    assert all(p.grad is None and not p.requires_grad for p in built_teachers[0].parameters())
    checkpoint=torch.load(run_dir/"resume.pt",map_location="cpu",weights_only=False)
    assert "teacher" not in checkpoint and "teacher_optimizer" not in checkpoint
    assert checkpoint["teacher_artifact"]["sha256"]
    assert checkpoint["metrics_receipt"]["offset"]>0
    import json
    failed_state=json.loads((run_dir/"STATE.json").read_text())
    assert (failed_state["status"],failed_state["stage"],failed_state["update"]) == ("FAILED","J",1)
    teacher_ref=checkpoint["teacher_artifact"]
    teacher_path=Path(teacher_ref["path"])
    meta_path=Path(str(teacher_path)+".meta.json")
    original_meta=meta_path.read_bytes()
    meta_value=json.loads(original_meta); meta_value["status"]="PROFILE"
    meta_path.write_text(json.dumps(meta_value))
    with pytest.raises(ValueError,match="PROFILE"):
        engine._load_teacher_reference(teacher_ref,checkpoint["identity"],FakeTeacher(),torch.device("cpu"),profile=False)
    meta_path.write_bytes(original_meta)
    original_teacher=teacher_path.read_bytes()
    teacher_path.write_bytes(original_teacher[:-1]+bytes([original_teacher[-1]^1]))
    with pytest.raises(ValueError,match="SHA"):
        engine._load_teacher_reference(teacher_ref,checkpoint["identity"],FakeTeacher(),torch.device("cpu"),profile=False)
    teacher_path.write_bytes(original_teacher)
    with (run_dir/"metrics.jsonl").open("ab") as stream:
        stream.write(b'{"partial":"uncommitted suffix"}\n')
    restored=fuse_data.FuseSampler(fuse_data.FuseData(manifests["train"]["path"],manifests["train"]["sha256"],role="train",synthetic=True),seed=42)
    restored.load_state_dict(checkpoint["sampler"])
    expected_second=restored.next_pair()[1]
    monkeypatch.setattr(engine,"joint_loss",lambda student,teacher,labeled,unlabeled,**kwargs:
                        (student.weight.square(),{"loss":float(student.weight.detach()),"lambda_u":0.,"gate_means_l":[0.]*5,"gate_means_u":[0.]*5}))
    result=engine.run_training(cfg,run_dir,resume=True,max_updates=2,
                               teacher_artifact=teacher_path)
    assert result["status"]=="COMPLETED"
    assert result["teacher_has_no_grads_after_J"] is True
    assert result["export_clean"] is True
    assert set(result["exports"])=={"teacher","student","fusion"}
    assert result["deployment"]["mode"]=="student"
    assert result["export_path"]==result["deployment"]["path"]
    assert loaded_u[2]==expected_second==loaded_u[1]
    assert len(loaded_u)==3
    log=(run_dir/"metrics.jsonl").read_bytes()
    assert b"uncommitted suffix" not in log
    archives=list(run_dir.glob("metrics.uncommitted.*.jsonl"))
    assert len(archives)==1 and b"uncommitted suffix" in archives[0].read_bytes()
    assert (run_dir/"RESULT.json").is_file()
    t_only=engine.run_training(cfg,tmp_path/"t-only",stage="T",max_updates=2)
    t_state=json.loads((tmp_path/"t-only"/"STATE.json").read_text())
    assert t_only["status"]=="PAUSED" and t_state["status"]=="COMPLETED" and t_state["stage"]=="T"
    assert (tmp_path/"t-only"/"RESULT.json").is_file()
    t_log_path=tmp_path/"t-only"/"metrics.jsonl"
    t_log_before=t_log_path.read_bytes()
    t_resume=engine.run_training(cfg,tmp_path/"t-only",stage="T",resume=True,max_updates=2)
    assert t_resume["status"]=="PAUSED" and t_resume.get("validation") is None
    assert t_log_path.read_bytes()==t_log_before
    transition=engine.run_training(cfg,tmp_path/"t-only",stage="J",resume=True,
        max_updates=2,teacher_artifact=tmp_path/"t-only"/"teacher.pt")
    assert transition["status"]=="COMPLETED" and transition["stage"]=="J" and transition["update"]==2
    stage_rows=[json.loads(line) for line in t_log_path.read_text().splitlines()]
    assert sum(row.get("stage")=="T" and row.get("type")=="validation" for row in stage_rows)==1
    assert sum(row.get("stage")=="J" and row.get("type")=="train_update" for row in stage_rows)==2
