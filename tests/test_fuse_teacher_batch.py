import torch
from torch import nn
import hashlib
import json

from self_audit.candidate_fuse.config import default_config, normalize_config
from self_audit.candidate_fuse.data import FuseData, FuseSampler, write_synthetic_manifests
from self_audit.candidate_fuse.engine import _teacher_batch_loss


def test_teacher_batch_config_is_opt_in_and_locked(tmp_path):
    cfg=default_config()
    assert cfg["training"]["teacher_batch_size"] == 1
    cfg["training"]["teacher_batch_size"] = 2
    assert normalize_config(cfg)["training"]["teacher_batch_size"] == 2
    cfg["training"]["teacher_batch_size"] = 3
    try:
        normalize_config(cfg)
    except ValueError as exc:
        assert "teacher_batch_size" in str(exc)
    else:
        raise AssertionError("unsupported teacher batch size accepted")


def test_same_patient_batches_cover_each_epoch_once_and_resume(tmp_path):
    manifests=write_synthetic_manifests(tmp_path/"manifests")
    data=FuseData(manifests["train"]["path"],manifests["train"]["sha256"],role="train",synthetic=True)
    sampler=FuseSampler(data,seed=42)
    first=sampler.next_labeled_batch(2)
    assert len(first)==2 and first[0][0]==first[1][0]
    restored=FuseSampler(data,seed=42)
    restored.load_state_dict(sampler.state_dict())
    consumed=list(first)
    while sampler.state_dict()["teacher_cursor"] < sampler.teacher_epoch_size:
        consumed.extend(sampler.next_labeled_batch(2))
    expected=list(consumed)
    resumed=list(first)
    while restored.state_dict()["teacher_cursor"] < restored.teacher_epoch_size:
        resumed.extend(restored.next_labeled_batch(2))
    assert resumed==expected
    assert len(resumed)==sampler.teacher_epoch_size
    assert len(set(resumed))==sampler.teacher_epoch_size


def test_teacher_batch_loss_is_mean_of_per_frame_losses_and_gradients():
    weights=torch.tensor(0.7,requires_grad=True)
    prediction=torch.tensor([1.,3.],requires_grad=True)*weights
    target=torch.zeros(2,1,1,1,1)
    def loss_fn(outputs,y,geometry):
        return (outputs["value"]-y.reshape(y.shape[0],-1).mean(1)).square().mean()
    from self_audit.candidate_fuse import engine
    original=engine.teacher_loss
    engine.teacher_loss=loss_fn
    try:
        batched=_teacher_batch_loss({"value":prediction},target,{})
        manual=(loss_fn({"value":prediction[:1]},target[:1],{})+
                loss_fn({"value":prediction[1:]},target[1:],{}))/2
        g_batched=torch.autograd.grad(batched,weights,retain_graph=True)[0]
        g_manual=torch.autograd.grad(manual,weights)[0]
    finally:
        engine.teacher_loss=original
    assert torch.allclose(batched,manual,atol=0,rtol=0)
    assert torch.allclose(g_batched,g_manual,atol=0,rtol=0)


def test_synthetic_teacher_batch_two_updates_two_steps_and_resumes(tmp_path,monkeypatch):
    from self_audit.candidate_fuse import engine

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__(); self.weight=nn.Parameter(torch.tensor(.5))
            self.feature_channels=(1,)*5; self.feature_phases=(((1,1,1),(0.,0.,0.)),)*5
            self.calls=0; self.fail=False; self.batch_sizes=[]
        def forward(self,image,geometry,*,return_features=True):
            self.calls+=1; self.batch_sizes.append(image.shape[0])
            if self.fail and self.calls==2: raise RuntimeError("interrupt after committed batch")
            return {"value":self.weight.expand(image.shape[0])}
        def freeze_for_joint(self): self.requires_grad_(False); self.eval()

    manifests=write_synthetic_manifests(tmp_path/"manifests")
    cfg=default_config(synthetic=True); cfg["training"]["teacher_batch_size"]=2
    for label,name in (("train","train"),("val","development"),("test","test")):
        cfg["data"][f"{label}_manifest"]=manifests[name]["path"]
        cfg["data"][f"{label}_manifest_sha256"]=manifests[name]["sha256"]
    fail_next=[True]; teachers=[]
    def make_teacher(*a,**k):
        t=TinyTeacher(); t.fail=fail_next[0]; teachers.append(t); return t,{"synthetic":True}
    monkeypatch.setattr(engine,"build_teacher",make_teacher)
    monkeypatch.setattr(engine,"teacher_parameter_groups",lambda t,c:[{"name":"heads","params":list(t.parameters()),"lr":1e-3,"base_lr":1e-3},{"name":"encoder","params":[],"lr":1e-4,"base_lr":1e-4}])
    monkeypatch.setattr(engine,"configure_teacher_step",lambda *a,**k:{})
    monkeypatch.setattr(engine,"teacher_loss",lambda outputs,target,geometry:outputs["value"].square().mean())
    monkeypatch.setattr(engine,"evaluate_modes",lambda *a,**k:{"teacher":{"fg_dice":.5,"per_class":{},"patient_count":1}})
    def fake_export(student,teacher,config,path,*,mode,metadata):
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"schema":"self_audit.candidate_fuse.export.v1","mode":"teacher","teacher_state":teacher.state_dict(),"student_state":None},path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest=path.with_suffix(path.suffix+".json")
        manifest.write_text(json.dumps({"artifact_sha256":digest}))
        return {"path":str(path),"sha256":digest,"manifest_path":str(manifest)}
    monkeypatch.setattr(engine,"export_model",fake_export)
    out=tmp_path/"run"
    try:
        engine.run_training(cfg,out,stage="T",max_updates=4)
    except RuntimeError as exc:
        assert "interrupt" in str(exc)
    else:
        raise AssertionError("expected the synthetic interruption")
    resume=torch.load(out/"resume.pt",map_location="cpu",weights_only=False)
    assert resume["update"]==1 and resume["teacher_optimizer_steps"]==1
    assert teachers[0].batch_sizes==[1,2]
    fail_next[0]=False
    result=engine.run_training(cfg,out,stage="T",resume=True,max_updates=4)
    assert result["update"]==4 and result["examples_seen"]==4
    assert result["optimizer_steps"]==3 and result["batch_size"]==2
    assert teachers[1].batch_sizes==[2,1]
