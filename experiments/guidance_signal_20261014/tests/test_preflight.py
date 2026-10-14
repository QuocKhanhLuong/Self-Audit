import json
from types import SimpleNamespace
import pytest
from experiments.guidance_signal_20261014.preflight import fresh_output,assert_original_bindings,digest

def test_fresh_output_preserves_frozen_ancestor(tmp_path):
    frozen=tmp_path/"original"
    frozen.mkdir()
    (frozen/"FROZEN.json").write_text("{}")
    with pytest.raises(ValueError,match="frozen"):
        fresh_output(frozen/"new")
    assert list(frozen.iterdir()) == [frozen/"FROZEN.json"]

def test_existing_output_not_overwritten(tmp_path):
    (tmp_path/"keep.txt").write_text("preserve")
    with pytest.raises(ValueError,match="fresh"):
        fresh_output(tmp_path)
    assert (tmp_path/"keep.txt").read_text()=="preserve"

def test_original_hash_checked_before_payload_loader(tmp_path):
    bank=tmp_path/"bank";bank.mkdir()
    (bank/"FROZEN.json").write_text("{}")
    class Refuse:
        @staticmethod
        def verify(*_):
            raise AssertionError("must not read unbound artifact")
    with pytest.raises(ValueError,match="bank manifest SHA256"):
        assert_original_bindings(SimpleNamespace(io=Refuse),bank,tmp_path/"student",
                                 tmp_path/"recipe","0"*64,"0"*64)

def test_sanitized_recipe_never_rewritten(tmp_path):
    bank=tmp_path/"bank";bank.mkdir()
    checkpoint=tmp_path/"student";checkpoint.mkdir()
    (bank/"FROZEN.json").write_text("{}")
    (checkpoint/"last.pt").write_bytes(b"unread test payload")
    recipe=tmp_path/"RECIPE.json"
    original=json.dumps({"data_root":"${REPO_ROOT}/data"})
    recipe.write_text(original)
    class Stub:
        @staticmethod
        def verify(*_):return {}
    with pytest.raises(ValueError,match="sanitized"):
        assert_original_bindings(SimpleNamespace(io=Stub),bank,checkpoint,recipe,
                                 digest(bank/"FROZEN.json"),digest(checkpoint/"last.pt"))
    assert recipe.read_text()==original


def test_original_recipe_mutation_rejected_before_tensor_load(tmp_path,monkeypatch):
    import torch
    from experiments.guidance_signal_20261014 import preflight as pf
    bank=tmp_path/'bank';bank.mkdir()
    ck=tmp_path/'student';ck.mkdir()
    (bank/'FROZEN.json').write_text('{}');(ck/'last.pt').write_bytes(b'not a checkpoint')
    recipe=tmp_path/'RECIPE.json';recipe.write_text('{"lr":0.9}')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('unbound tensor load'))
    rt=SimpleNamespace(io=SimpleNamespace(verify=lambda *a:{}))
    with pytest.raises(ValueError,match='recipe independent SHA256'):
        pf.assert_original_bindings(rt,bank,ck,recipe,digest(bank/'FROZEN.json'),digest(ck/'last.pt'))


def test_positive_synthetic_original_binding_and_permission_mismatch(tmp_path,monkeypatch):
    """Only synthetic pins are injected here; CLI has no override for them."""
    import torch
    from experiments.guidance_signal_20261014 import preflight as pf
    bank=tmp_path/'bank';bank.mkdir();student=tmp_path/'student';student.mkdir()
    source={'fixture':'SYNTHETIC_ONLY'}
    split={'train':['a','b'],'val':['c'],'test':['d']}
    bm={'split':'train','patients':['a','b'],'locked_splits':split,'config':{'grid':2},
        'teacher':{'fixture':'SYNTHETIC_ONLY'},'source_sha256':source,
        'patient_reports':{'a':{'class_pixels':[1,1,0,1]},'b':{'class_pixels':[1,0,0,1]}}}
    (bank/'FROZEN.json').write_text(json.dumps(bm));bank_sha=digest(bank/'FROZEN.json')
    sm={'source_sha256':source,'locked_splits':split,'teacher':bm['teacher'],'bank_manifest_sha256':bank_sha}
    (student/'FROZEN.json').write_text(json.dumps(sm))
    recipe=tmp_path/'RECIPE.json';recipe.write_text(json.dumps({'schema':'self_audit.full_guidance_recipe.v1',
        'source_sha256':source,'splits':split,'config':bm['config']}))
    ck={'bank_manifest_sha256':bank_sha,'config':bm['config'],'locked_splits':split,'teacher':bm['teacher'],
        'train_patients':['a','b'],'allowed_classes':[True,False,False,True]}
    torch.save(ck,student/'last.pt')
    monkeypatch.setattr(pf,'ORIGINAL_RECIPE_SHA256',digest(recipe))
    monkeypatch.setattr(pf,'ORIGINAL_BANK_SHA256',bank_sha)
    monkeypatch.setattr(pf,'ORIGINAL_STUDENT_SHA256',digest(student/'FROZEN.json'))
    monkeypatch.setattr(pf,'ORIGINAL_CHECKPOINT_SHA256',digest(student/'last.pt'))
    rt=SimpleNamespace(io=SimpleNamespace(verify=lambda path,role:bm if role=='guidance_bank' else sm),source_sha256=source)
    result=pf.assert_original_bindings(rt,bank,student,recipe,bank_sha,digest(student/'last.pt'))
    assert result['status']=='PASS_ORIGINAL_ARTIFACT_BINDINGS'
    assert result['class_patients']==[2,1,0,2] and result['training_run'] is False
    ck['allowed_classes']=[True]*4;torch.save(ck,student/'last.pt')
    monkeypatch.setattr(pf,'ORIGINAL_CHECKPOINT_SHA256',digest(student/'last.pt'))
    with pytest.raises(ValueError,match='permissions differ'):
        pf.assert_original_bindings(rt,bank,student,recipe,bank_sha,digest(student/'last.pt'))

def test_preflight_original_pins_match_verified_publication():
    from experiments.guidance_signal_20261014 import preflight as pf,runtime
    runtime.verify_source()
    entries={e['published_path']:e for e in json.loads((runtime.SNAPSHOT_ROOT/'MANIFEST.json').read_text())['files']}
    for relative,expected in [('run_evidence/run/RECIPE.json',pf.ORIGINAL_RECIPE_SHA256),
                              ('run_evidence/run/bank/FROZEN.json',pf.ORIGINAL_BANK_SHA256),
                              ('run_evidence/run/student/FROZEN.json',pf.ORIGINAL_STUDENT_SHA256)]:
        assert entries[relative]['original_sha256']==expected
    sm=json.loads((runtime.SNAPSHOT_ROOT/'run_evidence/run/student/FROZEN.json').read_text())
    assert sm['files']['last.pt']==pf.ORIGINAL_CHECKPOINT_SHA256
