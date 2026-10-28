"""Physical manufactured evidence checks; these fixtures cannot authorize patient training."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from self_audit.data import write_json
from self_audit.engine import DEFAULT, source_digest
from self_audit.evaluation import class_metrics, confusion_matrix
from self_audit.model import digest
from self_audit.recovery_gates import (
    BASELINE_TOLERANCE_FG, RecoveryGateError, MissingEvidenceError,
    ImplementationVerificationError, EmpiricalGateFailureError,
    build_baseline_gate_certificate, verify_baseline_gate_certificate,
    block_shared_training_unless_baseline_gated, evaluate_baseline_gate,
    verify_historical_lineage, run_tiny_fit_diagnostic,
)
from scripts.run_baseline_recovery import predeclare_recovery_protocol, main


def physical_fixture(tmp_path):
    d=tmp_path/'bundle'; d.mkdir()
    data=d/'data'; data.mkdir()
    predictions=d/'predictions/acdc'; predictions.mkdir(parents=True)
    y=np.array([[[0,1,2,3,255]]],dtype=np.uint8)
    pred=np.array([[[[0,1,2,3,0]]]],dtype=np.uint8)
    np.save(data/'mask.npy',y); np.save(predictions/'p001.npy',pred)
    train={'schema':'self_audit.data.v2','role':'training','synthetic':True,
           'labeled':[{'id':'l1'},{'id':'l2'}],'unlabeled':[{'id':'u1'}]}
    write_json(d/'train.json',train)
    ev={'schema':'self_audit.data.v2','role':'development','synthetic':True,
        'patients':[{'id':'p001','masks':{'0':{'path':str(data/'mask.npy'),'sha256':digest(data/'mask.npy')}}}]}
    write_json(d/'eval.json',ev)
    cfg=copy.deepcopy(DEFAULT);cfg.update(method='ema_full',steps=3,warmup_steps=1)
    cfg['model']['backbone']='tiny_test';write_json(d/'config.json',cfg)
    proto=predeclare_recovery_protocol(d/'protocol.json', config_path=d/'config.json',
        train_manifest_path=d/'train.json',evaluation_manifest_path=d/'eval.json',
        synthetic=True,target_steps=3,warmup_steps=1)
    # Explicitly manufactured payload: positive integrity fixture only, never a real model claim.
    (d/'weights.pt').write_bytes(b'manufactured integrity fixture')
    ck={'schema':'self_audit.checkpoint.v2','synthetic':True,'step':3,'config':proto['config'],
        'payload_sha256':digest(d/'weights.pt'),'data_sha256':digest(d/'train.json'),
        'labeled_ids':['l1','l2'],'unlabeled_ids':['u1'],'source_sha256':source_digest()}
    write_json(d/'MANIFEST.json',ck)
    recipe={'synthetic':True,'threshold':None,'checkpoint_sha256':digest(d/'MANIFEST.json'),
            'checkpoint_payload_sha256':digest(d/'weights.pt'),
            'domains':{'acdc':{'manifest_sha256':digest(d/'eval.json')}}}
    write_json(d/'RECIPE.json',recipe)
    frozen={'recipe_sha256':digest(d/'RECIPE.json'),'checkpoint_manifest_sha256':digest(d/'MANIFEST.json'),
            'checkpoint_payload_sha256':digest(d/'weights.pt'),
            'domains':{'acdc':[{'id':'p001','file':'acdc/p001.npy','sha256':digest(predictions/'p001.npy')}]}}
    write_json(d/'PREDICTIONS_FROZEN.json',frozen)
    cm=confusion_matrix(y,pred[0]);raw=class_metrics(cm)
    dm={'frames':[{'patient':'p001','frame':0,'confusion':cm.tolist(),'raw':raw}],
        'patient_metrics':{'p001':dict(zip(('BG','RV','MYO','LV'),[1.]*4))},
        'foreground_patient_macro':1.}
    metrics={'synthetic':True,'recipe_sha256':digest(d/'RECIPE.json'),
             'checkpoint_manifest_sha256':digest(d/'MANIFEST.json'),'checkpoint_payload_sha256':digest(d/'weights.pt'),
             'frozen_predictions_sha256':digest(d/'PREDICTIONS_FROZEN.json'),'domains':{'acdc':dm}}
    write_json(d/'METRICS.json',metrics)
    mods=dict.fromkeys(('encoder','decoder','detail','fuse','head'),.01)
    tiny={'synthetic':True,'source_sha256':source_digest(),'train_manifest_sha256':digest(d/'train.json'),
          'status':'CONVERGED','converged':True,'final_fg_dice':1.,'final_class_dices':dict.fromkeys(('RV','MYO','LV'),1.),
          'steps_taken':2,'probe_patients':['l1','l2'],'optimizer_ownership':{'verified':True},
          'step_0_gradient_norms':mods,'step_0_update_deltas':mods,'module_update_deltas':mods}
    write_json(d/'TINY_FIT_REPORT.json',tiny)
    args=dict(checkpoint_manifest_path=d/'MANIFEST.json',checkpoint_payload_path=d/'weights.pt',
              train_manifest_path=d/'train.json',train_config_path=d/'config.json',protocol_path=d/'protocol.json',
              frozen_predictions_path=d/'PREDICTIONS_FROZEN.json',metrics_path=d/'METRICS.json',
              tinyfit_receipt_path=d/'TINY_FIT_REPORT.json',evaluation_manifest_path=d/'eval.json',synthetic=True)
    return d,args


def mutate(p, fn):
    obj=json.loads(p.read_text());fn(obj);write_json(p,obj)


def test_physical_bundle_roundtrip_and_no_synthetic_authorization(tmp_path):
    d,args=physical_fixture(tmp_path)
    cert_path=d/'gate.json'
    cert=build_baseline_gate_certificate(**args,out_path=cert_path)
    assert verify_baseline_gate_certificate(cert_path)==cert
    with pytest.raises(ImplementationVerificationError,match='Synthetic'):
        block_shared_training_unless_baseline_gated(cert_path,expected_cert_sha=digest(cert_path))
    with pytest.raises(MissingEvidenceError):
        verify_baseline_gate_certificate(cert_path,require_physical_files=False)


@pytest.mark.parametrize('name',['weights.pt','METRICS.json','predictions/acdc/p001.npy','data/mask.npy','RECIPE.json'])
def test_missing_or_changed_physical_evidence_never_verifies(tmp_path,name):
    d,args=physical_fixture(tmp_path);cert=build_baseline_gate_certificate(**args)
    (d/name).unlink()
    with pytest.raises((RecoveryGateError,FileNotFoundError)):
        verify_baseline_gate_certificate(cert)


@pytest.mark.parametrize('field,value',[
    ('foreground_patient_macro',.99),('foreground_patient_macro',float('nan')),
])
def test_metrics_cannot_override_confusion(tmp_path,field,value):
    d,args=physical_fixture(tmp_path)
    # json.dumps permits NaN in this deliberately corrupted fixture.
    p=d/'METRICS.json';obj=json.loads(p.read_text());obj['domains']['acdc'][field]=value
    p.write_text(json.dumps(obj))
    with pytest.raises(RecoveryGateError):build_baseline_gate_certificate(**args)


def test_forged_files_and_macro_only_scalars_rejected(tmp_path):
    d,args=physical_fixture(tmp_path)
    for name in ('MANIFEST.json','config.json','protocol.json','PREDICTIONS_FROZEN.json','METRICS.json'):
        (d/name).write_text('{}')
    with pytest.raises(RecoveryGateError): build_baseline_gate_certificate(**args)
    with pytest.raises(MissingEvidenceError):
        evaluate_baseline_gate({'macro_patient':dict.fromkeys(('RV','MYO','LV'),.99)},synthetic=True)


@pytest.mark.parametrize('target,field,value',[
    ('MANIFEST.json','source_sha256','a'*64),('MANIFEST.json','step',4),
    ('RECIPE.json','checkpoint_sha256','b'*64),('PREDICTIONS_FROZEN.json','recipe_sha256','c'*64),
    ('METRICS.json','frozen_predictions_sha256','d'*64),('METRICS.json','checkpoint_manifest_sha256',None),
    ('TINY_FIT_REPORT.json','train_manifest_sha256','e'*64),('TINY_FIT_REPORT.json','steps_taken',201),
    ('TINY_FIT_REPORT.json','final_fg_dice',.94),('TINY_FIT_REPORT.json','probe_patients',['query']),
])
def test_lineage_and_optimization_crosslinks_required(tmp_path,target,field,value):
    d,args=physical_fixture(tmp_path);mutate(d/target,lambda x:x.update({field:value}))
    with pytest.raises(RecoveryGateError):build_baseline_gate_certificate(**args)


def test_references_rescored_after_freeze_and_false_counts_rejected(tmp_path):
    d,args=physical_fixture(tmp_path)
    # An internally consistent but false confusion/metric report cannot override physical arrays.
    def falsify(m):
        row=m['domains']['acdc']['frames'][0]
        row['confusion'][0][0]+=10
        row['raw']=class_metrics(np.array(row['confusion']))
    mutate(d/'METRICS.json',falsify)
    with pytest.raises(ImplementationVerificationError,match='rescored frame confusion'):
        build_baseline_gate_certificate(**args)


def test_patient_macro_not_class_macro_and_absent_does_not_become_zero():
    frames=[];patients={}
    for pid,diagonal in [('a',[1,8,1,1]),('b',[1,0,1,1])]:
        cm=np.zeros((4,5),dtype=np.int64)
        for c,n in enumerate(diagonal):cm[c,c]=n
        if pid=='a':cm[1,0]=2
        raw=class_metrics(cm);frames.append({'patient':pid,'frame':0,'confusion':cm.tolist(),'raw':raw})
        patients[pid]={r['class']:r['dice'] for r in raw}
    result=evaluate_baseline_gate({'frames':frames,'patients':patients},synthetic=True)
    expected=((8/9+1+1)/3+1)/2
    assert result['foreground_patient_macro']==pytest.approx(expected)
    assert result['foreground_patient_macro']!=pytest.approx(np.mean(list(result['class_dices'].values())))


def test_coverage_missing_frames_ignored_support_and_duplicate_frames_fail(tmp_path):
    d,args=physical_fixture(tmp_path);m=json.loads((d/'METRICS.json').read_text())
    with pytest.raises(RecoveryGateError):evaluate_baseline_gate(m,expected_frame_count=2,synthetic=True)
    with pytest.raises(RecoveryGateError):evaluate_baseline_gate(m,synthetic=False)
    m['domains']['acdc']['frames']*=2
    with pytest.raises(RecoveryGateError):evaluate_baseline_gate(m,synthetic=True)


def test_protocol_cannot_be_overwritten_or_omit_artifacts(tmp_path):
    with pytest.raises(MissingEvidenceError):predeclare_recovery_protocol(tmp_path/'p.json')
    d,args=physical_fixture(tmp_path)
    with pytest.raises(ValueError,match='must be new'):
        predeclare_recovery_protocol(d/'protocol.json',config_path=d/'config.json',
            train_manifest_path=d/'train.json',evaluation_manifest_path=d/'eval.json',synthetic=True)


def test_history_metadata_does_not_claim_missing_weights_recovered():
    path=Path('reports/window_pilot_test50_20261024/baseline/CHECKPOINT_MANIFEST.json')
    result=verify_historical_lineage(path)
    assert result['physical_payload_verified'] is False
    with pytest.raises(MissingEvidenceError):verify_historical_lineage(path,require_payload_exists=True)


@pytest.mark.parametrize('budget',[0,201,True])
def test_tinyfit_invalid_budget_rejected_before_data_access(tmp_path,budget):
    with pytest.raises(ImplementationVerificationError):
        run_tiny_fit_diagnostic('/nonexistent','a'*64,tmp_path/'fit',max_steps=budget)


def test_cli_gate_requires_reference_manifest_and_training_protocol():
    with pytest.raises(SystemExit):main(['gate-check'])
    with pytest.raises(SystemExit):main(['train-baseline','--data','x','--data-sha256','a'*64,'--out','y'])


def test_startup_receipt_verification_never_opens_reference_masks(tmp_path, monkeypatch):
    d,args=physical_fixture(tmp_path)
    cert=build_baseline_gate_certificate(**args)
    original=np.load
    def no_reference(path,*a,**kw):
        assert Path(path).name != "mask.npy", "learner startup opened reference mask"
        return original(path,*a,**kw)
    monkeypatch.setattr(np,"load",no_reference)
    assert verify_baseline_gate_certificate(cert,read_reference_arrays=False)["status"]=="PASSED"
