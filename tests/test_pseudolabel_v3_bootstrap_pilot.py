"""TRAIN-only pilot planning/stop contracts; manufactured fixtures are not quality evidence."""
import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from scripts.run_bootstrap_pilot_v3 import choose_patients,main,support_report
from test_pseudolabel_v3_regressions import _nifti_fixture

ROOT=Path(__file__).resolve().parents[1]


def training_fixture(tmp_path):
    data,split=_nifti_fixture(tmp_path)
    split.write_text(json.dumps({'train':['patient001','patient002'],'val':[],'test':[]}))
    return data,split


def test_selection_is_order_invariant_train_only_and_seed_deterministic():
    split={'train':['b','d','a','c'],'val':['heldout'],'test':['test']}
    first=choose_patients(split,2,42)
    assert first==choose_patients(dict(split,train=list(reversed(split['train']))),2,42)
    assert set(first)<=set(split['train']) and len(first)==2
    for n in (0,1,5,True):
        with pytest.raises(ValueError):choose_patients(split,n,42)


def test_dry_plan_reads_images_only_and_never_writes_or_launches(tmp_path,monkeypatch,capsys):
    import nibabel as nib
    data,split=training_fixture(tmp_path);load=nib.load
    def guard(path,*args,**kwargs):
        assert '_gt' not in str(path)
        return load(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',guard)
    launched=MagicMock();monkeypatch.setattr('scripts.run_bootstrap_pilot_v3.run_stage',launched)
    out=tmp_path/'pilot'
    result=main(['--root',str(data),'--split-manifest',str(split),'--out',str(out),'--patients','2','--dry-run'])
    assert result==0 and not out.exists() and not launched.called
    assert 'planned_updates=36' in capsys.readouterr().out


@pytest.mark.parametrize('extra',[['--epochs','1'],['--max-planned-updates','1'],
    ['--config',str(ROOT/'configs/pseudolabel_v3.json')]])
def test_impossible_or_legacy_pilot_refused_before_outputs_or_training(tmp_path,monkeypatch,extra):
    data,split=training_fixture(tmp_path);out=tmp_path/'pilot'
    launched=MagicMock();monkeypatch.setattr('scripts.run_bootstrap_pilot_v3.run_stage',launched)
    with pytest.raises(ValueError):
        main(['--root',str(data),'--split-manifest',str(split),'--out',str(out),'--patients','2',*extra])
    assert not out.exists() and not launched.called


def test_completed_pilot_reports_stop_without_evaluation_or_student(tmp_path,monkeypatch):
    from scripts import run_bootstrap_pilot_v3 as pilot
    data,split=training_fixture(tmp_path);out=tmp_path/'pilot';commands=[]
    def stage(name,cmd,env,log):commands.append((name,cmd));return .1
    monkeypatch.setattr(pilot,'run_stage',stage)
    monkeypatch.setattr(pilot,'support_report',lambda _: {'status':'STOP_INSUFFICIENT_SUPPORT',
        'class_pixels':[0,0,0,0],'gates':{'bootstrap_activated':False}})
    result=main(['--root',str(data),'--split-manifest',str(split),'--out',str(out),'--patients','2','--no-progress'])
    assert result==2 and len(commands)==1 and commands[0][0]=='teacher'
    cmd=commands[0][1]
    assert cmd[cmd.index('--export-split')+1]=='train' and '--max-train-batches' not in cmd
    assert '--no-progress' in cmd and not any('evaluate_' in x or 'train_student' in x for x in cmd)
    selected=json.loads((out/'pilot_split.json').read_text())
    assert selected['val']==selected['test']==[] and len(selected['train'])==2
    assert len(selected['pilot']['source_split_sha256'])==64
    assert json.loads((out/'PILOT_REPORT.json').read_text())['planned_optimizer_steps']==36


@pytest.mark.parametrize('section,key',[('loss_weights','semantic_seed'),('training','lr')])
def test_zero_semantic_objective_refused_before_training(tmp_path,monkeypatch,section,key):
    data,split=training_fixture(tmp_path)
    cfg=json.loads((ROOT/'configs/pseudolabel_v3_bootstrap_experimental.json').read_text())
    cfg[section][key]=0;config=tmp_path/'zero.json';config.write_text(json.dumps(cfg))
    launched=MagicMock();monkeypatch.setattr('scripts.run_bootstrap_pilot_v3.run_stage',launched)
    with pytest.raises(ValueError,match='positive semantic'):
        main(['--root',str(data),'--split-manifest',str(split),'--out',str(tmp_path/'out'),
              '--patients','2','--config',str(config)])
    assert not launched.called and not (tmp_path/'out').exists()


def frozen_pilot(tmp_path):
    from self_audit_pseudolabel.data_v3 import read_patient_splits
    from self_audit_pseudolabel.pipeline_v3 import inventory,discover
    from self_audit_pseudolabel.freeze import export_pseudo_npz,sha256_file,_digest
    from pseudolabel_v3_fixtures import seal_test_teacher
    data,split=training_fixture(tmp_path);splits=read_patient_splits(split)
    records=inventory(discover('acdc',data),splits);out=tmp_path/'teacher';out.mkdir();entries=[]
    for record in records:
        for t in range(record['shape'][3]):
            for z in range(record['shape'][2]):
                label=np.zeros((record['shape'][1],record['shape'][0]),dtype=np.uint8)
                label[1:4,1:4]=1;label[5:8,1:4]=2;label[9:12,1:4]=3
                valid=np.ones_like(label);soft=np.eye(4,dtype=np.float32)[label].transpose(2,0,1)
                meta={'patient_id':record['patient_id'],'t':t,'z':z};name=f"{meta['patient_id']}_{t}_{z}.npz"
                digest=export_pseudo_npz(out/name,pseudo_label=label,valid=valid,soft_label=soft,metadata=meta)
                entries.append({**meta,'path':name,'sha256':digest,'split':'train'})
    cfg={'dataset':'acdc','image_records':records,'export_records':records,'split_patients':splits,'producer_patient_ids':splits['train']}
    payload=seal_test_teacher(out,entries,cfg)
    payload['config']['resolved_config']=json.loads((ROOT/'configs/pseudolabel_v3_bootstrap_experimental.json').read_text())
    payload['config']['bootstrap_summary']={'enabled':True,'ready':True,'stage':'semantic'}
    (out/'train_metrics.json').write_text(json.dumps([{'accepted_bg_regions':0,'accepted_rv_regions':1,'accepted_myo_regions':1,'accepted_lv_regions':1}]))
    payload['artifacts']['train_metrics.json']=sha256_file(out/'train_metrics.json')
    payload.pop('manifest_id');payload['manifest_id']=_digest(payload)
    (out/'FROZEN.json').write_text(json.dumps(payload))
    return out,payload


def test_support_report_is_frozen_read_only_not_quality_certification(tmp_path):
    out,payload=frozen_pilot(tmp_path)
    before=(out/'FROZEN.json').read_bytes();report=support_report(out)
    assert report['status']=='SUPPORT_OBSERVED_NOT_QUALITY_CERTIFIED' and all(report['gates'].values())
    assert report['foreground_optimizer_steps']==1 and len(report['foreground_patients'])==2
    assert report['evaluation_performed'] is False and report['quality_status']=='NOT_EVALUATED'
    assert (out/'FROZEN.json').read_bytes()==before
    with (out/payload['entries'][0]['path']).open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='changed'):support_report(out)


def test_positive_predictions_without_foreground_training_do_not_pass_pilot(tmp_path):
    from self_audit_pseudolabel.freeze import sha256_file,_digest
    out,payload=frozen_pilot(tmp_path)
    (out/'train_metrics.json').write_text('[]')
    payload['artifacts']['train_metrics.json']=sha256_file(out/'train_metrics.json')
    payload.pop('manifest_id');payload['manifest_id']=_digest(payload)
    (out/'FROZEN.json').write_text(json.dumps(payload))
    report=support_report(out)
    assert report['class_pixels'][1]>0 and report['status']=='STOP_INSUFFICIENT_SUPPORT'
    assert report['gates']['foreground_optimizer_updates'] is False
