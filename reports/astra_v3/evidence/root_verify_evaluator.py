"""Independent post-freeze evaluator rerun and root metric checks.

Only this separate evaluator process may open GT; it imports no training/model
code. Frozen generation artifacts and worker results are never overwritten.
"""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np


def main():
    p=Path('reports/astra_v3/workers/H_evidence/evaluator.py')
    spec=importlib.util.spec_from_file_location('postfreeze_evaluator',p)
    e=importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    e.run_synthetic_tests()
    checks=[]
    # Exercise the actual verification function, not a tautological bad-hash comparison.
    real_sha=e.sha256_file
    manifest=json.loads(Path(e.FROZEN_MANIFEST_PATH).read_text())
    paths=[e.FROZEN_MANIFEST_PATH,manifest['config_path'],manifest['script_path'],
           manifest['split_path'],manifest['records'][0]['source_image'],manifest['records'][0]['prediction']]
    for target in paths:
        def fake_sha(path): return '0'*64 if str(path)==target else real_sha(path)
        with patch.object(e,'sha256_file',side_effect=fake_sha), patch.object(e.nib,'load',side_effect=AssertionError('GT read before verification')) as loader:
            try: e.verify_freeze_integrity()
            except ValueError: pass
            else: raise AssertionError('actual freeze verifier failed to reject tampering')
            assert loader.call_count==0
        checks.append('actual verifier rejected simulated hash mismatch: '+target)
    # Patient-equal, phase-equal aggregation differs from pooled counts.
    a=e.compute_volume_metrics(np.array([1,2,3]),np.array([1,2,3]))
    b=e.compute_volume_metrics(np.tile([1,2,3],100),np.full(300,255))
    assert (a['macro_fg_dice']+b['macro_fg_dice'])/2 == 0.5
    checks.append('phase equal synthetic mean=0.5 despite 100x voxel-count difference')
    before=e.verify_freeze_integrity()
    rerun=e.evaluate_postfreeze(before)
    after=e.verify_freeze_integrity()  # also config/script/split after evaluation
    rerun=json.loads(json.dumps(rerun))  # JSON serializes integer class keys as strings
    original=json.loads(Path('reports/astra_v3/workers/H_evidence/evaluation_results.json').read_text())
    assert original['split_summaries']==rerun['split_summaries']
    assert original['record_evaluations']==rerun['record_evaluations']
    checks.append('all record metrics and split metrics exactly reproduce worker H')
    independently_derived={}
    for split in ['train','dev']:
        independently_derived[split]={}
        records=[r for r in rerun['record_evaluations'] if r['split']==split]
        for arm in original['split_summaries'][split]['arms']:
            patient_class={}; conf=np.zeros((4,5),dtype=np.int64)
            for r in records:
                cm=np.asarray(r['arms'][arm]['confusion_matrix_4x5'])
                assert cm.sum()==np.prod(r['shape'])
                assert (cm.sum(1)[1:]>0).all()  # no absent GT foreground volumes in this cohort
                class_dice=2*np.diag(cm[:,:4])[1:]/(cm.sum(1)[1:]+cm[:,:4].sum(0)[1:])
                patient_class.setdefault(r['patient_id'],[]).append(class_dice)
                conf+=cm
            means=np.array([np.mean(values,axis=0) for values in patient_class.values()])
            primary=float(means.mean())
            assert abs(primary-original['split_summaries'][split]['arms'][arm]['primary_dice'])<1e-12
            independently_derived[split][arm]={'primary_dice':primary,'patient_phase_macro_class_dice_RV_MYO_LV':means.mean(0).tolist(),
                'fg_precision':float(np.diag(conf[:,:4])[1:].sum()/conf[:,1:4].sum()) if conf[:,1:4].sum() else None,
                'foreground_correct_accepted':int(np.diag(conf[:,:4])[1:].sum())}
    result={'status':'PASS','checks':checks,'before':before,'after':after,
            'first_gt_read_utc':rerun['first_gt_read_utc'],'gt_opens_log':rerun['gt_opens_log'],
            'independent_aggregations':independently_derived,
            'limitations':['worker synthetic tamper test was tautological; root tests actual verifier',
              'worker harm correct_voxel_delta includes BG, not foreground only',
              'topology arm equals anchor on this cohort; not every cue changes outputs',
              'sparse seed Dice is not a trained teacher or impossibility result',
              'postfreeze stop/go on development GT is research evaluation feedback, not blind clinical validation']}
    Path('reports/astra_v3/evidence/root_evaluator_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(independently_derived,indent=2))


if __name__=='__main__': main()
