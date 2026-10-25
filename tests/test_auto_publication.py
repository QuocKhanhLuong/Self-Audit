import json
import pytest
from scripts.watch_experiment_publish import completion_event


def test_auto_publication_does_not_trigger_during_training(tmp_path):
    (tmp_path/'STATUS.json').write_text(json.dumps({'stage':'TRAINING'}))
    assert completion_event(tmp_path) is None
    (tmp_path/'STATUS.json').write_text(json.dumps({'stage':'FAILED'}))
    assert completion_event(tmp_path)=='FAILED_REQUIRES_INSPECTION'


def test_auto_publication_requires_every_arm_split_and_check_receipt(tmp_path):
    (tmp_path/'STATUS.json').write_text(json.dumps({'stage':'COMPLETED'}))
    rows=[dict(arm=f'A{i}',split=s,checkpoint=c) for i in range(7)
          for s,c in (('VAL18','step_005000'),('VAL18','final'),('TEST50','final'))]
    (tmp_path/'RESULTS.json').write_text(json.dumps(rows[:-1]))
    with pytest.raises(RuntimeError,match='21 expected'):completion_event(tmp_path)
    (tmp_path/'RESULTS.json').write_text(json.dumps(rows))
    with pytest.raises(RuntimeError,match='evidence missing'):completion_event(tmp_path)
    for name in ('check-repo-final.log','pytest-final.log','TRAINING_RESULTS.json',
                 'FROZEN_FINAL_CHECKPOINTS.json','RESULTS.csv'):
        (tmp_path/name).write_text('test fixture, not patient evidence')
    assert completion_event(tmp_path)=='COMPLETED_VERIFY_AND_PUBLISH'
