"""Regression checks for profiler receipt formats and fail-closed capacity gates."""
import json
from scripts.run_teacher_memory_full import Experiment


def test_dense_capacity_profile_reads_actual_reserved_peak_field(tmp_path,monkeypatch):
    exp=Experiment(tmp_path,10000,1000);monkeypatch.setattr(exp,'idle_gpu',lambda:None)
    def command(cmd,log,**kwargs):
        out=tmp_path/'profiles/dense';out.mkdir()
        (out/'PROFILE.json').write_text(json.dumps({'status':'COMPLETED','runs':[
            {'cuda_peak_reserved_bytes_including_model_and_teacher':100},
            {'cuda_peak_reserved_bytes_including_model_and_teacher':200}]}))
        return 0
    monkeypatch.setattr(exp,'command',command)
    result=exp.profile(32,2,None,None,'dense',dense=True)
    assert result['status']=='PASS' and result['peak_reserved_bytes']==200


def test_dense_oom_never_becomes_passing_capacity(tmp_path,monkeypatch):
    exp=Experiment(tmp_path,10000,1000);monkeypatch.setattr(exp,'idle_gpu',lambda:None)
    def command(cmd,log,**kwargs):
        out=tmp_path/'profiles/dense';out.mkdir()
        (out/'PROFILE.json').write_text(json.dumps({'status':'RUNNING','runs':[]}))
        (tmp_path/log).write_text('{"status": "FAILED_OOM", "successful_profile": false}\n')
        return 1
    monkeypatch.setattr(exp,'command',command)
    result=exp.profile(64,2,None,None,'dense',dense=True)
    assert result['status']=='OOM' and 'suite_samples_per_second' not in result
