"""Run the pinned continuation and record process-tree resources without changing its training."""
import json
import os
from pathlib import Path
import subprocess
import time
import psutil

out = Path('/workspace/cptu-launch-20261031')
command = json.loads((out/'LAUNCH_COMMAND.json').read_text())
assert not (out/'RUNNER_EXIT.json').exists(), 'Never silently restart a completed or stopped pilot'
assert not (out/'LOCKED_PLAN.json').exists(), 'Existing pilot must be resumed explicitly'
occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if occupied:
    raise RuntimeError('GPU occupied; no jobs stopped and no pilot launched: ' + occupied)
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
started = time.time()
with (out/'runner.log').open('x') as log, (out/'RESOURCE_SAMPLES.jsonl').open('x') as resources:
    p = subprocess.Popen(command['argv'],cwd=command['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/'RUNNER_PROCESS.json').write_text(json.dumps({'pid':p.pid,'observer_pid':os.getpid(),
        'start_unix':started,'tmux':command['tmux'],'source_commit':command['source_commit']},indent=2)+'\n')
    peak = 0
    while True:
        rss = 0
        rows = []
        try:
            root = psutil.Process(p.pid)
            processes = [root] + root.children(recursive=True)
        except psutil.Error:
            processes = []
        for proc in processes:
            try:
                n = proc.memory_info().rss
                rss += n
                rows.append({'pid':proc.pid,'rss':n,'name':proc.name()})
            except psutil.Error:
                pass
        peak = max(peak,rss)
        gpu = subprocess.run(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        resources.write(json.dumps({'unix':time.time(),'elapsed_seconds':time.time()-started,
            'tree_rss_bytes':rss,'peak_observed_tree_rss_bytes':peak,'processes':rows,
            'gpu_csv':gpu.stdout.strip(),'gpu_query_exit':gpu.returncode})+'\n')
        resources.flush()
        code = p.poll()
        if code is not None:
            break
        time.sleep(10)
    result={'exit_code':code,'start_unix':started,'end_unix':time.time(),
            'wall_seconds':time.time()-started,'peak_observed_tree_rss_bytes':peak,
            'sampling_interval_seconds':10,'source_commit':command['source_commit']}
    tmp = out/'RUNNER_EXIT.json.tmp'
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(out/'RUNNER_EXIT.json')
raise SystemExit(code)
