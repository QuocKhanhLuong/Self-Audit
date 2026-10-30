import os,subprocess,json,time
from pathlib import Path
r=Path("/workspace/Self-Audit-cptu-20261030"); out=Path("/workspace/cptu-server-setup-20261030/verification_v3")
env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"); env["PATH"]="/workspace/cptu-runtime-20261030/bin:"+env.get("PATH","")
t=time.monotonic()
with (out/"pytest.stdout").open("w") as stdout,(out/"pytest.stderr").open("w") as stderr:
 p=subprocess.run(["/workspace/cptu-runtime-20261030/bin/python","-m","pytest","-q"],cwd=r,env=env,stdout=stdout,stderr=stderr,timeout=600)
(out/"FINAL_SUITE.json").write_text(json.dumps({"status":"PASS" if p.returncode==0 else "FAIL","exit_code":p.returncode,"seconds":time.monotonic()-t,"git_sha":subprocess.check_output(["git","-C",str(r),"rev-parse","HEAD"],text=True).strip(),"scope":"repository tests only; does not override the failed exact CUDA reproduction gate"},indent=2)+"\n")
