"""Local software preflight; optional verification of original on-host artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL_RECIPE_SHA256 = "bed149d73bb0527c1dbcea8ff72e9b28e093149974dcb0c5893fd43d53a6092b"
ORIGINAL_BANK_SHA256 = "04fb2d30df6c0e0ed235dcf851a491895661b7fd4763e6b89ce95ca6811accae"
ORIGINAL_STUDENT_SHA256 = "85340cf5f4502d974638edd08eaca030d4642c530b5bf1680d2c21b18e09971e"
ORIGINAL_CHECKPOINT_SHA256 = "90bc16bb9f89963b452355376673e3ba9aa24b38870e4595237d75e402ce873e"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def fresh_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("output must be a fresh path")
    if any((parent/"FROZEN.json").exists() or (parent/"PREDICTIONS_LOCKED.json").exists()
           for parent in path.parents):
        raise ValueError("output cannot be inside a frozen or locked artifact")
    path.mkdir(parents=True)
    return path

def assert_original_bindings(runtime, bank, checkpoint, recipe, bank_sha, checkpoint_sha):
    """Read original files; never rewrite sanitized copies or patch their digests."""
    bank,checkpoint,recipe = Path(bank),Path(checkpoint),Path(recipe)
    if digest(bank/"FROZEN.json") != bank_sha:
        raise ValueError("original bank manifest SHA256 mismatch")
    if digest(checkpoint/"last.pt") != checkpoint_sha:
        raise ValueError("original checkpoint SHA256 mismatch")
    bank_manifest = runtime.io.verify(bank,"guidance_bank")
    student_manifest = runtime.io.verify(checkpoint,"student")
    raw = recipe.read_text()
    if "${" in raw:
        raise ValueError("sanitized recipe placeholders are not original artifacts")
    if digest(recipe) != ORIGINAL_RECIPE_SHA256:
        raise ValueError("original recipe independent SHA256 mismatch")
    if bank_sha != ORIGINAL_BANK_SHA256 or checkpoint_sha != ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError("expected identity is not the published world-left-v2 run")
    if digest(checkpoint/"FROZEN.json") != ORIGINAL_STUDENT_SHA256:
        raise ValueError("original student manifest independent SHA256 mismatch")
    resolved = json.loads(raw)
    if resolved.get("schema") != "self_audit.full_guidance_recipe.v1":
        raise ValueError("original recipe schema mismatch")
    for item in (resolved,bank_manifest,student_manifest):
        if item.get("source_sha256") != runtime.source_sha256:
            raise ValueError("original artifact does not use the pinned executed source")
    if bank_manifest.get("split") != "train":
        raise ValueError("original bank is not TRAIN")
    if not bank_manifest.get("patients"):
        raise ValueError("empty TRAIN bank")
    split = bank_manifest["locked_splits"]
    if resolved.get("splits") != split:
        raise ValueError("recipe and original bank locked split differ")
    if not set(bank_manifest["patients"]).issubset(split["train"]):
        raise ValueError("non-TRAIN patient in bank")
    if student_manifest.get("bank_manifest_sha256") != bank_sha:
        raise ValueError("student freeze is not bound to original bank")
    if resolved.get("config") != bank_manifest["config"]:
        raise ValueError("recipe and bank configuration differ")
    if student_manifest.get("locked_splits") != split:
        raise ValueError("student and bank locked split differ")
    if student_manifest.get("teacher") != bank_manifest.get("teacher"):
        raise ValueError("student and bank teacher identities differ")
    # The trusted local checkpoint is an original tensor-only payload.
    import torch
    ck = torch.load(checkpoint/"last.pt",map_location="cpu",weights_only=True)
    for key in ("bank_manifest_sha256","config","locked_splits","teacher"):
        expected = bank_sha if key == "bank_manifest_sha256" else bank_manifest[key]
        if ck.get(key) != expected:
            raise ValueError("checkpoint binding differs: "+key)
    if set(ck["train_patients"]) != set(bank_manifest["patients"]):
        raise ValueError("checkpoint TRAIN patient identities differ")
    permission_counts = [sum(row["class_pixels"][c]>0 for row in bank_manifest["patient_reports"].values())
                         for c in range(4)]
    if ck["allowed_classes"] != [n>=2 for n in permission_counts]:
        raise ValueError("checkpoint permissions differ from frozen bank")
    return {"status":"PASS_ORIGINAL_ARTIFACT_BINDINGS",
            "bank_manifest_sha256":bank_sha,"checkpoint_sha256":checkpoint_sha,
            "recipe_sha256":digest(recipe),"patients":len(bank_manifest["patients"]),
            "class_patients":permission_counts,"allowed_classes":ck["allowed_classes"],
            "training_run":False,"model_forward_run":False,"target_masks_read":False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--bank")
    parser.add_argument("--checkpoint")
    parser.add_argument("--recipe")
    parser.add_argument("--expected-bank-sha")
    parser.add_argument("--expected-checkpoint-sha")
    args=parser.parse_args(argv)
    fields=[args.bank,args.checkpoint,args.recipe,args.expected_bank_sha,args.expected_checkpoint_sha]
    if any(fields) and not all(fields):
        parser.error("original-artifact checks require bank, checkpoint, recipe and both expected hashes")
    out=fresh_output(args.output)
    started=time.monotonic()
    receipt={"status":"RUNNING","scope":"LOCAL_SOFTWARE_PREFLIGHT",
             "real_segmentation_evidence":False,"cuda_pilot_run":False,
             "official_test_scored":False,"production_changed":False}
    code=1
    try:
        from .runtime import load_runtime
        runtime=load_runtime()
        import numpy,scipy,torch
        receipt["versions"]={"python":sys.version.split()[0],"numpy":numpy.__version__,
                             "scipy":scipy.__version__,"torch":torch.__version__}
        receipt["experimental_source_sha256"]={
            str(p.relative_to(HERE)):digest(p) for p in sorted(HERE.rglob("*.py"))
        }
        receipt["original_artifacts"] = (
            assert_original_bindings(runtime,*fields)
            if all(fields) else {"status":"NOT_REQUESTED_SOFTWARE_ONLY"})
        command=[sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider","-c","/dev/null",
                 str(HERE/"tests"),"--junitxml="+str(out/"junit.xml")]
        env=os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["PYTHONPATH"]=str(REPO)+os.pathsep+env.get("PYTHONPATH","")
        # Offline contract: no model/network downloads from tests.
        env["HF_HUB_OFFLINE"]="1"
        env["TRANSFORMERS_OFFLINE"]="1"
        env["CUDA_VISIBLE_DEVICES"]=""
        with (out/"pytest.log").open("w") as log:
            run=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt["command"]=command
        receipt["pytest_exit_code"]=run.returncode
        code=run.returncode
        receipt["status"]="PASS_SOFTWARE_ONLY" if code==0 and not all(fields) else (
            "PASS_SOFTWARE_AND_ORIGINAL_IDENTITIES" if code==0 else "FAIL_SOFTWARE")
        receipt["limitations"]=["Synthetic CPU checks are not original-run tensor reproduction.",
            "No target-GPU or anatomical quality claim.",
            "Run prepare and confirm identity/reproduction before any paired pilot.",
            "No full 50-epoch run is launched by preflight."]
    except Exception as exc:
        receipt["status"]="FAIL_CLOSED"
        receipt["error"]=type(exc).__name__+": "+str(exc)
    finally:
        receipt["elapsed_seconds"]=time.monotonic()-started
        (out/"PREFLIGHT.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps({"status":receipt["status"],"exit_code":code,"output":str(out)}))
    return code

if __name__=="__main__":
    raise SystemExit(main())

