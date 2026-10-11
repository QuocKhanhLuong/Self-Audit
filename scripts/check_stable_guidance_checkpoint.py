#!/usr/bin/env python3
"""CI integration probe: real MAE weights + real NIfTI IO, MANUFACTURED image content.

NOT an ACDC/M&Ms experiment and NOT a substitute for server/data preflight.
"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import nibabel as nib
from self_audit_guidance.io import write_json
from self_audit_guidance.core import GuidanceConfig
from self_audit_guidance.preflight import run_preflight


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    raw=out/'manufactured_images';raw.mkdir();rng=np.random.default_rng(123)
    ids=['patient001','patient002','patient003','patient004']
    for pid in ids:
        d=raw/pid;d.mkdir();a=rng.normal(size=(32,28,2,2)).astype(np.float32)
        n=nib.Nifti1Image(a,np.diag([1.3,1.3,10,1]));n.header.set_xyzt_units('mm','sec')
        nib.save(n,d/f'{pid}_4d.nii.gz')
    split=out/'split.json';write_json(split,dict(train=ids[:2],val=ids[2:3],test=ids[3:]))
    config=out/'config.json';write_json(config,GuidanceConfig(grid=8,feature_dim=4,clusters=(2,),temporal=False).dictionary())
    report=run_preflight(raw,'acdc',split,out/'PREFLIGHT.json',config,device='cpu',batch_size=2,width=8,threads=2)
    write_json(out/'SCOPE.json',dict(image_content='MANUFACTURED_NOT_CLINICAL',
        native_nifti_library='GENUINE_NIBABEL',teacher='OFFICIAL_HASH_PINNED_CINEMA_MAE',
        clinical_accuracy_evaluated=False,preflight_status=report['status']))
    return 0 if report['status'].startswith('PASS_') else 1
if __name__=='__main__':raise SystemExit(main())
