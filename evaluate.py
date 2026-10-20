#!/usr/bin/env python3
import argparse,json
from self_audit.evaluation import evaluate
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Frozen v2 development/external evaluation, no training')
    p.add_argument('--checkpoint',required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--data',required=True);p.add_argument('--data-sha256',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--threshold',type=float)
    a=p.parse_args();r=evaluate(a.checkpoint,a.checkpoint_sha256,a.data,a.data_sha256,a.out,a.device,a.threshold)
    print(json.dumps(r['macro_patient'],indent=2))
