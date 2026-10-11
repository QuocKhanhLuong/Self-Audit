#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_audit_guidance.full import derive_split
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Explicit new holdout; preserves old split file and TRAIN identities')
    p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=42)
    derive_split(**vars(p.parse_args()))
