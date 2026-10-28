#!/usr/bin/env python3
"""Attach raw-cine-verified affine metadata to a historical evaluation manifest."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from self_audit.compact_geometry import attach_geometry

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','manifest-sha256','sources','sources-sha256','out'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args()
    print(json.dumps(attach_geometry(a.manifest,a.manifest_sha256,a.sources,a.sources_sha256,a.out),indent=2))

if __name__=='__main__': main()
