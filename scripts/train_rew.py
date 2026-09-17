#!/usr/bin/env python3
"""Supervised Read--Evaluate--Write: explicit new model and experiment identity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from self_audit.training.rew_runner import main
if __name__=='__main__':
    raise SystemExit(main())
