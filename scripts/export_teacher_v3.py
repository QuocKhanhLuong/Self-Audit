#!/usr/bin/env python3
"""Checkpoint-only native export; no training or reference input."""
from self_audit_pseudolabel.inference_v3 import teacher_export_main

if __name__ == "__main__":
    teacher_export_main()
