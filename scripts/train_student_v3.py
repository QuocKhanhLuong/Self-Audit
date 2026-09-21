#!/usr/bin/env python3
"""Train only from frozen TRAINING pseudo-labels; never silently include development."""
from self_audit_pseudolabel.pipeline_v3 import FrozenPseudoDataset,student_main
if __name__=='__main__': student_main()
