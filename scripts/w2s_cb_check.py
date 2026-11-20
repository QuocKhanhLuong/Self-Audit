#!/usr/bin/env python3
"""CB fallback rule (§6): read the S3-SA receipt; S2 with codebook only if q_T is close to raw W and perplexity >= 10."""
from __future__ import annotations

import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
rec = json.loads((run / "S3_RECEIPT.json").read_text())
vr = rec["val_reconstruction"]
mean = vr["mean"]
gap = mean["fg_raw_vs_gt"] - mean["fg_rec_vs_gt"]
use_codebook = gap <= 0.05 and vr["perplexity"] >= 10
decision = {"schema": "self_audit.w2s.cb_decision.v1", "use_codebook": bool(use_codebook), "fg_raw_vs_gt": mean["fg_raw_vs_gt"],
            "fg_rec_vs_gt": mean["fg_rec_vs_gt"], "gap": gap, "perplexity": vr["perplexity"], "dead_codes": vr["dead_codes"],
            "rule": "fallback to S2-arb if q_T FG Dice on VAL10 is > 0.05 below raw W or perplexity < 10",
            "frames": vr["frames"], "split": "VAL10"}
(run / "CB_DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print(json.dumps(decision))
sys.exit(0 if use_codebook else 3)
