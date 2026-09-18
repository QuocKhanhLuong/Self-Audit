#!/usr/bin/env python3
"""Image-only runner. Use a separate process for frozen-reference evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from self_audit_nogt.runner import main
if __name__ == "__main__":
    main()
