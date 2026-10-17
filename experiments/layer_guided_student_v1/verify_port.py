"""Compatibility alias for verification of the current cleaned repository."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([sys.executable, str(root / 'scripts/verify_current.py'), *sys.argv[1:]]))
