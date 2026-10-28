#!/bin/bash
set -euo pipefail
cd /workspace/Self-Audit
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/check_repo.py 2>&1 | tee reports/automatic_pilot_pro4000/check_repo_postrun.log
python -m pytest -q 2>&1 | tee reports/automatic_pilot_pro4000/pytest-B128-postrun.log
python reports/automatic_pilot_pro4000/build_report.py
python reports/automatic_pilot_pro4000/finalize_report.py
echo POSTRUN_VERIFY_COMPLETE
