#!/bin/bash
set -euo pipefail
cd /workspace/Self-Audit
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PILOT_BATCH_SIZE=128 PILOT_MICROBATCH_SIZE=40
python - <<'PY'
import json
from pathlib import Path
from self_audit.automatic import automatic_source_digest
p=Path('reports/automatic_pilot_pro4000')
r=json.loads((p/'profile_batch_B128_M40_final.json').read_text())
assert set(r['arms'])=={'full_ema','joint_dice'}
assert r['training_manifest']['sha256']=='e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4'
for arm, item in r['arms'].items():
 t=item['receipt']
 assert t['status']=='COMPLETED' and t['steps']==3 and t['pseudo_transactions_proposed']>0
 assert t['peak_reserved_bytes']<22*1024**3 and t['source_sha256']==automatic_source_digest()
assert '535 passed' in (p/'pytest-B96-final.log').read_text()
assert not Path('runs/pro4000/warm_start_B128_final').exists()
print('B128_FULL_RUN_GATE_PASS', automatic_source_digest(), flush=True)
PY
python reports/automatic_pilot_pro4000/run_stage.py warm_start --steps 512 --suffix _B128_final
export PILOT_WARM_START_RECEIPT=warm_start_B128_final.json
python reports/automatic_pilot_pro4000/run_stage.py equal_update --steps 128 --suffix _B128_M40
python reports/automatic_pilot_pro4000/run_stage.py equal_time --time-budget 600 --suffix _B128_M40
python reports/automatic_pilot_pro4000/evaluate_stage.py equal_update_B128_M40
python reports/automatic_pilot_pro4000/evaluate_stage.py equal_time_B128_M40
echo FULL_ACDC_B128_MATRIX_AND_EVALUATION_COMPLETE
python reports/automatic_pilot_pro4000/build_report.py
echo ALL_ACDC_B128_EXPERIMENTS_COMPLETE
