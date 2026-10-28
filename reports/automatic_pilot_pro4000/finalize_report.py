"""Write the observed, provenance-aware final research report."""
import csv
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/workspace/Self-Audit')
R = ROOT / 'reports/automatic_pilot_pro4000'

def load(name):
    return json.loads((R / name).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def f(value, digits=3):
    return f'{value:.{digits}f}'

def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '|' + '|'.join(['---'] * len(headers)) + '|'] + [
                '| ' + ' | '.join(map(str, row)) + ' |' for row in rows]

eq_u = load('equal_update_B128_M40.json')
eq_t = load('equal_time_B128_M40.json')
results_u = {x['arm']: x for x in load('results_equal_update_B128_M40.json')}
results_t = {x['arm']: x for x in load('results_equal_time_B128_M40.json')}
mnm = {x['arm']: x for x in load('results_equal_time_B128_M40_MnM_verified_VAL.json')}
extra = load('mnm_full_ema_supplemental.json')
mnm['full_ema'] = load('results_equal_time_B128_M40_MnM_full_ema.json')[0]
dec = load('decision_equal_time_B128_M40.json')['decision_gate']
acdc = load('acdc_integrity.json')
mm = load('mnm_integrity.json')
warm = load('warm_start_B128_final.json')
times = [a['receipt']['phase_cost_seconds'] for a in eq_t['arms'].values()]
assert len(eq_u['arms']) == len(eq_t['arms']) == len(results_u) == len(results_t) == 10
assert all(x['metrics']['total_patients_scored'] == 50 and x['metrics']['prediction_coverage'] == 1 for x in results_t.values())
assert max(times) / min(times) - 1 <= .05
assert dec['decision'] == 'NO_GO' and dec['matched_compute']['status'] == 'VERIFIED_MATCHED'

samples = []
for row in csv.reader((R / 'utilization.csv').open()):
    try:
        stamp = dt.datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f')
        if dt.datetime(2026, 10, 28, 7, 12) <= stamp <= dt.datetime(2026, 10, 28, 10, 25):
            samples.append((stamp, float(row[1]), float(row[2]), float(row[3]), int(row[4])))
    except (ValueError, IndexError):
        pass
assert samples
u = [x[1] for x in samples]
ram = [x[4] / 2**30 for x in samples]
vram = [x[2] for x in samples]
phase_u = sum(a['receipt']['phase_cost_seconds'] for a in eq_u['arms'].values())
phase_t = sum(times)
export_u = sum(x['metrics']['output_export_seconds'] for x in results_u.values())
export_t = sum(x['metrics']['output_export_seconds'] for x in results_t.values())

L = [
    '# Self-Audit real-patient PRO4000 execution', '',
    f"**Decision: {dec['decision']}.** The declared joint/guard/U query gates all failed under verified fresh equal-time compute. This is one pilot seed, not a clinical conclusion.", '',
    '## Dataset and provenance', '',
]
L += table(['Dataset', 'Raw files', 'NIfTI', 'Patients', 'Image/mask pairs', 'Integrity'], [
    ['ACDC', acdc['files'], acdc['nifti'], acdc['patients'], acdc['image_mask_pairs'], acdc['status']],
    ['M&Ms', mm['files'], mm['nifti'], mm['patients'], mm['image_mask_pairs'], mm['status']],
])
L += [
    '', 'ACDC original data: `data/ACDC/`; M&Ms original data: `data/MnM/`; immutable downloaded ZIPs: `data/archives/MnM/`. Remote paths were discovered with `rclone lsd KhanhDrive:` and verified by checksum/CRC. Original files, archive names and directory structures were preserved.',
    '', 'ACDC locked TRAIN80/VAL18/TEST50 split: L8, U72, TEST50 has 100 labeled frames. Two image-duplicate training IDs (055, 076) were already excluded by the reviewed split. M&Ms original TRAIN175/VAL34/TEST136 split is unchanged; eight duplicate-image IDs were quarantined only in derived manifests, yielding active TRAIN170/VAL32/TEST135. Generalization uses VAL32. No M&Ms data entered ACDC training. TRAIN-U manifests have no mask paths and no U GT reads; TEST/VAL GT was opened only by the offline scorer after prediction export.',
    '', 'M&Ms label mapping: original 1→LV(3), 2→MYO(2), 3→RV(1). All 690 NIfTI files have zero temporal spacing; it is treated as unknown and no physical time is invented. Image/mask shape, affine, spatial zoom, orientation and class checks passed. Separate loading/preprocessing smoke tests passed for both datasets.',
    '', f"Training manifest SHA256: `{eq_u['training_manifest']['sha256']}`. ACDC TEST50 manifest SHA256: `8f89d631bc67f7eccef94740866513fe6f794872e7cb6e61302b1663e17e8f9e`. M&Ms prepared train manifest SHA256: `a7fc92227c555d0de3895b5b37154e49bd3de5b44d26958bde34863d86dfaae3`.",
    '', 'DINOv3 ConvNeXt-Tiny source commit `6876159a11b4df116f30f667f8c9888617df0751`; authorized backup weight SHA256 `21b726bb286e037f00a23fb4699fa9bda9c75b6a615bd57ce3013cec1b528d54`. This is self-supervised LVD1689M pretraining; no historical segmentation teacher or random substitute was used. Independent upstream full-weight-hash certification is unavailable.',
    '', '## Execution and hardware', '',
    'Base commit `93f7c7a1f9606a68955f849dd580b90462ad329e`; executable overlay `SOURCE_PATCH_B128_FINAL.patch` SHA256 `' + sha(R / 'SOURCE_PATCH_B128_FINAL.patch') + '`. No model architecture or canonical engine changes. PyTorch 2.14.1+cu132 on RTX PRO 4000 Blackwell 24GB; reuse `/venv/main`. FP32, no AMP. Batch 128 in all arms; full_ema uses four microbatches of up to 40 with one globally normalized optimizer step and one EMA step. Shared support-only warm start: 512 steps/27.37s, manifest SHA256 `' + warm['manifest_sha256'] + '`.',
    '', 'Resource sampling during core B128 execution: ' + str(len(samples)) + ' samples, GPU utilization mean ' + f(statistics.mean(u), 1) + '%, median ' + f(statistics.median(u), 1) + '%, sampled max device memory ' + f(max(vram), 0) + ' MiB; cgroup RAM peak ' + f(max(ram), 1) + ' GiB of 120.7 GiB. Arm-level PyTorch peak reserved VRAM was 20616 MiB for head arms and 19770 MiB for full_ema. GPU samples include startup/extraction/IO and therefore are lower than active-kernel utilization.',
    '', 'Measured phase sums: equal-update ' + f(phase_u, 1) + 's; fresh equal-time ' + f(phase_t, 1) + 's; shared warm ' + f(warm['receipt']['phase_cost_seconds'], 1) + 's once. Native ACDC export sums: equal-update ' + f(export_u, 1) + 's, equal-time ' + f(export_t, 1) + 's. These are measured costs, not a sum claimed as a single uninterrupted job wall clock. Fresh equal-time phases span ' + f(min(times), 2) + '–' + f(max(times), 2) + 's, spread ' + f(100 * (max(times) / min(times) - 1), 2) + '% (<5%). Equal-update phase times differ and are not equal-compute evidence.',
    '', '## ACDC TEST50 per-arm native patient Dice', '',
]
rows = []
for arm in eq_t['arms']:
    for name, results in [('equal-update', results_u), ('fresh equal-time', results_t)]:
        x = results[arm]
        m, t = x['metrics'], x['training_report']
        rows.append([name, arm, t['steps'], f(m['macro_patient']['RV']['dice']), f(m['macro_patient']['MYO']['dice']), f(m['macro_patient']['LV']['dice']), f(m['foreground_patient_macro']), f(t['phase_cost_seconds'], 1), t['pseudo_transactions_accepted']])
L += table(['Regime', 'Arm', 'Steps', 'RV', 'MYO', 'LV', 'FG mean', 'Phase s', 'Pseudo accepted'], rows)
L += ['', 'All 20 ACDC evaluations scored 50/50 patients, 100/100 annotated frames, with prediction and scoring coverage 1.0. Native predictions and manifest/checkpoint hashes are in `runs/pro4000/evaluation_*/` and `evaluation_index_*.json`; patient-level Dice is in `PATIENT_METRICS.csv`.',
      '', '## Failure analysis', '']
L += [
    'All four guarded arms accepted zero pseudo transactions in both regimes and predicted background on every scored ACDC query frame. The frozen, support-only, plain_head_u, extra_supervised and smaller_lr controls also produced zero foreground Dice. This supports a foreground-collapse diagnosis for the fixed-encoder/head-only pilot under this initialization and budget; the reported scores do not isolate which individual component caused it.',
    '', 'The stronger full_ema executable comparator trains encoder/body despite frozen-encoder wording in the runbook. It was the only nonzero ACDC predictor: equal-update FG ' + f(results_u['full_ema']['metrics']['foreground_patient_macro']) + ', equal-time FG ' + f(results_t['full_ema']['metrics']['foreground_patient_macro']) + '. For equal-time full_ema, RV/MYO/LV are ' + '/'.join(f(results_t['full_ema']['metrics']['macro_patient'][c]['dice']) for c in ('RV','MYO','LV')) + '. The worst five patient foreground means are recorded below; per-patient and per-class details remain in the CSV.',
]
vals = []
for pid, per in results_t['full_ema']['metrics']['patients'].items():
    scores = [per[c] for c in ('RV', 'MYO', 'LV') if per[c] is not None]
    vals.append((statistics.mean(scores), pid))
L += table(['Patient', 'FG Dice'], [[pid, f(v)] for v, pid in sorted(vals)[:5]])
L += ['', '## M&Ms VAL32 external generalization', '']
L += table(['Arm', 'RV', 'MYO', 'LV', 'FG mean', 'Patients'], [[a, *[f(x['metrics']['macro_patient'][c]['dice']) for c in ('RV','MYO','LV')], f(x['metrics']['foreground_patient_macro']), x['metrics']['total_patients_scored']] for a, x in mnm.items()])
L += ['', 'The four declared primary arms score 0 foreground Dice on VAL32. The full_ema row is a supplemental separate-dataset comparator, exported and scored with the same verified adapter; it does not alter the ACDC GO/NO-GO gate. M&Ms original TEST136 was not scored.',
      '', '## Decision and limits', '',
      'Offline equal-time gate: **' + dec['decision'] + '**; joint>scalar=' + str(dec['gates']['joint_beats_scalar']) + ', guarded Dice>smaller LR=' + str(dec['gates']['dice_guard_beats_smaller_lr']) + ', U>extra supervised=' + str(dec['gates']['u_beats_extra_supervised']) + '. The equal-update gate is `INCONCLUSIVE_NOT_RUN` by design because its actual compute was unmatched.',
      '', 'Single label seed and one run per arm; no uncertainty interval. ACDC Drive TEST50 was previously used in this repository and is not a newly unseen confirmatory cohort; challenge provenance is not independently certified. M&Ms VAL32 is a verified but nonofficial filtered subset after duplicate quarantine. No M&Ms retraining or pooling. The implementation discrepancy in full_ema makes it a stronger comparator. W&B credentials were unavailable; tmux log, JSONL steps, utilization CSV, checkpoint manifests and native prediction receipts provide the audit trail.',
      '', '## Reproduction and paths', '',
      'Run `bash reports/automatic_pilot_pro4000/execute_B128.sh` from `/workspace/Self-Audit` in the existing `/venv/main` only when creating a new output tree; the original output directories are immutable. Per-arm config overrides are in `run_stage.py`, model source/weight hashes in `model.json`, numeric backend in `numeric_backend.json`, data checks in `acdc_integrity.json` and `mnm_integrity.json`. Checkpoints and native predictions: `/workspace/Self-Audit/runs/pro4000/`. Reports: `/workspace/Self-Audit/reports/automatic_pilot_pro4000/`. Tmux window: `ssh_tmux:self-audit-pilot`. Raw patient data and checkpoint binaries were not pushed to Git.',
]
post = R / 'pytest-B128-postrun.log'
if post.exists():
    lines = post.read_text().splitlines()
    summary = next((line for line in reversed(lines) if ' passed' in line), 'UNKNOWN')
    L += ['', 'Post-run full test suite: `' + summary + '`.']
(R / 'FINAL_REPORT.md').write_text('\n'.join(L) + '\n')
print('FINAL_REPORT_WRITTEN', len(L), 'lines', len(rows), 'ACDC table rows')
