"""Synthetic failure probes; run from repository root. No GT enters adapter."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
adapt = runpy.run_path(str(ROOT / 'tests/shared_benchmark/test_adapter_topology.py'))['_adapt']

base = np.zeros((17, 17), dtype=np.int64)
base[6:11, 6:11] = 2
base[7:10, 7:10] = 3
base[7:10, 4:6] = 1
cases = {'ideal': base}
p = base.copy(); p[6:11, 6:11][base[6:11, 6:11] == 2] = 0
cases['bg_myo_same_connected_component'] = p
p = base.copy(); p[6, 8] = 0
cases['one_pixel_ring_gap'] = p
p = base.copy(); p[7:10, 8:10] = 8
cases['lv_split_into_two_clusters'] = p
p = base.copy(); p[1:4, 1:4] = 8; p[2, 2] = 9
cases['additional_remote_enclosure'] = p
p = base.copy(); p[7, 11] = 8
cases['additional_one_pixel_rv_neighbor'] = p
p = base.copy(); p[6:11, 6:11][base[6:11, 6:11] == 2] = 0
p[5:12, 5:12][p[5:12, 5:12] == 0] = 8
# Restore the MYO ring to BG ID, separated from exterior by cluster 8.
p[6:11, 6:11][base[6:11, 6:11] == 2] = 0
cases['bg_myo_same_id_disconnected'] = p

results = []
for name, partition in cases.items():
    result = adapt(partition)
    pred = result.semantic_map
    dice = {}
    for c in (1, 2, 3):
        a, b = pred == c, base == c
        dice[str(c)] = float(2 * (a & b).sum() / (a.sum() + b.sum()))
    results.append({'case': name, 'roles': result.metadata['role_reasons'],
                    'predicted_pixels': {str(c): int((pred == c).sum()) for c in range(5)},
                    'coverage': result.metadata['coverage'],
                    'synthetic_dice': dice})
output = Path(__file__).with_name('synthetic_results.json')
output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
print(json.dumps(results, indent=2))
