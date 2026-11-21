# TPC-Gate offline analysis on frozen T-M3HL (pre-registered; no training)

val10 chooses q only; U63 ground truth is a diagnostic (never selection/training); test20 not used. Signals/errors/flags live in the aligned frame
(translation of slices), so Dice/HD95 here are not identical to unaligned values. n = 20 val10 and 126 U63 volumes (ED/ES of 63 patients), one teacher, one seed.

## E0 error map

**val10**: error rate in ROI 0.1015; errors with c_T >= 0.99: 11.6%; error rate base/mid/apex terciles 0.1061 / 0.0845 / 0.1310; class Dice RV 0.833 MYO 0.883 LV 0.922.

**U63**: error rate in ROI 0.1182; errors with c_T >= 0.99: 16.8%; error rate base/mid/apex terciles 0.1220 / 0.0921 / 0.1684; class Dice RV 0.852 MYO 0.852 LV 0.912.

E0 GO (enough errors at high confidence or concentrated by position): val10 **GO**, U63 **GO**.

## E1 AUROC for 'teacher voxel is wrong' (patient-bootstrap 95% CI; ROI = 5 px around predicted FG)

### val10

| signal | variant | overall | boundary band | inside c_T >= 0.99 |
|---|---|---|---|---|
| tpc | generic | 0.633 [0.594, 0.668] | 0.587 | 0.647 |
| s_flip | generic | 0.689 [0.657, 0.719] | 0.607 | 0.659 |
| s_area | generic | 0.586 [0.523, 0.648] | 0.562 | 0.625 |
| s_topo | generic | 0.548 [0.521, 0.580] | 0.532 | 0.558 |
| s_tta | generic | 0.820 [0.789, 0.852] | 0.777 | 0.668 |
| one_minus_conf | - | 0.881 [0.866, 0.898] | 0.792 | 0.821 |
| entropy | - | 0.881 [0.866, 0.897] | 0.793 | 0.820 |
| one_minus_margin | - | 0.881 [0.865, 0.898] | 0.792 | 0.822 |
| slice_position | - | 0.570 [0.537, 0.606] | 0.543 | 0.611 |
| tpc | cardiac | 0.632 [0.593, 0.667] | 0.587 | 0.644 |
| s_flip | cardiac | 0.686 [0.654, 0.716] | 0.607 | 0.657 |
| s_area | cardiac | 0.586 [0.523, 0.648] | 0.562 | 0.625 |
| s_topo | cardiac | 0.552 [0.525, 0.583] | 0.534 | 0.558 |

Best TPC-family signal: s_flip (generic) AUROC 0.689; best baseline: one_minus_margin 0.881; delta -0.192; best TPC inside c_T >= 0.99: 0.659.

E1 val10: **NO-GO** (kill if delta < 0.05 or AUROC inside c_T >= 0.99 < 0.65).

### U63

| signal | variant | overall | boundary band | inside c_T >= 0.99 |
|---|---|---|---|---|
| tpc | generic | 0.659 [0.647, 0.671] | 0.605 | 0.669 |
| s_flip | generic | 0.694 [0.684, 0.705] | 0.613 | 0.656 |
| s_area | generic | 0.621 [0.600, 0.640] | 0.595 | 0.638 |
| s_topo | generic | 0.555 [0.543, 0.567] | 0.537 | 0.568 |
| s_tta | generic | 0.807 [0.798, 0.816] | 0.767 | 0.651 |
| one_minus_conf | - | 0.866 [0.859, 0.872] | 0.776 | 0.803 |
| entropy | - | 0.866 [0.860, 0.873] | 0.777 | 0.803 |
| one_minus_margin | - | 0.865 [0.858, 0.872] | 0.775 | 0.801 |
| slice_position | - | 0.587 [0.573, 0.601] | 0.559 | 0.636 |
| tpc | cardiac | 0.659 [0.646, 0.672] | 0.606 | 0.668 |
| s_flip | cardiac | 0.687 [0.678, 0.698] | 0.610 | 0.650 |
| s_area | cardiac | 0.621 [0.600, 0.640] | 0.595 | 0.638 |
| s_topo | cardiac | 0.562 [0.549, 0.577] | 0.542 | 0.575 |

Best TPC-family signal: s_flip (generic) AUROC 0.694; best baseline: entropy 0.866; delta -0.172; best TPC inside c_T >= 0.99: 0.656.

E1 U63: **NO-GO** (kill if delta < 0.05 or AUROC inside c_T >= 0.99 < 0.65).

## E2 oracle headroom (GT replaces the teacher label at flagged voxels; TPC variant = best TPC family on val10 per E1)

### val10

| q | flagged fraction of ROI | Dice before | Dice after | gain | HD95 mm before | after | gain |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.314 | 0.8794 | 0.9286 | +4.93 pt | 4.75 | 2.83 | +1.91 |
| 0.9 | 0.280 | 0.8794 | 0.9255 | +4.61 pt | 4.75 | 2.85 | +1.90 |
| 0.95 | 0.259 | 0.8794 | 0.9213 | +4.20 pt | 4.75 | 2.91 | +1.84 |

### U63

| q | flagged fraction of ROI | Dice before | Dice after | gain | HD95 mm before | after | gain |
|---|---|---|---|---|---|---|---|
| 0.8 | 0.284 | 0.8717 | 0.9245 | +5.28 pt | 6.52 | 3.18 | +3.34 |
| 0.9 | 0.234 | 0.8717 | 0.9154 | +4.37 pt | 6.52 | 3.38 | +3.13 |
| 0.95 | 0.219 | 0.8717 | 0.9123 | +4.06 pt | 6.52 | 3.47 | +3.04 |

q chosen on val10 (max Dice gain): **0.8**. E2 val10 **GO**, U63 **GO** (kill if gain < 0.5 Dice pt and < 0.5 mm HD95).

## E3 student feasibility (flagged voxels where student and teacher disagree; existing checkpoints)

| split | student | disagreeing flagged voxels | student right | teacher right |
|---|---|---|---|---|
| val10 | S0 | 36895 | 46.3% | 51.3% |
| val10 | S2arbM3 | 27700 | 51.0% | 45.7% |
| U63 | S0 | 220614 | 51.3% | 43.4% |
| U63 | S2arbM3 | 154138 | 63.5% | 32.2% |

E3: override **NO-GO (drop-only)** (kill override if the student is right < 55% on both val10 and U63 for every student checkpoint).

## GO / NO-GO summary

| experiment | val10 | U63 |
|---|---|---|
| E0 | GO | GO |
| E1 | NO-GO | NO-GO |
| E2 | GO | GO |
| E3 | NO-GO | GO |
