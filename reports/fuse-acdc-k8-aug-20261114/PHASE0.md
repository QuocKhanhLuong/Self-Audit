# Phase0 — old A checkpoint diagnosis

GPU inference on the 16 locked L8 ED/ES frames without augmentation. No TEST opened.

| Mode | L8 FG | L8 RV | L8 MYO | L8 LV | historical VAL18 FG |
|---|---:|---:|---:|---:|---:|
| teacher | 94.435 | 94.822 | 92.051 | 96.433 | 80.358 |
| student | 97.464 | 97.560 | 96.318 | 98.514 | 82.407 |
| fusion | 97.802 | 98.037 | 96.764 | 98.605 | 82.566 |

This is a report-only diagnostic. The primary future A-aug comparison uses the original committed VAL18 metrics.
The final diagnostic uses the historical deterministic/TF32 flags and reproduces every historical VAL18 per-patient metric exactly. The initial default-TF32 diagnostic is retained separately; its largest FG deviation was 0.0004 percentage points.
