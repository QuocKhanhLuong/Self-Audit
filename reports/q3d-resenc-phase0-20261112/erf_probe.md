# ResEnc-L empirical receptive-field probe

This CPU-only probe used deterministic random Z-score inputs and the locked encoder checkpoint. It is architecture evidence, not a patient experiment.

Status: **PASS**. Samples: 16. Absolute mean centroid offset from nominal main phase (per-axis max): **1.9470 voxels**; threshold: 2.0000.

Mean centroid offset (Z,Y,X): `[0.67213374376297, 1.9470438957214355, 1.722203016281128]`; maximum individual absolute offset (reported, not gated): `5.1300`. Mean r90: `48.4225` voxels; mean r95: `52.6782` voxels.

Logical probe shape: `[33, 97, 97]`; the full six-stage teacher shape contract pads high-side to `[34, 128, 128]` with `[1, 31, 31]` voxels using `replicate`. The selected logical stage-3 cell remains `[8,6,6]` with nominal phase `[16,48,48]`.

Channel aggregation is the signed mean of all stage-3 channels before differentiation; the centroid uses absolute input gradients. The full per-sample receipt, checkpoint hash, registry hash, limitations, and exact command parameters are in `erf_probe.json`.
