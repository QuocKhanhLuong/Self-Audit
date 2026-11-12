# ResEnc-L empirical receptive-field probe

This CPU-only probe used deterministic random Z-score inputs and the locked encoder checkpoint. It is architecture evidence, not a patient experiment.

Status: **PASS**. Samples: 16. Absolute mean centroid offset from nominal main phase (per-axis max): **1.7267 voxels**; threshold: 2.0000.

Mean centroid offset (Z,Y,X): `[0.5716968774795532, 1.7267353534698486, 0.867565393447876]`; maximum individual absolute offset (reported, not gated): `2.9461`. Mean r90: `46.6200` voxels; mean r95: `50.8790` voxels.

Logical probe shape: `[33, 97, 97]`; the external residual shape contract pads high-side to `[34, 104, 104]` with `[1, 7, 7]` voxels, preserving the nominal center phase `[16,48,48]`.

Channel aggregation is the signed mean of all stage-3 channels before differentiation; the centroid uses absolute input gradients. The full per-sample receipt, checkpoint hash, registry hash, limitations, and exact command parameters are in `erf_probe.json`.
