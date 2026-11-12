# ResEnc-L empirical receptive-field probe

This CPU-only probe used deterministic random Z-score inputs and the locked encoder checkpoint. It is architecture evidence, not a patient experiment.

Status: **REVIEW_MEAN_OFFSET_OVER_THRESHOLD**. Samples: 16. Absolute mean centroid offset from nominal main phase (per-axis max): **2.1978 voxels**; threshold: 2.0000.

Mean centroid offset (Z,Y,X): `[0.5374372005462646, 2.197835922241211, 1.668520450592041]`; maximum individual absolute offset (reported, not gated): `4.3087`. Mean r90: `45.8700` voxels; mean r95: `50.0857` voxels.

Channel aggregation is the signed mean of all stage-3 channels before differentiation; the centroid uses absolute input gradients. The full per-sample receipt, checkpoint hash, registry hash, limitations, and exact command parameters are in `erf_probe.json`.
