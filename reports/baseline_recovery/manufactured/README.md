# Manufactured CPU evidence

These files describe a **tiny_test synthetic CPU** integration check. They are
not DINOv3, patient, clinical, PRO4000, convergence or segmentation-quality evidence.
No synthetic checkpoint is presented as a finished model. Absolute temporary
paths inside receipts identify the executed session; they are not portable inputs.
Figure provenance retains the original output filenames; the packaged overlays
are renamed `overlay_acdc.png` and `overlay_mnm.png`.

- `INTEGRATION_RECEIPT.json`: exact 3-step old-recipe trajectory parity with
  historical source 0f54455 (student/EMA, optimizers, Torch RNG); separate 4-step
  synthetic 4+4 training with one checkpoint evaluated on two domains.
- `EXPORT_RECEIPT.json`: student-only export reload equals all 91 source tensors.
- `CPU_PROFILE.json`: actual bounded 3-step timing including first commit; VRAM
  is null because the device was CPU. These numbers cannot size a PRO4000 run.
- Curves and two overlays were rendered from that same evaluation, visually
  inspected, and explicitly watermarked synthetic.

Reproducible behavioral coverage is in the five new recovery test modules. The
runbook provides the physical-data commands. Per-file hashes are in ../VALIDATION.json.
