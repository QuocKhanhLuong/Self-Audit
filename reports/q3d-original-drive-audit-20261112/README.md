# Original Drive CINE audit — 2026-11-12

Source supplied by the user: [ACDC Drive folder](https://drive.google.com/drive/u/0/folders/1EelzBVjIoDQ4uzt0_2JzmF_PuUHsD93e). All 150 patient folders were enumerated through the Google Drive connector; 150 original CINE `.nii` files (7,104,718,016 bytes) were downloaded separately to `/root/mount_data/audits/acdc-drive-original-20261112/`. The existing dataset was not overwritten.

**150/150 PASS:** exact metadata and NIfTI payload lengths, complete voxel reads, all voxels finite, and full SHA-256 identical to the corresponding server file. All pixdim vectors are finite, positive and nonunit. [Full per-patient evidence](AUDIT.json), [spacing/affine table](AUDIT.csv). No transfer/truncation/nonfinite-payload fault was found in the CINEs. This does not authenticate an independent official release checksum or validate annotation correctness; masks and non-CINE payloads remain NOT_RUN for this integrity audit.

114/150 CINEs already have affine-column lengths disagreeing with pixdim in this source copy: TRAIN80 52/80, DEV18 11/18, TEST50 49/50, and 2/2 patients outside the locked cohort. The two extra patients were audited for source integrity only and were not added to any training/evaluation split.

The separately downloaded `patient010_frame01.nii` also matches the server SHA `18c7e3e26da5213c327ca4b631cf12797d92108a19cc00ed585e6005d9ee2b07`; its pixdim XYZ is (1.5625,1.5625,10), selected affine lengths (1,1,1), sform_code=2, qform_code=0, units unknown. Its full voxel payload is finite.

Current `prepare_acdc.py` builds published geometry from `ci.affine`; `self_audit/preparation.py` computes spacing from affine-column norms. With the explicit mm unit declaration, that path emits unit spacing for these unit-scale sforms. Its later frame/CINE zoom check does not compare CINE pixdim to the selected affine. This explains compatibility with the observed manifests; the exact historical invocation was not independently recovered. The original header inconsistency is distinct from transfer corruption.

The user-approved `acdc_pixdim_native_v1` amendment repairs only Q3D manifest geometry, with unit provenance and raw/reconstructed affine audit, without rewriting original NIfTI files, arrays, labels, selectors, or splits. See [geometry amendment](../q3d-resenc-pixdim-20261112/README.md). This is file-integrity evidence, not segmentation, clinical or training evidence.
