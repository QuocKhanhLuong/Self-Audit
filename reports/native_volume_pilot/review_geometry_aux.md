# Independent Astra geometry and auxiliary review

Status: **ACCEPTED for the reviewed local CPU software scope at the final hashes below**. No open blocker remains in this review scope; real-patient, pretrained-weight and target-GPU evidence remain NOT RUN.

Current disposition (historical failures and repair receipts are retained below):

| Scope | Independent disposition |
|---|---|
| Geometry/data G1–G4 | Physical/source/sampling and bounded array-cache repairs independently passed at final scoped hashes. |
| Region R1–R3 | Per-sample partitioning, detached targets, boundary handling and combined batched-resume/capacity regression independently passed. |
| Temporal T1/T2/T3 | Motion/support/detachment, constant-content rejection and per-patient physical displacement metric independently passed. |
| Correction C1–C7 | Mask authorization, physical support, prompt refinement, manual persistence and strict temporal frame selection independently passed. |
| Native evaluation I1–I4 | Frame/patient aggregation, configured support, tile/slab coverage and image-only binding independently passed at final scoped hashes. |

Acceptance applies to the exact final source hashes and supported contracts below. All 31 independent probes pass, and all 67 code/config/test files in the coordinator post-CLI gate match current bytes; its complete repository regression reports 422 passed with one existing deprecation warning. Earlier failures below remain historical evidence with subsequent closures.

Reviewer dispatch: `task_041ea9f6373b` / `ctx_a1f3944f22c6`, terminal `term_3f856468-6fc1-4b86-9b48-817902529036`. Base HEAD: `486fae40c490e382d58896442215476d7d32095e`, branch `feat/native-volume-ema-pilot`. Shared uncommitted source is changing under the original owners; a base commit alone does not identify reviewed bytes. Only this report is edited by the reviewer. No source fixes, commits, downloads, server operations, long training, or non-Orca workers were performed.

Read the entire `docs/native_3d/ASTRA_IMPLEMENTATION_PROMPT.md`, `IMPLEMENTATION_CONTRACT.md`, `ACTIVE_PIPELINE.json`, and `docs/METHOD.md`. Reviewed all of `geometry.py`, `data.py`, `fixtures.py`, `regions.py`, `temporal.py`, `correction.py`, `scripts/volume_prepare.py`, `scripts/volume_correct.py`, and their owned test files. Also reviewed native evaluation and its owned tests, and inspected native export integration. Used Semble for discovery; this worktree has no `.codegraph` index. Historical A0 and all previous reports/figures are preserved.

## Evidence boundary

All results below are local CPU software checks or manufactured numerical/geometry evidence. Python `/Users/alvinluong/miniforge3/bin/python`, PyTorch 2.14.0, NumPy 2.4.6, SciPy 1.17.1. The server is not rented. Real patient geometry validation, real TRAIN smoke, patient Dice, target-GPU memory/throughput and accuracy are **NOT RUN**. No real U/VAL/TEST mask was opened. Instrumented forbidden-read probes use mocked arrays/path sentinels. No pretrained weight was used in this scope; weight hashes are **NOT APPLICABLE / NOT LOADED**, not an efficacy receipt.

## Initial executed checks

- `rtk proxy python -m pytest -q tests/test_volume_geometry.py tests/test_volume_data.py tests/test_volume_regions.py tests/test_volume_temporal.py tests/test_volume_correction.py`: **47 passed in 5.65 s**.
- `rtk proxy python scripts/check_repo.py`: **PASS**.
- `rtk proxy python -m pytest -q`: **263 passed, 1 failed, 1 warning in 26.46 s**. Failure: `tests/test_volume_models.py::test_planar_control_has_active_per_slice_parameters`, `CrossZResidual` rejects planar_control with depth_kernel=3. Model source changed during the run; this is an actual mixed-worktree check, not a final frozen-source result. Routed to coordinator/model owner.
- Independent asymmetric physical phantom: affine `[[0,-.9,0,30],[1.3,0,0,-12],[0,0,-4.2,10],[0,0,0,1]]`, shape TZYX=(3,5,7,9), scalar field `.03*z+.02*y+.04*x+.05*t`. Crop (5,7) plus X flip/native inversion: **175 supported voxels, max error 0**. Resampling to XYZ spacing (2.6,1.8,4.2) then native inversion: **315 supported voxels, max error 5.96046448e-8**. These are continuous-field invariants, not claims of recovering arbitrary lossy data.
- World point XYZ=(6,2,3) maps to (28.2,-4.2,-2.6) mm and resolves to ZYX=(3,2,6). Unequal spacing, flipped Z and permuted XY are exercised.
- Region directional derivative against independent central finite difference: loss 0.10712060229440631, autograd 0.00014191394084957178, numerical 0.0001419139408070924, absolute error **4.24794e-14**. Invalid-support student gradient exactly zero; image/teacher/features gradients absent. This passes fixed-target gradient correctness, not cache validity.
- Temporal derivative on actual dx=2 motion: loss 0.23196743428707123, autograd 5.182602035347372e-5, finite difference 5.21540641784668e-5, absolute error **3.28044e-7** (FP32). Teacher/current/adjacent image gradients absent. Global translation registration recovers dx=2 with **warp MAE 0** on accepted support.
- Same-sample region cache state roundtrip: exact region labels and target probabilities. Different-sample cache correctness fails below, so this is not a general resume approval.

## Initial actionable blockers (line numbers refer to initial reviewed source)

### G1 — P1: sparse labeled cine frames cannot be sampled safely

`self_audit/volume/data.py:199-204` chooses t from all cine frames, then requires a mask. With fixture shape (10,3,5,7), keeping only L frame masks 0 and 5, `sample_batch(...rng=default_rng(0),labeled=True,slab_depth=3,crop_size=0)` raises `ValueError: labeled record has no mask for frame 8`. This is the ordinary sparse ED/ES-in-cine case. Select labeled frame IDs explicitly, preserve patient schedules, and never copy ED/ES labels to another t. Geometry must bind selected frame IDs to the prepared cine mapping.

### G2 — P1: clipped Z padding becomes valid label/region exposure

`data.py:166-171,232-237` clips out-of-range slab indices to edge slices and marks every copy valid. For center=0, depth=5, native Z=3, indices are `[0,0,0,1,2]`, each with 35 valid voxels. Physical distance algorithms see repeated planes as distinct spaced slices and objectives overcount padded supervision. Keep edge image context if needed, but mask padded centers invalid and labels255; retain explicit raw indices/native correspondence for export.

### G3 — P1: native inverse support admits partial invalid interpolation stencils

`geometry.py:225-226` uses linear probability interpolation but nearest validity interpolation. Native width5/current width3 at spacingX2, cropped current x1:3, constant one-hot class1: native support is `[false,true,true,true,true]`, probability sums `[0,.5,1,1,1]`. Native x1 mixes an unsupported neighbor while being labeled valid. Reconstruct support from emitted geometric support and require the full interpolation stencil valid. Do not silently fabricate missing probability mass or count padding.

### G4 — P1: physical provenance is not actually bound or verified

`data.py:147-154` validates shape/affine but ignores sidecar `image_sha256`. Setting the L sidecar image hash to 64 zeros and its affine translation to 999 still loads and samples that geometry. `scripts/volume_prepare.py` reads only prepared arrays, does not validate L mask/frame/cine affine or voxel correspondence, assigns `units=mm` and range(T), and accepts `synthetic=True` for a manifest marked real, manufacturing identity affines. Require source hash binding, explicit temporal/native geometry, and actual checked source correspondence or fail closed. No real NIfTI image/mask/frame correspondence was verified by this initial code or this review.

### R1 — P1: cached teacher targets can cross patient/crop/source boundaries

`regions.py:572-584` calculates current_hash then ignores it; shape+cadence alone allows reuse. Repro: `RegionManager(mode=ema,refresh_every=5)`, step0 patientA zeros/probability class1; step1 patientB ones/probability class3, same shape: returned object is the exact A object, target class1, refresh_count1. `compute_cache_hash:98-107` hashes sparse tensor samples and wrong metadata keys `patient_id/frame_index` instead of loader `patient/frame`, omitting crop, physical grid and source identity. Need full sample identity/support/config hashing separated from teacher refresh version, explicit changed-sample invalidation, and changing-sample resume tests. Additionally `targets:603-610` silently assumes 1mm when geometry is missing; `mode=ema` accepts no teacher/features and degenerates to image mode. These unsupported situations must fail closed.

### R2 — P1: rejecting boundary voxels still pools conflicting class interiors

`regions.py:306-350,417-420`. Image zeros(1,1,8,8,8), teacher class1 on x<4/class3 on x>=4, target_num_regions1, spatial_weight1000, boundary_rejection=True: one region crosses the confident class boundary; weights `[1,1,1,0,0,1,1,1]` along x. A student matching teacher still incurs loss0.4990006685 and gradient toward the wrong class in the interior (-1.30013e-6 on left class3 logit). Reject/downweight the crossing region as a whole or split its components under the declared image/prediction-only rule; merely removing the interface still smooths both classes together.

### T1 — P1: invalid support influences registration, partial stencils influence targets

`temporal.py:227-341` takes current_valid/adjacent_valid but never uses them to estimate correspondence. Valid anatomy translated dx2 stays fixed while adding a bright invalid-band texture translated dx6 changes estimated dx to6.0001845. `:449-453` uses nearest validity with bilinear teacher probabilities: a forced dx=.25 samples an invalid adjacent neighbor while the current voxel receives gradient +/-0.166667. Mask registration evidence correctly and conservatively exclude partial invalid interpolation support.

### T2 — P1: clipping and per-slice alignment accept physically wrong correspondences

`temporal.py:221-224` clips an unrestricted phase-correlation peak; true dx6 with bound2 is changed to dx2, with 1156 voxels accepted because symmetric forward/backward clipping cancels. Reject unsupported motion or use a validated bounded estimator; do not convert out-of-model motion into apparently supported supervision. `:283-291` compares same zi even after global dz!=0; `:316-320` evaluates backward flow at unmapped coordinates. On random texture and true XYZ shift (-3,2,1), global mode recovers exactly; slice_wise=True returns arbitrary XY displacements (e.g. [8.69,-6.17,1]) with occlusion_fraction0. Must align matching Z coordinates and evaluate inverse composition at mapped locations, or explicitly fail closed for this unsupported mode. Blank images yield arbitrary flow but all-occluded in the tested initial case; this does not establish robust registration quality.

### C1 — P1: forbidden mask read precedes role rejection

`scripts/volume_correct.py:160+` reads `--simulate-from-gt` before `correction.py:637+` checks the role. Instrument `_load_volume` and invoke `--image IMAGE --simulate-from-gt TEST_MASK --dataset-role TEST`: opened list is `[IMAGE,TEST_MASK]` before ValueError. The initial test asserts only the exception, not access order. The function blacklist also accepts `TRAIN-U`, `TESTING`, and `TYPO` and generates a click. Reject unsupported roles before opening any reference, require a positive explicit allowlist/declared TRAIN-L or manufactured fixture authorization, and test IO sentinels rather than exception text alone.

### C2 — P1: unknown labels/provenance and earlier manual corrections are lost

`correction.py:431-477,598-612`. With prior_labels all255 and manual_only click at one voxel, untouched voxels become background0 with ORIGINAL provenance. With no prompts and all-false geometric_valid, labels remain0 although provenance255. Missing prior likewise creates apparent ORIGINAL background. Preserve unknown/sparse support separately from probability priors. Iterative use `correct(image,[class1 click at x1])` then `correct(image,[class3 click at x2],prior_labels=previous.labels)` changes the old manual point1->3 and provenance to propagated; there is no prior manual/provenance input. Require replay of accumulated prompts with explicit persistence or support hard prior manual masks/provenance. Conflicting edits need a documented explicit policy.

### C3 — P1: physical correction/export guesses geometry and ignores grid mismatches

`correction.py:329,417-423` invents identity affine/1mm spacing when absent. `scripts/volume_correct.py:48-59,132-155` uses raw NIfTI affine without units conversion, silently replaces it with arbitrary sidecar-patient affine, does not check image/source/frame identity and discards prior/GT affine. Synthetic NIfTI declared meters with spacing(.001,.002,.003) loads those numbers as if millimeters. Require known units, matching patient/source/frame geometry, matching prior/reference grids, and explicit refusal when unavailable. NIfTI output also needs declared mm units and preserved native provenance.

### C4 — P2: no-prompt CLI crashes; declared options/export cases are incomplete

No-prompt invocation raises `KeyError: class_distribution` at CLI summary print because the no-prompt result omits it. `.npz` output uses `np.save`, silently producing `.npz.npy`. NIfTI probability/provenance export is tied to output-label branch rather than independently honoring requested paths. PointPrompt weight/prompt_type are accepted but do not alter correction semantics; implement or reject unsupported values explicitly. These are executable-path requirements, not patient accuracy claims.

### I1 — integration warning sent to coordinator

Initial `evaluation.py:53-88` argmaxes sampled logits before any native mapping and globally pools confusion; it does not implement per-frame then per-patient native-grid evaluation. Empty evaluation support returns Dice1 from smoothing. Integration owner is editing this path; do not approve from this initial snapshot. Student-only serialized model export exists, but absence of external weights/cache requirements is reviewed by the integration/model reviewers, not established by the geometry tests alone.

## Initial source SHA-256

| File | Initial inspected SHA-256 |
|---|---|
| self_audit/volume/geometry.py | b7e33e130e0000a6f035b301d3021752ed54999e11e53ce1755429f9c0fde1f2 |
| self_audit/volume/data.py | bbe2c73c586943eccbfda9988ce6f46e8def1b764b3cd8ae140334a02fd7aa1d |
| self_audit/volume/fixtures.py | f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90 |
| scripts/volume_prepare.py | 8217501f5902407aa974925206903928f604f199e7b2c56e16fb7fbe5a7c18a1 |
| self_audit/volume/regions.py | dfaa0016e5fecb9ef863674b06bca8ea24e249d6520caae4848a348977b57dc6 |
| self_audit/volume/temporal.py | 6f0dd2dc6fd47079e90594a1ab7f08b3071ecf66ec89b88a1a23ef5b8dee38cc |
| self_audit/volume/correction.py | f8b1397e9e3816c5b5df0c722f2e99f3548ac2ef5b1b9cf0d8cc3c97e5515cc3 |
| scripts/volume_correct.py | 9b4f467fc94aae7456bbf390927a557ee96ff98cf61443feecc0c47cdc45687b |

## Final independent recheck

Pending owner fix completion and frozen source verification. Initial passing unit tests do not override the adversarial failures above. Final source hashes, rerun results and unresolved limitations will be appended here before lifecycle settlement.


## Replayable independent adversarial checks

The following block is a reviewer-owned harness, not an implementation test mirror. Run it in this repository with:

```sh
rtk proxy python - <<'PY_REPLAY'
from pathlib import Path
text = Path('reports/native_volume_pilot/review_geometry_aux.md').read_text()
code = text.rsplit('<!-- ADVERSARIAL_HARNESS_BEGIN -->',1)[1].split('```python\n')[1].split('\n```')[0]
exec(compile(code, '<independent-review-harness>', 'exec'))
PY_REPLAY
```

<!-- ADVERSARIAL_HARNESS_BEGIN -->
```python
import copy
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume.data import VolumeTrainingData
from self_audit.volume.geometry import sha256_file, invert_probabilities
from self_audit.volume.regions import RegionManager
from self_audit.volume.temporal import register_volumes, temporal_consistency
from self_audit.volume.correction import VolumeCorrector, PointPrompt, simulate_clicks_from_ground_truth
from scripts.volume_prepare import build_sidecar
from scripts import volume_correct as correction_cli

torch.set_num_threads(1)
torch.manual_seed(73)

def metadata(patient='A', shape=(2,4,4)):
    return [{'patient':patient, 'frame':0, 'z_indices':list(range(shape[0])),
        'spacing_xyz_mm':[1,1,4], 'source_image_sha256':patient*64,
        'native_shape_zyx':list(shape), 'current_shape_zyx':list(shape),
        'current_affine_xyz_to_world_mm':np.diag([1,1,4,1]).tolist(),
        'reversible_transform':{'z_indices':list(range(shape[0])), 'flip':False,
            'crop':{'top':0,'left':0,'height':shape[1],'width':shape[2],'pad_lrtb':[0,0,0,0]},
            'current_shape_zyx':list(shape)}}]

def sparse_frames():
    with tempfile.TemporaryDirectory() as td:
        p,ms,g,gs=make_volume_fixture(td,shape_tzyx=(10,3,5,7))
        m=json.loads(p.read_text())
        m['labeled'][0]['masks']={k:v for k,v in m['labeled'][0]['masks'].items() if k in ['0','5']}
        p.write_text(json.dumps(m))
        geo=json.loads(g.read_text());geo['source_manifest_sha256']=sha256_file(p);g.write_text(json.dumps(geo))
        d=VolumeTrainingData(p,sha256_file(p),g,sha256_file(g),synthetic=True)
        for seed in range(8):
            b=d.sample_batch(0,np.random.default_rng(seed),1,labeled=True,slab_depth=3,crop_size=0)
            assert b['metadata'][0]['frame'] in (0,5)

def z_padding():
    with tempfile.TemporaryDirectory() as td:
        d=VolumeTrainingData(*make_volume_fixture(td,shape_tzyx=(2,3,5,7)),synthetic=True)
        b=d._sample_one(d.labeled[0],np.random.default_rng(0),t=0,center_z=0,labeled=True,
            slab_depth=7,crop_size=0,spacing_xyz_mm=None,fov_xyz_mm=None,flip=False)
        assert int(b['geometric_valid'].sum())<=105
        assert torch.all(b['labels'][~b['geometric_valid']]==255)

def source_geometry_binding():
    with tempfile.TemporaryDirectory() as td:
        p,ms,g,gs=make_volume_fixture(td)
        geo=json.loads(g.read_text());geo['patients']['synthetic_volume_000']['image_sha256']='0'*64
        g.write_text(json.dumps(geo))
        try:
            d=VolumeTrainingData(p,ms,g,sha256_file(g),synthetic=True)
            d.sample_batch(0,np.random.default_rng(1),1,labeled=True,slab_depth=1,crop_size=0)
        except ValueError:
            return
        raise AssertionError('mismatched sidecar image hash accepted')

def synthetic_identity_gate():
    with tempfile.TemporaryDirectory() as td:
        p,ms,g,gs=make_volume_fixture(td)
        m=json.loads(p.read_text());m['synthetic']=False;p.write_text(json.dumps(m))
        try: build_sidecar(p,sha256_file(p),synthetic=True)
        except ValueError: return
        raise AssertionError('real manifest accepted synthetic identity geometry')

def cross_patient_cache():
    x=torch.zeros(1,1,2,4,4);v=torch.ones(1,2,4,4,dtype=torch.bool)
    p=torch.zeros(1,4,2,4,4);p[:,1]=1;q=torch.zeros_like(p);q[:,3]=1
    m=RegionManager({'mode':'ema','refresh_every':5,'target_num_regions':2,'min_region_size':1})
    r=m.targets(x,v,metadata('A'),probabilities=p,step=0)
    s=m.targets(x+1,v,metadata('B'),probabilities=q,step=1)
    assert s is not r and torch.all(s.target_probabilities.argmax(1)==3)
    n=RegionManager();n.load_state_dict(copy.deepcopy(m.state_dict()))
    u=n.targets(x+1,v,metadata('B'),probabilities=q,step=2)
    assert torch.equal(s.region_labels,u.region_labels)
    assert torch.equal(s.target_probabilities,u.target_probabilities)

def immutable_targets():
    x=torch.zeros(1,1,2,4,4);v=torch.ones(1,2,4,4,dtype=torch.bool)
    p=torch.softmax(torch.randn(1,4,2,4,4),1)
    m=RegionManager({'mode':'ema','refresh_every':5})
    r=m.targets(x,v,metadata(),probabilities=p);old=r.target_probabilities.clone();p.zero_()
    assert torch.equal(r.target_probabilities,old)

def region_boundary():
    x=torch.zeros(1,1,8,8,8);v=torch.ones(1,8,8,8,dtype=torch.bool)
    p=torch.zeros(1,4,8,8,8);p[:,1,:,:,:4]=1;p[:,3,:,:,4:]=1
    m=RegionManager({'mode':'ema','target_num_regions':1,'spatial_weight':1000,'boundary_rejection':True,'weight':1})
    t=m.targets(x,v,metadata(shape=(8,8,8)),probabilities=p)
    loss,_=m.loss((p*.999+.001/4).log().requires_grad_(),t,v)
    assert float(loss.detach())<1e-4, float(loss.detach())

def registration_invalid_input():
    x=torch.zeros(1,1,5,17,19);torch.manual_seed(73)
    x[:,:,1:4,4:12,4:14]=torch.rand(1,1,3,8,10)
    y=torch.zeros_like(x);y[:,:,1:4,4:12,6:16]=x[:,:,1:4,4:12,4:14]
    v=torch.ones(1,5,17,19,dtype=torch.bool);v[:,:,0:4,:]=False
    cfg={'mode':'motion','slice_wise':False}
    a=register_volumes(x,y,v,v,cfg)
    x[:,:, :,0:4,:]=torch.rand(1,1,5,4,19)*100
    y[:,:, :,0:4,:]=torch.roll(x[:,:, :,0:4,:],6,-1)
    b=register_volumes(x,y,v,v,cfg)
    assert torch.allclose(a.flow_xyz,b.flow_xyz,atol=1e-4)

def registration_out_of_bound():
    x=torch.zeros(1,1,5,17,19);torch.manual_seed(7)
    x[:,:,1:4,3:12,2:10]=torch.rand(1,1,3,9,8)
    y=torch.zeros_like(x);y[:,:,1:4,3:12,8:16]=x[:,:,1:4,3:12,2:10]
    v=torch.ones(1,5,17,19,dtype=torch.bool)
    try: r=register_volumes(x,y,v,v,{'mode':'motion','slice_wise':False,'max_disp_xyz':(2,2,2)})
    except ValueError: return
    assert not (r.in_bounds_mask & ~r.occlusion_mask).any()

def through_z_motion():
    torch.manual_seed(7);x=torch.rand(1,1,7,25,27);y=torch.roll(x,(1,2,-3),dims=(2,3,4))
    v=torch.ones(1,7,25,27,dtype=torch.bool)
    try:r=register_volumes(x,y,v,v,{'mode':'motion','slice_wise':True})
    except ValueError:return
    accepted=r.in_bounds_mask & ~r.occlusion_mask
    if not accepted.any():return
    ref=torch.tensor([-3.,2.,1.])[None,:,None,None,None].expand_as(r.flow_xyz)
    assert (r.flow_xyz-ref).abs()[accepted[:,None].expand_as(ref)].max()<.2

def forbidden_mask_io():
    opened=[]
    def fake(path,key=None):opened.append(path);return np.ones((1,2,2)),np.eye(4)
    with patch.object(correction_cli,'_load_volume',fake),patch.object(sys,'argv',
        ['volume_correct.py','--image','IMAGE','--simulate-from-gt','TEST_MASK','--dataset-role','TEST']):
        try:correction_cli.main()
        except (ValueError,SystemExit):pass
    assert 'TEST_MASK' not in opened,opened

def forbidden_role_allowlist():
    for role in ('TRAIN-U','TESTING','TYPO'):
        try:simulate_clicks_from_ground_truth(np.ones((1,2,2)),np.zeros((1,2,2)),dataset_role=role,max_clicks=1)
        except ValueError:continue
        raise AssertionError('accepted role '+role)

def unknown_retention():
    im=np.zeros((1,1,8),np.float32);labels=np.full(im.shape,255,np.uint8)
    c=VolumeCorrector(propagation_mode='manual_only')
    r=c.correct(im,[PointPrompt((0,0,0))],prior_labels=labels,affine=np.eye(4))
    assert np.all(r.labels[:,:,1:]==255) and np.all(r.provenance[:,:,1:]==255)
    r=c.correct(im,[],prior_labels=labels,affine=np.eye(4),geometric_valid=np.zeros_like(im,bool))
    assert np.all(r.labels==255) and np.all(r.provenance==255)

checks=[sparse_frames,z_padding,source_geometry_binding,synthetic_identity_gate,
        cross_patient_cache,immutable_targets,region_boundary,registration_invalid_input,
        registration_out_of_bound,through_z_motion,forbidden_mask_io,forbidden_role_allowlist,unknown_retention]
results=[]
for check in checks:
    try:check();results.append({'check':check.__name__,'status':'PASS'})
    except Exception as e:results.append({'check':check.__name__,'status':'FAIL','exception':type(e).__name__,'detail':str(e)})
print(json.dumps(results,indent=2))
print(json.dumps({'pass':sum(r['status']=='PASS' for r in results),'fail':sum(r['status']=='FAIL' for r in results)}))
```


## Geometry first repair recheck (2026-10-26)

Owned geometry/data tests: **15 passed in 1.39 s**. Independent sparse-frame, Z-padding, prepared-image source-hash mismatch, and synthetic-preparation gate witnesses now PASS. Inverse support witness now returns `[false,false,true,true,true]` with sums `[0,0,1,1,1]`, resolving initial G3.

**G4 remains NOT ACCEPTED:** (1) real manifest plus synthetic/source-less sidecar still samples; (2) U sidecar mask_sources are forwarded to the raw-mask validator; (3) complete valid raw NIfTI hashes/affines/units for all-zero cine/frame/masks can be attached to unrelated random prepared image and nonzero prepared class masks, and both preparation/loader accept them. Actual raw-to-prepared voxel correspondence is not checked. These manufactured adversarial cases were sent to the coordinator for another owner repair. No source was edited by reviewer.

First-repair inspected hashes (not final acceptance):
- `self_audit/volume/geometry.py`: `18364d5125ec2aa95743f5ef21825ff11c791b65d69631435976e17f39b45034`
- `self_audit/volume/data.py`: `cb9af492ce996d1bbd2b7011e7c6fa6ebb7c9965810c422533379a5b0b5dafa5`
- `scripts/volume_prepare.py`: `00d6e230bc6d455e5f1e8bc1592bc67b6e0292edb0063bdfe2db9bce70565344`


## Region/temporal first repair recheck

`rtk proxy python -m pytest -q tests/test_volume_regions.py tests/test_volume_temporal.py`: **30 passed in 2.48 s**. Independently, cross-patient cache isolation, immutable target storage, invalid-texture registration invariance, out-of-bound motion rejection, and through-Z motion witness PASS. The quarter-voxel invalid-stencil probe now gives zero student gradient at the unsupported location. Region source SHA `d88f8b5a049cd9047ab1aa01e0e5d1862248fee14394688c5a2145396bd13b38`; temporal SHA `cf24a302ddf09e5fb37745f1bcd59003c63af340c04e34cded7f9a11a8e14a20`.

**R2 still FAILS unchanged:** boundary_rejection=True gives loss0.4990006685 on the class-consistent student because the region still crosses confident class boundaries. `mode=ema` with neither teacher probabilities nor features still silently accepts image-only targets. Sent both back for owner repair; no overall auxiliary acceptance yet.


Additional first-repair R1 resume failure: restored config does not restore `self.max_cache_size`. Manager capacity2 caches A,B at step0/class1; checkpoint-resume into default manager; both add C at step1, then query A with class3. Uninterrupted target class3, resumed target class1 (capacity64). Routed to active region owner. Temporal remaining unobservability case: identical constant-one images shape(1,1,3,16,16) in motion mode accept768 voxels with flow max2.3e-9 because apodized support creates a peak; needs content observability gate or explicit unsupported limitation/failclosed.


## Correction first repair recheck

`rtk proxy python -m pytest -q tests/test_volume_correction.py`: **35 passed in 6.84 s**. Independent pre-IO forbidden-role rejection, role allowlist, prior255 preservation, no-prompt CLI success, actual NPZ export, independent NIfTI probabilities/provenance export with mm units, meter-to-mm conversion, and positive manual-edit persistence via `prior_provenance` all PASS. Source `correction.py` SHA `d11cbf81991c705c915f696de6f40298611558f948e5aeb2e123930d2d838055`; CLI SHA `7176bc33a71ef44cc2e29c07b62097f56a82caf6d069a6e1606d33dd293ae425`.

**Remaining C2/C3 failures:** background simulated clicks use `prompt_type=negative`; with prior_labels all3, a negative target0 click returns label3/ORIGINAL/manual_count0 because untouched prior restoration overwrites the negative edit. No-prior/no-prompt result still reports ORIGINAL background, rather than unknown. NIfTI unknown spatial units are silently treated as mm. Direct library correction without geometry and direct NIfTI export still invent 1mm/identity despite the CLI gate. Routed to original owner; these are independent failures despite all35 owned tests passing.


## Geometry final scoped recheck

Geometry/data subset is **ACCEPTED for the bounded software contract** at the hashes below. `rtk proxy python -m pytest -q tests/test_volume_geometry.py tests/test_volume_data.py`: **19 passed in 1.32 s**. Independent sparse-frame, padded-Z, source-hash mismatch, and synthetic-identity witnesses PASS. A synthetic sidecar under a real manifest and any U-sidecar mask references both reject before the raw validator is called (validator call count zero).

A manufactured real-mode positive control with raw NIfTI cine/frame/mask hashes, mm units, affine correspondence, canonical percentile normalization and axis transpose passes preparation and L/U sampling. Replacing prepared image voxels with zeros while updating their hash rejects with `prepared image/raw cine voxel correspondence failed`; replacing prepared labeled mask voxels while updating their hash rejects with `prepared mask/raw mask voxel correspondence failed`. These checks validate real-mode software on fabricated input, not real patient data. The inverse-support fix from the first repair remains in this snapshot.

| File | Scoped accepted SHA-256 |
|---|---|
| self_audit/volume/geometry.py | 3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd |
| self_audit/volume/data.py | 34d1a2727762ae1111865a71cde63cef000c56be4adc8d052116a3a52b321493 |
| scripts/volume_prepare.py | 32929bb2e1fec70dbd6ff9ca2381c6312b2e0a3433a700e9854e486eeefed881 |
| self_audit/volume/fixtures.py | f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90 |
| tests/test_volume_geometry.py | 08c3dda96c2d8abb0896b64774476afc8d737dbae82d7ae408f73e6e95bfcc3f |
| tests/test_volume_data.py | 36fc3f2e107dc8d653a447cde0064e64de3a869ef3528f5c423d29dfe21a7a76 |

Correction follow-up: a no-op with prior labels `[0,1,2,255,3]` and provenance `[1,4,3,255,2]` preserves labels but rewrites provenance to `[1,1,1,255,2]`, destroying pseudo/propagated provenance. This is assigned with the remaining correction repair. TRAIN-L simulator authorization must bind to the declared manifest membership rather than trust role/budget strings alone; coordinator has included this in the same repair.


## Additional independent repair witnesses

These witnesses extend the first harness and remain pending until the relevant owner-ready snapshot is rechecked. They use only manufactured arrays.

<!-- AUXILIARY_REPAIR_HARNESS_BEGIN -->
```python
from pathlib import Path
source = Path('reports/native_volume_pilot/review_geometry_aux.md').read_text()
base = source.rsplit('<!-- ADVERSARIAL_' + 'HARNESS_BEGIN -->',1)[1].split('```python\n')[1].split('\n```')[0]
exec(compile(base[:base.index('checks=[sparse_frames')], '<review-base>', 'exec'))

def cache_lru_resume():
    x=torch.zeros(1,1,2,4,4);v=torch.ones(1,2,4,4,dtype=torch.bool)
    p=torch.zeros(1,4,2,4,4);p[:,1]=1;q=torch.zeros_like(p);q[:,3]=1
    a=RegionManager({'mode':'ema','refresh_every':5,'max_cache_size':2})
    a.targets(x,v,metadata('A'),probabilities=p,step=0)
    a.targets(x+1,v,metadata('B'),probabilities=p,step=0)
    b=RegionManager();b.load_state_dict(copy.deepcopy(a.state_dict()))
    for manager in (a,b):manager.targets(x+2,v,metadata('C'),probabilities=p,step=1)
    ra=a.targets(x,v,metadata('A'),probabilities=q,step=1)
    rb=b.targets(x,v,metadata('A'),probabilities=q,step=1)
    assert a.max_cache_size==b.max_cache_size==2
    assert torch.equal(ra.target_probabilities,rb.target_probabilities)
    assert torch.all(rb.target_probabilities.argmax(1)==3)

def ema_requires_teacher():
    x=torch.zeros(1,1,2,4,4);v=torch.ones(1,2,4,4,dtype=torch.bool)
    try:RegionManager({'mode':'ema'}).targets(x,v,metadata('A'),step=0)
    except ValueError:return
    raise AssertionError('ema mode silently accepts no teacher probabilities/features')

def constant_content_motion():
    x=torch.ones(1,1,3,16,16);v=torch.ones(1,3,16,16,dtype=torch.bool)
    try:r=register_volumes(x,x,v,v,{'mode':'motion','slice_wise':False})
    except ValueError:return
    assert not (r.in_bounds_mask & ~r.occlusion_mask).any()

def background_manual_persistence():
    c=VolumeCorrector(propagation_mode='manual_only');x=np.zeros((1,1,5),np.float32)
    prior=np.full(x.shape,3,np.uint8)
    r=c.correct(x,[PointPrompt((2,0,0),target_class=0,prompt_type='negative')],
                prior_labels=prior,affine=np.eye(4))
    assert r.labels[0,0,2]==0 and r.provenance[0,0,2]==2
    s=c.correct(x,[PointPrompt((0,0,0),target_class=1)],prior_labels=r.labels,
                prior_provenance=r.provenance,affine=np.eye(4))
    assert s.labels[0,0,2]==0 and s.provenance[0,0,2]==2

def no_prior_is_unknown():
    c=VolumeCorrector(propagation_mode='manual_only')
    r=c.correct(np.zeros((1,2,3),np.float32),[],affine=np.eye(4))
    assert np.all(r.labels==255) and np.all(r.provenance==255)

def untouched_provenance():
    labels=np.array([[[0,1,2,255,3]]],np.uint8)
    provenance=np.array([[[1,4,3,255,2]]],np.uint8)
    r=VolumeCorrector(propagation_mode='manual_only').correct(
        np.zeros(labels.shape,np.float32),[],prior_labels=labels,
        prior_provenance=provenance,affine=np.eye(4))
    assert np.array_equal(r.labels,labels)
    assert np.array_equal(r.provenance,provenance)

def unknown_nifti_units():
    import nibabel as nib
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'unknown_units.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((3,2,1),np.float32),np.eye(4)),path)
        try:correction_cli._load_volume(path)
        except ValueError:return
        raise AssertionError('unknown NIfTI spatial units treated as mm')

def missing_correction_geometry():
    try:VolumeCorrector().correct(np.zeros((1,2,3),np.float32),[])
    except ValueError:return
    raise AssertionError('direct correction silently invents physical geometry')

def per_sample_cache_partition():
    x=torch.zeros(1,1,2,4,4);v=torch.ones(1,2,4,4,dtype=torch.bool)
    p=torch.zeros(1,4,2,4,4);p[:,1]=1;q=torch.zeros_like(p);q[:,3]=1
    batched=RegionManager({'mode':'ema','refresh_every':5})
    split=RegionManager({'mode':'ema','refresh_every':5})
    batched.targets(torch.cat([x,x+1]),v.expand(2,-1,-1,-1),
                    metadata('A')+metadata('B'),probabilities=p.expand(2,-1,-1,-1,-1),step=0)
    for image,patient in [(x,'A'),(x+1,'B')]:
        split.targets(image,v,metadata(patient),probabilities=p,step=0)
    rb=batched.targets(torch.cat([x,x+2]),v.expand(2,-1,-1,-1),
                       metadata('A')+metadata('C'),probabilities=q.expand(2,-1,-1,-1,-1),step=1)
    ra=split.targets(x,v,metadata('A'),probabilities=q,step=1)
    rc=split.targets(x+2,v,metadata('C'),probabilities=q,step=1)
    assert torch.all(rb.target_probabilities[0].argmax(0)==1)
    assert torch.all(rb.target_probabilities[1].argmax(0)==3)
    assert torch.equal(rb.target_probabilities,torch.cat([ra.target_probabilities,rc.target_probabilities]))

def batched_cache_resume_capacity():
    x=torch.zeros(1,1,2,4,4);v=torch.ones(2,2,4,4,dtype=torch.bool)
    p=torch.zeros(2,4,2,4,4);p[:,1]=1;q=torch.zeros_like(p);q[:,3]=1
    a=RegionManager({'mode':'ema','refresh_every':5,'max_cache_size':2})
    a.targets(torch.cat([x,x+1]),v,metadata('A')+metadata('B'),probabilities=p,step=0)
    b=RegionManager();b.load_state_dict(copy.deepcopy(a.state_dict()))
    ra=a.targets(torch.cat([x,x+2]),v,metadata('A')+metadata('C'),probabilities=q,step=1)
    rb=b.targets(torch.cat([x,x+2]),v,metadata('A')+metadata('C'),probabilities=q,step=1)
    assert torch.equal(ra.target_probabilities,rb.target_probabilities)
    assert torch.all(rb.target_probabilities[0].argmax(0)==1)

def authoritative_correction_mask_binding():
    from self_audit.volume.correction import authorize_simulation
    with tempfile.TemporaryDirectory() as td:
        m,ms,g,gs=make_volume_fixture(td,n_labeled=1,n_unlabeled=1,shape_tzyx=(2,3,5,7))
        manifest=json.loads(m.read_text());manifest['synthetic']=False
        m.write_text(json.dumps(manifest))
        rec=manifest['labeled'][0];item=rec['masks']['0']
        base=dict(dataset_role='TRAIN-L',label_budget=1,patient_id=rec['id'],
                  train_manifest=m,train_manifest_sha256=sha256_file(m),frame_index=0,
                  mask_path=Path(td)/item['path'],image_path=Path(td)/rec['image'])
        assert authorize_simulation(**base)['authorized']
        foreign=Path(td)/'TEST'/Path(item['path']).name
        foreign.parent.mkdir();np.save(foreign,np.zeros((3,5,7),np.uint8))
        for change in [{'frame_index':99,'mask_path':foreign},
                       {'mask_path':foreign},{'label_budget':777}]:
            try:authorize_simulation(**(base|change))
            except (ValueError,FileNotFoundError):continue
            raise AssertionError('accepted unauthorized binding '+str(change))
        generic=Path(td)/'split.json'
        generic.write_text(json.dumps({'train':[rec['id']],'val':[],'test':[]}))
        try:authorize_simulation(**(base|{'train_manifest':generic,'train_manifest_sha256':sha256_file(generic)}))
        except ValueError:return
        raise AssertionError('generic TRAIN split treated as labeled-budget proof')

def correction_invalid_support_invariance():
    shape=(1,5,9);a=np.zeros(shape,np.float32);a[:,:4,4:]=1
    b=a.copy();b[:,4,:]=1000
    valid=np.ones(shape,bool);valid[:,4,:]=False;prior=np.zeros(shape,np.uint8)
    for mode in ['geodesic_3d','geodesic_2d','image_guided','manual_only']:
        c=VolumeCorrector(propagation_mode=mode)
        kw=dict(prior_labels=prior,affine=np.eye(4),geometric_valid=valid)
        try:
            x=c.correct(a,[PointPrompt((1,1,0),target_class=1)],**kw)
            y=c.correct(b,[PointPrompt((1,1,0),target_class=1)],**kw)
        except ValueError as e:
            assert mode!='manual_only' and any(word in str(e).lower() for word in ['support','invalid','geometry']),str(e)
            continue
        assert np.array_equal(x.labels[valid],y.labels[valid]),mode
        assert np.allclose(x.probabilities[:,valid],y.probabilities[:,valid],atol=1e-7),mode

def correction_disconnected_support():
    image=np.zeros((1,3,7),np.float32);valid=np.ones(image.shape,bool);valid[:,:,3]=False
    prior=np.zeros(image.shape,np.uint8)
    for mode in ['geodesic_3d','geodesic_2d','image_guided','manual_only']:
        c=VolumeCorrector(propagation_mode=mode)
        kw=dict(prior_labels=prior,affine=np.eye(4),geometric_valid=valid)
        baseline=c.correct(image,[],**kw)
        try:out=c.correct(image,[PointPrompt((1,1,0),target_class=1)],**kw)
        except ValueError as e:
            assert mode!='manual_only' and any(word in str(e).lower() for word in ['support','invalid','geometry']),str(e)
            continue
        assert np.all(out.labels[:,:,4:]==0),mode
        assert np.all(out.provenance[:,:,4:]==1),mode
        assert np.allclose(out.probabilities[:,:,:,4:],baseline.probabilities[:,:,:,4:],atol=1e-7),mode
        assert out.labels[0,1,1]==1 and out.provenance[0,1,1]==2
        try:c.correct(image,[PointPrompt((3,1,0),target_class=1)],**kw)
        except ValueError:continue
        raise AssertionError('invalid-support prompt accepted by '+mode)

def correction_frame_index_bounds():
    import contextlib,io
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);p=root/'cine.npz';q=root/'frame.npz'
        cine=np.stack([np.full((1,2,2),.2,np.float32),np.full((1,2,2),.8,np.float32)])
        np.savez(p,image=cine,affine=np.eye(4));np.savez(q,image=cine[0],affine=np.eye(4))
        real_load=correction_cli._load_volume;real_hash=correction_cli.sha256_file
        touched=[]
        def guard_load(path,*args,**kwargs):
            assert Path(path) in (p,q),'forbidden volume read '+str(path)
            touched.append(('load',str(path)))
            return real_load(path,*args,**kwargs)
        def guard_hash(path):
            assert Path(path) in (p,q),'forbidden hash read '+str(path)
            touched.append(('hash',str(path)))
            return real_hash(path)
        def forbidden(*args,**kwargs):raise AssertionError('prompt/GT authorization reached before frame rejection')
        with patch.object(correction_cli,'_load_volume',guard_load),patch.object(correction_cli,'sha256_file',guard_hash),patch.object(correction_cli,'load_prompts',forbidden),patch.object(correction_cli,'authorize_simulation',forbidden):
            for path,index in [(p,-1),(p,99),(q,1)]:
                touched.clear()
                argv=['--image',str(path),'--time-index',str(index),'--mode','manual_only',
                      '--prompts',str(root/'forbidden_prompts.json'),'--simulate-from-gt',str(root/'forbidden_GT.npy'),
                      '--dataset-role','FIXTURE','--synthetic-fixture']
                try:
                    with contextlib.redirect_stdout(io.StringIO()):correction_cli.main(argv)
                except ValueError as e:
                    assert any(t in str(e) for t in ['time-index','3D','range']),str(e)
                    if index<0:assert touched==[]
                else:raise AssertionError('unsupported frame accepted '+str((path,index)))
        seen=[];real_correct=correction_cli.VolumeCorrector.correct
        def observe(self,image,*args,**kwargs):
            seen.append(image.copy());return real_correct(self,image,*args,**kwargs)
        with patch.object(correction_cli.VolumeCorrector,'correct',observe):
            for path,index,expected in [(p,0,cine[0]),(p,1,cine[1]),(q,0,cine[0])]:
                with contextlib.redirect_stdout(io.StringIO()):
                    assert correction_cli.main(['--image',str(path),'--time-index',str(index),'--mode','manual_only'])==0
                assert np.array_equal(seen[-1],expected)

checks=[per_sample_cache_partition,batched_cache_resume_capacity,cache_lru_resume,ema_requires_teacher,constant_content_motion,authoritative_correction_mask_binding,correction_invalid_support_invariance,correction_disconnected_support,correction_frame_index_bounds,
        background_manual_persistence,no_prior_is_unknown,untouched_provenance,
        unknown_nifti_units,missing_correction_geometry]
results=[]
for check in checks:
    try:check();results.append({'check':check.__name__,'status':'PASS'})
    except Exception as e:results.append({'check':check.__name__,'status':'FAIL','exception':type(e).__name__,'detail':str(e)})
print(json.dumps(results,indent=2))
```


## Region/temporal second repair and numerical recheck

At region SHA `33a854766f0566a384efda66f53d763de31ae1b955bcf06cf2cc5a531de29d9b` and temporal SHA `06ae5c877714177d045a59d636ee28d6ad36df38016f15c592a3d453d75a7740`, **34 owned tests passed in 1.88 s**. Independent R2 cross-class interior pooling, immutable target storage, cross-patient isolation, missing EMA source, LRU capacity resume, constant-content rejection, invalid-texture registration, out-of-bound displacement and through-Z probes PASS. Per-sample batch-companion/accumulation partition check still FAILS; coordinator assigned another repair. Scoped test hashes: regions `ba656ec8593acc09667ca3eceb3d988ec59a6c3571eaa10a01fda03786338f01`; temporal `07d2dc2cd954afcf608ebd61b2e116c5116d4d7ba741db8b730c3106b7275f09`.

Fresh independent fixed-target central finite differences with double student logits and nonzero losses: temporal loss0.29003019059245977, gradient norm0.007794384264011341, directional derivative -6.748866500479108e-5 vs FD -6.74886646567785e-5, error3.48013e-13; actual recovered XYZ displacement [2,-3.69e-9,-4.26e-9], warp max error0. Region loss0.22096222219351444, gradient norm0.03599753003492413, directional derivative -0.0005108585215182621 vs FD -0.0005108585221513806, error6.33119e-13. Region invalid-support gradient is exactly zero; teacher, image and feature gradients remain absent.

**T3 — P2 physical diagnostic:** temporal.py:605-612 uses metadata[0] spacing for every batch member. Duplicating the dx2 pair into B2 with XYZ spacings [1,1,4] and [3,1,4] produces mean_displacement_mm2 instead of4. The diagnostic needs per-sample spacing and explicit validation; routed to coordinator. This does not invalidate the tested voxel-space warp, but prevents accepting its physical metric.

## Native evaluation extension requested by coordinator

Reviewed all `self_audit/volume/evaluation.py`, `scripts/volume_evaluate.py`, and `tests/test_volume_evaluation.py`. At evaluation SHA `cfaf7a4f06ff220e67d102e92bf57292f5adac4b21ea2a66dab3b3e8d015b45c`, CLI SHA `fbb52a521319076fb62dccedfb2a492f12c51ce4c7903ca590c103a249903de9`, test SHA `c77985923e34568ffee949d30641fe6b87791479b30f8b148279a2212df973f2`, **13 tests passed in 1.67 s**. Configured spacing/FOV is operational and scoring argmax occurs after native probability inversion. Two independent blockers remain despite those passing tests.

**I2 — P1 configured support contamination:** evaluation.py:243-256 uses linear image interpolation but nearest native-valid interpolation, then :325-326 returns merged support without reintersecting declared native validity. Exact case: image TZYX(1,1,3,7), zero except invalid native x1 changed from0 to1, native_valid x1=false, identity affine, configured spacingXYZ(1.5,1,1), slab_depth1, tile_size10. A pointwise model with logits [10*center*valid,0,0,0] returns support true at every x, including invalid x1, while accepted native x2 probability0 changes .25 to .7367910743 solely from invalid input. Use full-stencil forward support and intersect native support on return.

**I3 — P1 image-only physical binding:** evaluation.py:423-430 does not pass prepared_image_tzyx to validate_nifti_correspondence; only authorized scoring later checks it at :642-650. Real-mode manufactured fixture TZYX(2,3,5,7), prepared range[.00401133,.9991073], all-zero raw NIfTI XYZT(7,5,3,2) with correct affine/mm units/RAS/hash, real manifest/sidecar: image-only inference accepts and emits FROZEN_IMAGE_ONLY, labels_read0. The original synthetic fixture orientation correctly rejects before this point; use the actual RAS orientation for the bypass. Bind prepared image voxels before all inference, leaving masks=None. Both blockers were routed for owner fixes.


<!-- NATIVE_EVAL_REPAIR_HARNESS_BEGIN -->
```python
import json, tempfile
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from self_audit.volume.fixtures import make_volume_fixture
from self_audit.volume.geometry import sha256_file
from self_audit.volume.evaluation import predict_native_volume, evaluate_manifest
from self_audit.volume.temporal import temporal_consistency
torch.set_num_threads(1)

def configured_support_stencil():
    class PixelModel(torch.nn.Module):
        def forward(self,x,valid_mask=None):
            q=x[:,1:2]
            if valid_mask is not None:q=q*valid_mask[:,None]
            return torch.cat([10*q,torch.zeros_like(q),torch.zeros_like(q),torch.zeros_like(q)],1)
    a=np.zeros((1,1,3,7),np.float32);b=a.copy();b[:,:,:,1]=1
    v=np.ones(a.shape,bool);v[:,:,:,1]=False
    kw=dict(slab_depth=1,tile_size=10,native_valid=v,spacing_xyz_mm=(1.5,1,1),resampling='configured')
    p,s=predict_native_volume(PixelModel(),a,np.eye(4),**kw)
    q,t=predict_native_volume(PixelModel(),b,np.eye(4),**kw)
    assert np.all(s<=v) and np.all(t<=v)
    assert np.array_equal(s,t) and s[0,0,1,6]
    assert np.max(np.abs(p-q)[np.broadcast_to(s[:,None],p.shape)])<1e-7

def real_image_only_correspondence():
    class ZeroModel(torch.nn.Module):
        def forward(self,x,valid_mask=None):return torch.zeros(x.shape[0],4,*x.shape[2:])
    with tempfile.TemporaryDirectory() as td:
        p,ms,g,gs=make_volume_fixture(td,n_labeled=1,n_unlabeled=1,shape_tzyx=(2,3,5,7))
        rec=json.loads(p.read_text())['labeled'][0];pid=rec['id']
        geo=json.loads(g.read_text());entry=geo['patients'][pid]
        affine=np.asarray(entry['affine_xyz_to_world_mm'])
        entry['orientation']=''.join(nib.aff2axcodes(affine))
        raw=Path(td)/'raw.nii.gz'
        obj=nib.Nifti1Image(np.zeros((7,5,3,2),np.float32),affine)
        obj.header.set_xyzt_units('mm','sec');nib.save(obj,raw)
        payload={'schema':'self_audit.volume.evaluation.v1','role':'evaluation',
                 'split':'development','synthetic':False,'records':[rec]}
        m=Path(td)/'eval.json';m.write_text(json.dumps(payload))
        geo['synthetic']=False;geo['source_manifest_sha256']=sha256_file(m)
        entry['source_image']={'path':raw.name,'sha256':sha256_file(raw)}
        g.write_text(json.dumps(geo))
        try:evaluate_manifest(ZeroModel(),m,sha256_file(m),g,sha256_file(g),
                              synthetic=False,allow_labels=False,slab_depth=3,tile_size=10)
        except ValueError as error:
            assert 'voxel correspondence' in str(error),str(error)
            return
        raise AssertionError('image-only real inference accepts unrelated prepared/raw voxels')

def physical_displacement_per_sample():
    torch.manual_seed(103)
    x=torch.zeros(1,1,5,17,19);x[:,:,1:4,4:12,4:14]=torch.rand(1,1,3,8,10)
    y=torch.zeros_like(x);y[:,:,1:4,4:12,6:16]=x[:,:,1:4,4:12,4:14]
    v=torch.ones(2,5,17,19,dtype=torch.bool)
    logits=torch.randn(2,4,5,17,19);teacher=torch.softmax(torch.randn_like(logits),1)
    _,m=temporal_consistency(logits,teacher,x.expand(2,-1,-1,-1,-1),
        y.expand(2,-1,-1,-1,-1),v,v,
        metadata=[{'spacing_xyz_mm':[1,1,4]},{'spacing_xyz_mm':[3,1,4]}],
        config={'mode':'motion','slice_wise':False,'weight':1})
    assert abs(m['mean_displacement_mm']-4)<1e-4,m

def configured_tiling_support():
    class Constant(torch.nn.Module):
        def forward(self,x,valid_mask=None):
            y=torch.zeros(x.shape[0],4,*x.shape[2:]);y[:,1]=4;return y
    image=np.zeros((1,1,3,9),np.float32)
    kw=dict(slab_depth=1,spacing_xyz_mm=(1.5,1,1),resampling='configured')
    a,s=predict_native_volume(Constant(),image,np.eye(4),tile_size=99,**kw)
    b,t=predict_native_volume(Constant(),image,np.eye(4),tile_size=2,**kw)
    assert np.array_equal(s,t),(s[0,0,1],t[0,0,1])
    assert np.allclose(a,b,atol=1e-7)

checks=[configured_support_stencil,configured_tiling_support,real_image_only_correspondence,physical_displacement_per_sample]
results=[]
for check in checks:
    try:check();results.append({'check':check.__name__,'status':'PASS'})
    except Exception as e:results.append({'check':check.__name__,'status':'FAIL','detail':str(e)})
print(json.dumps(results,indent=2))
```


**I4 — P1 configured tiling loses interior support:** with constant class1 model, zeros TZYX(1,1,3,9), identity affine, configured spacingXYZ(1.5,1,1), slab1 and overlap0, tile99 yields X support [T,T,T,T,T,T,T,T,F], while tile2 yields [T,T,F,T,T,F,T,T,F]. Six interior native voxels disappear solely because inverse interpolation is performed separately for each tile at evaluation.py:319. Merge current-grid probabilities/support before one native inverse, or equivalently guarantee full stencil coverage. The native final boundary is already unsupported for tile99; the finding concerns the additional interior holes. Routed to the active evaluation repair.

Independent macro aggregation positive control: manufactured patient A has two frames with RV Dice1 and1/3, patient B has one frame with RV Dice1. Actual patient values are2/3 and1; macro RV Dice0.8333333333333333 matches5/6 and differs from pooled Dice0.8461538461538461. No patient efficacy inference follows from this manufactured check.


## Bounded data-cache final scoped recheck

**ACCEPTED software scope:** data SHA `a5e7448c112e9cd70cd8b9e222837cfd0aa9b0a8d23414c51ecdc8f2d1a18cc8`; test SHA `00a0d5438e24c7b50ef229e2e6e84d814b26e4c574e67857169fdaa4d0a7cc41`; geometry/geometry-test SHA unchanged from accepted G0. **28 data+geometry tests passed in 1.35 s**. Independent capacity2 over six image arrays, retained evicted mmap-view safety, exact labeled/unlabeled sample tensors and metadata versus capacity8, sparse frames, padded Z, source-hash and synthetic-identity probes all PASS. The bound is on retained cache references, not on memory held by live caller views; evictions intentionally do not invalidate those views.

## Native evaluation final scoped recheck

**ACCEPTED software scope:** evaluation SHA `90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6`; CLI SHA `9a73e678ae9f8b8b0829ce7cb087d2c6f6121a4af6713f92fbc29f5fc99e5acf`; test SHA `1bb1eb0295b4a0044468c2748322a37664601669668470450c13c1f5beeaadd8`. **25 evaluation+geometry tests passed in 1.68 s**. Independent I2/I3/I4 repros now PASS. Source inspection confirms native invalid-input sanitization, conservative full-stencil current support, current-grid probability merge before one inverse, final native-valid intersection, unsupported probability zeroing, and prepared/raw image binding before inference with masks=None. Owned tests also execute a spatial convolution that is demonstrably sensitive to the invalid neighbor, and a manufactured real-mode positive correspondence control with nonexistent label paths.

Additional independent nonconstant probability phantom: TZYX(2,5,7,9), field .03z+.02y+.04x+.05t, affine [[0,-.9,0,30],[1.3,0,0,-12],[0,0,-4.2,10],[0,0,0,1]], spacingXYZ(2.6,1.8,4.2), FOVXYZ(5.2,3.6,8.4), flip=True, tile/slab overlap1. Model probabilities are [.2+.5q,.6-.4q,.1-.05q,.1-.05q]. Native output retains630/630 voxels, max probability error8.94070e-8 and max sum error1.19209e-7. This checks invertible linear-field geometry, not lossless reconstruction of arbitrary images or any patient accuracy.


## Per-sample region/temporal repair recheck

At region SHA `1474c539eb8ca64e92b6eaa367107a45206b1eefad55394034e14c54f200bb9c`, temporal SHA `854bf74222402446d9d984dfe1eccabec44dd152a7ef52cdd43932e48f26881d`, region-test SHA `42e78808f96858a15d0ea794ac3f171accd26e1fe5b878b6e598e013583d9f59` and temporal-test SHA `32dc8df4c4939115ae3036d27b2c889b8cbd423a5181e233b2495d2c32e15afc`, **36 owned tests passed in2.04s**. Independent partition invariance, isolated LRU resume, missing teacher, boundary conflict, constant-content rejection, physical mixed-spacing4mm, and original motion probes PASS. Temporal scope is accepted at this hash; valid provided metadata is checked per sample.

**R3 — P1 batched cache/resume interaction still FAILS:** RegionManager(mode=ema,refresh_every=5,max_cache_size=2), step0 batch[A=zeros,B=ones], teacherclass1; deepcopy state into default manager; step1 batch[A=zeros,C=twos], teacherclass3. Uninterrupted A remainsclass1 but resumed A becomesclass3, max target difference1.0. regions.py:777-789 reinserts the combined last-return target into the per-sample LRU, evicting a legitimate sample. Preserve the combined return object separately; do not let it consume sample-cache capacity on restore. Exact executable witness added to the auxiliary harness and routed to coordinator.


## Correction second repair independent recheck

At correction SHA `c3b8d26f5fa7b78754d55e0ec4d43e063d89faaace9bcfb3a278bfc5bd23ab97`, CLI SHA `1e3cd85f78346b74f363848ae80761692e9da19e5a230c6d886b84600d86688c`, test SHA `b8166ee7880a623187b6f11918fef9aa538b27c9424cf116da29d06afe8d54cf`, **42 owned tests passed in6.68s**. Independent forbidden-role pre-IO, role allowlist, unknown255 retention, manual background persistence across another edit, no-prior unknown, untouched all-category provenance, unknown NIfTI units and missing direct correction geometry all PASS. Native exporter now rejects an absent physical affine unless explicitly synthetic; prior/mask affine checks and independent export-path tests remain passing.

**C5 — P1 authoritative TRAIN-L authorization bypass remains:** correction.py:931-974 recognizes an arbitrary labeled list or generic train split, ignores the manifest budget/schema/role, skips mask binding when a frame is absent, and compares only path basenames. Exact manufactured v2 real manifest: label_budget1, labeled patientA with frame0 pathL/gt.npy, distinct U and TEST identities. authorize_simulation accepts frame99 + TEST/forbidden.npy; frame0 + TEST/gt.npy; budget777; and a generic {train:[A],val:[],test:[]} split. Instrumented CLI with claimed TRAIN-L/A/frame0 opens TEST/gt.npy and exits0. No real patient mask was accessed; _load_volume was an IO spy returning manufactured arrays.

Require a hash-pinned v2 training manifest with explicit real identity, unique/disjoint L/U/evaluation IDs, exact dense-budget count, a required declared frame, and resolved full mask-path plus declared hash binding before any mask access. Generic TRAIN split lacks L-budget evidence and must reject. Real API use must also retain this binding when consuming its GT array. An executable negative/positive authorization witness is in the auxiliary harness. Routed to coordinator before final acceptance; positive full-suite receipts cannot override this boundary failure.


## R3 final scoped acceptance

Region SHA `a192976e0a08247589a6b2e2ee688782cbf28b76cb53c1774ddd0fd6caa3eca6`, region-test SHA `379080cd766d2e84ba280bab9aa9cfa76245fef976ca8c18b04c9886b6fbe620`: independent exact combined capacity2 batched-resume, partition invariance, isolated LRU resume and missing-EMA probes PASS. Source confirms the combined last-return object is restored separately without consuming per-sample cache capacity. Earlier numerical gradients and boundary/motion tests remain applicable to unchanged loss code. Owner separately reported44 region/resume/engine tests; that owner receipt is reported evidence, not a reviewer rerun. Root is running the final repository gate.

**C6 — P1 invalid geometry influences valid correction output:** shape ZYX(1,5,9), a=zeros with a[:,:4,4:]=1, b=a except b[:,4,:]=1000, geometric_valid[:,4,:]=false, prior_labels=zeros, affineI, promptXYZ(1,1,0) class1. Changing only invalid intensities gives geodesic3d/2d valid probability delta0.3713473678 and1 changed valid label; image_guided delta0.7810035348 and16 changed valid labels. Manual-only is invariant. Invalid outputs themselves remain255; marking them unknown at the end does not prevent their earlier influence. correction.py:521-524 includes invalid values in normalization, and propagation has no support input. Exclude invalid support from statistics/propagation and reject invalid prompts, or fail closed on unsupported partial-support propagation. Exact witness appended; all evidence is manufactured. Routed to coordinator for the same correction follow-up.

## OWNER_REPORTED appendix: correction provenance and support repair

This appendix was appended by the correction implementation owner, not by this independent reviewer. Its claims are retained as **OWNER_REPORTED** evidence; the reviewer verdict is recorded separately below.

Correction source SHA `fdb27bad361d67786652f2d8515ba132639fb606738d99b4a9a0cc1134947f08`, CLI SHA `4660922ea1a738f6eebe5928c0d21e064a6048c04465966f43b3b7ddefea1b01`, and owned test SHA `e4c801ccf5edf6f8e226c85cfe34b74d696491ac1446d7348de1ea39d4d3ad69`. `rtk pytest -q tests/test_volume_correction.py tests/test_volume_cli.py`: **73 passed**. The 49 correction tests include exact real-mode negative I/O witnesses for role/schema/synthetic/duplicate/budget/frame/patient/path/hash conditions, same-basename foreign masks, CLI pre-read rejection, and a hash-pinned real NIfTI positive; no real patient mask was opened.

The C5 authorization boundary is repaired for the public `authorize_simulation` and `simulate_clicks_from_ground_truth` APIs and CLI: real calls require hash-pinned `self_audit.data.v2`, role `training`, `synthetic=false`, explicit unique/disjoint labeled/unlabeled/evaluation IDs, exact positive `label_budget == manifest label_budget == len(labeled)`, exact labeled patient membership, explicit frame, exact resolved full image/mask paths relative to the manifest, well-formed declared SHA-256 values, and matching image/mask bytes before GT loading. Generic train/val/test split manifests and foreign frame/path/schema/role/synthetic/budget/patient/duplicate cases fail before foreign mask stat/hash/open; synthetic fixtures remain a separate opt-in path. The CLI now binds `--image` to the declared labeled patient source hash before any GT read.

The C6 support boundary is also repaired in all three non-manual propagation modes: percentile normalization uses valid support only; geodesic and image-guided affinities cannot seed, traverse, or refine invalid voxels; image-guided affinity is connected-component gated; cross-slice geodesic paths require continuous valid support; invalid prompts reject; all-invalid no-op outputs remain UNKNOWN. The manufactured invalid-intensity invariance, disconnected-component barrier, invalid-prompt, and empty-support checks pass for geodesic-3D, geodesic-2D, and image-guided modes. These are local CPU/software and manufactured-format receipts only; patient, GPU, clinical accuracy and long training remain NOT RUN.


## Reviewer C5/C6 independent closure

The reviewer executed **73 correction+CLI tests in10.22s** and independently reran background manual persistence, no-prior unknown, all-category provenance, units/missing geometry, exact foreign-path/undeclared-frame/budget/generic-schema authorization cases, invalid-intensity invariance, disconnected support and invalid prompts. These probes PASS after supplying the new required image_path to the legitimate authorization positive control. This API tightening is expected fail-closed behavior, not a regression. Final normalized correction source is `fdb27bad361d67786652f2d8515ba132639fb606738d99b4a9a0cc1134947f08`; CLI `4660922ea1a738f6eebe5928c0d21e064a6048c04465966f43b3b7ddefea1b01`; test `e4c801ccf5edf6f8e226c85cfe34b74d696491ac1446d7348de1ea39d4d3ad69`.

**C7 — P2 unsupported temporal frame silently substituted:** scripts/volume_correct.py:170 clamps time_index to [0,T-1], and 3D inputs ignore nonzero values. Manufactured NPZ image(2,1,2,2)+affineI with --time-index -1 or99 --mode manual_only both exit0. Require out-of-range 4D indices and nonzero 3D indices to reject before GT/prompt reads; retain valid-index behavior. Exact witness added and routed as a narrow final frame-selection correction.


## Final independent acceptance — 2026-10-26

**No open blocker remains in the assigned geometry/data/fixture/preparation/region/temporal/correction/native-evaluation scope.** G1–G4, R1–R3, T1–T3, C1–C7 and I1–I4 are closed for supported software contracts by owner repairs plus the recorded independent negative and positive controls. This is not approval of real-patient accuracy, clinical use, pretrained execution, target-GPU performance or scientific novelty.

Final reviewer-executed checks after C7:

- All three embedded harnesses: **31 PASS / 0 FAIL** (13 original adversarial, 14 auxiliary repair and four native evaluation/physical metric cases). These cover source/geometry binding, sparse cine masks, padded support, cache/sample identity, detached targets, boundary conflict, resume/capacity, actual nonidentity motion, forbidden mask I/O, manual provenance, correction support and native inverse/tiling.
- Strengthened `correction_frame_index_bounds` with sentinels for GT hash/load, prompt load and authoritative mask authorization. Four-dimensional indices -1/99 and 3-D index1 reject before those operations; negative indexing rejects before any image I/O. Actual CLI positive calls for 4-D frame0/frame1 and 3-D frame0 pass exactly the selected input values into the corrector.
- `rtk proxy python -m pytest -q tests/test_volume_correction.py -k 'invalid_time_index or valid_time_index'`: **4 passed, 49 deselected in1.30s**.
- `rtk proxy python scripts/check_repo.py`: **PASS**, native decoder v2, semi-supervised dense TRAIN subset including audit, explicitly not no-GT.
- Earlier independently executed nonzero finite differences, asymmetric physical/inverse phantoms, manufactured real-format correspondence, bounded-cache/live-view checks, native export geometry and patient-macro aggregation controls remain documented above. None is patient or GPU evidence.

C7 implementation is `scripts/volume_correct.py:162-187,244-284`: strict shape-dependent frame selection, pre-I/O negative rejection and authoritative mask binding after frame validation. Regression coverage is `tests/test_volume_correction.py:446-536`. Source and independent sentinels agree.

### Coordinator-executed gates independently inspected

These integration checks were executed by the coordinator; the reviewer inspected the durable receipts and recomputed their current source inventory rather than rerunning the entire matrix:

- `POST_CLI_GATE.json`: **PASS**, check_repo exit0 and full pytest exit0, **422 passed, one warning in29.27s**. The warning is the existing `torch.jit.script` deprecation in `tests/test_pipeline.py::test_decoder_jvp_transaction`. All **67** code/config/test hashes match independently recomputed current bytes. Core source digest remains `3668c8cdd9523d2440a7e93ab2ed0564ba08ddd6bf8998ef88210ff6c96632ac`.
- `verified/FINAL_VALIDATION.json`: core gate **418 passed, one warning**, **59 commands**, 22 executable arm preflights/smokes plus historical-only V00, and six CPU profiles; core digest identical before/after. Subsequent C7 changes affect correction CLI/tests and are covered by the 422-test gate.
- `verified/ROUNDTRIP.json`, `COMMANDS.json` and stdout: manufactured actual CLI full4 versus stop2/resume4 has no payload differences; intermediate save cadence was checked, image-only inference label reads0, and exported student logits match bitwise without teacher/external-teacher/pretrained-weight dependencies. This is bounded four-step synthetic software evidence.
- Initial verification attempts, historical A0, reports and figures remain preserved. Historical mixed-worktree failures and owner-reported results above are not silently upgraded to final verification.

### Final source and test byte hashes

Recomputed after the post-CLI gate. Listed implementation/test hashes match its inventory; documentation hashes separately pin the reviewed specifications.

| File | SHA-256 |
|---|---|
| `docs/native_3d/ASTRA_IMPLEMENTATION_PROMPT.md` | `8a72241d4e25b7c227461e7c7153f52923b7ef45bb1eecad74e3feb249a94244` |
| `docs/native_3d/IMPLEMENTATION_CONTRACT.md` | `43f5d26aa3b4876d6dcc3a1a670422569e4604166af6545b70c23b0627cc996a` |
| `ACTIVE_PIPELINE.json` | `4a4645611c31f56d3bb2b33da5f3abbb7d54cca2987dbe7d9e339742adc4ae5e` |
| `docs/METHOD.md` | `4f40d50647eeb7c0cd3ae5fe73655cde8e0ce1650f4f3437f3707edfab874ae7` |
| `self_audit/volume/geometry.py` | `3fdb0986e4ff572c6855743414fa1f443e631d57450fe3b7803e5e52f3e1c9bd` |
| `self_audit/volume/data.py` | `a5e7448c112e9cd70cd8b9e222837cfd0aa9b0a8d23414c51ecdc8f2d1a18cc8` |
| `self_audit/volume/fixtures.py` | `f1b88049b6748f6ce50247a575500abad885b52a072ae2f759a9b0ad88810a90` |
| `scripts/volume_prepare.py` | `32929bb2e1fec70dbd6ff9ca2381c6312b2e0a3433a700e9854e486eeefed881` |
| `self_audit/volume/regions.py` | `a192976e0a08247589a6b2e2ee688782cbf28b76cb53c1774ddd0fd6caa3eca6` |
| `self_audit/volume/temporal.py` | `854bf74222402446d9d984dfe1eccabec44dd152a7ef52cdd43932e48f26881d` |
| `self_audit/volume/correction.py` | `fdb27bad361d67786652f2d8515ba132639fb606738d99b4a9a0cc1134947f08` |
| `scripts/volume_correct.py` | `85439b39b33c6f0ac8d814ec8b1e11faee6223660434325a0d1f3303fb86e69f` |
| `self_audit/volume/evaluation.py` | `90e5b7207af274a841a5944a33775e57e9d656af5730f8a9289e6c4455ae82c6` |
| `scripts/volume_evaluate.py` | `9a73e678ae9f8b8b0829ce7cb087d2c6f6121a4af6713f92fbc29f5fc99e5acf` |
| `scripts/volume_export.py` | `0ed301b677864d50ff949b4cc4a180b8929d1dd6145bd93aab91eb763c84a247` |
| `tests/test_volume_geometry.py` | `08c3dda96c2d8abb0896b64774476afc8d737dbae82d7ae408f73e6e95bfcc3f` |
| `tests/test_volume_data.py` | `00a0d5438e24c7b50ef229e2e6e84d814b26e4c574e67857169fdaa4d0a7cc41` |
| `tests/test_volume_regions.py` | `379080cd766d2e84ba280bab9aa9cfa76245fef976ca8c18b04c9886b6fbe620` |
| `tests/test_volume_temporal.py` | `3af741402f665f522671949070ae2c5f842abddf4eef9e3ca8ea03835abf5c23` |
| `tests/test_volume_correction.py` | `0fbd19770dffc99393241f60ac4e68989619447b809917009ef70a7447e77108` |
| `tests/test_volume_evaluation.py` | `780f7063bdb101306fb7042538280ca17749e16364d8543380a5ea89bf8ca6ef` |
| `tests/test_volume_cli.py` | `724adf4717300bca05299abe10e1239595e70c952cc421e456aab117be8fad8b` |

### Durable receipt and weight provenance

| Receipt | SHA-256 |
|---|---|
| `reports/native_volume_pilot/POST_CLI_GATE.json` | `730821e3a2aa6129c8a0dcadaa9be1bf605413f9276bfde1dc64524b5c154c7e` |
| `reports/native_volume_pilot/verified/FINAL_VALIDATION.json` | `7a8dd5417ff3e032d9e2e668fb32a2e7d274f2a63386612b53c169a39bb8a81a` |
| `reports/native_volume_pilot/verified/ROUNDTRIP.json` | `b4f086710c0be8711dd22c77ea396418405f7ec84a4e8e75a30c52190a8ea9ed` |
| `reports/native_volume_pilot/verified/COMMANDS.json` | `d02f4d4135e26d35b631ba5e9e6a61598296a5852c7dee65eddbb05ef4100957` |

No pretrained weight was downloaded or loaded in this review. An inherited configuration weight SHA is a declaration, not verified execution. Coordinator-generated synthetic checkpoint **manifest** hashes are `98e1991ba3fb4b05483bddb9a78c83c32f2fca16265cab34847b1e80b2382060` (full4), `c580100a56895c5e8e0c68b11a7a0393e758f7d013f2c74d8aa61000084b6b9b` (resumed4) and `ef4d96061080cf767cdc502a291f6ee057aeb0b45e1ce14dc7a268e0a299f8b0` (student export). Inspected stdout confirms these are manifest hashes, not raw weight-file hashes. Temporary checkpoint/export directories from `COMMANDS.json` no longer exist at final review, so payload bytes could not be independently rehashed; durable receipts retain the executed comparison, not deliverable model weights. Fixture manifest SHA is `6932474cbab796f47ee5e98088b588158a737ddac1e0c37064d8c2bdefcc6794`, geometry SHA `f7ef503cf11ead739f6596359f81592375d423cc44c4bb874442bac3c25cd205`.

### Supported limits and missing evidence

Tested unproven roles/budgets/paths/frames, missing/inconsistent geometry, unknown NIfTI units, absent EMA probabilities and unsupported temporal selection fail closed. Invalid interpolation stencils and out-of-model/unobservable motion abstain. Cache capacity bounds retained references, while live caller views may retain memory after eviction. Native inverse output uses conservative emitted support; continuous-field invariants do not establish lossless recovery of arbitrary resampled data. Temporal registration is a bounded translation baseline; reliable patient-specific nonrigid cardiac correspondence is unestablished. Correction modes are software controls, not proof of clinical refinement benefit.

**NOT RUN:** real DINO/pretrained-weight execution; real TRAIN-patient smoke; patient accuracy/Dice; real-patient geometry validation; PRO4000/CUDA memory, speed or throughput; human annotation time; long training; clinical utility; novelty or efficacy. The server is not rented. The reviewer performed no actual U/VAL/TEST mask read, source edit, data/checkpoint deletion, long training, external publication or non-Orca delegation. No owned software blocker remains; scientific and hardware experiments remain separate work and cannot be inferred from manufactured checks.
