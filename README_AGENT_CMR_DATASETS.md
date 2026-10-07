# README — Agent Task: Audit, Preprocess & Add Dataset Adapters for CMR-MULTI and CMRxMotion

## 1. Mục tiêu

Mục tiêu của task này **không phải lập tức nhét 2 dataset vào training**.

Agent phải thực hiện theo thứ tự:

1. **Audit codebase hiện tại** để hiểu chính xác project đang train/infer theo contract nào.
2. **Audit dữ liệu thực tế** của `CMR-MULTI` và `CMRxMotion`.
3. Đánh giá **hai dataset có thật sự phù hợp để training cho project hiện tại hay không**.
4. Chỉ khi phù hợp mới:
   - thiết kế preprocessing;
   - viết dataset adapter;
   - thêm config;
   - thêm test;
   - chứng minh adapter tương thích với pipeline hiện tại.
5. Không được thay đổi training logic cốt lõi nếu chưa chứng minh là cần thiết.

Project hiện tại sử dụng mô hình **3.5D**: input là một nhóm vài lát cắt spatial lân cận từ một volume 3D, còn target thường là mask của lát trung tâm.

---

# 2. Nguyên tắc bắt buộc

## 2.1. Không đoán cấu trúc project

Không được giả định:

- model dùng 3 slices hay 5 slices;
- label mapping hiện tại;
- resize target;
- spacing target;
- normalization;
- cách xử lý boundary slices;
- orientation;
- loss;
- dataset split;
- target là center slice hay multi-slice output.

Phải đọc code hiện tại để xác định.

## 2.2. Không sửa dữ liệu gốc

Dataset source phải được xem là **read-only**.

Không overwrite:

```text
CMR-MULTI/
CMRxMotion/
```

Nếu cần preprocess offline, output phải nằm ở thư mục mới, ví dụ:

```text
data_processed/
```

hoặc theo convention đang có trong repo.

## 2.3. Không split theo slice/frame

Train/val/test phải split theo **subject/case**, không theo slice, frame hoặc acquisition.

Đây là yêu cầu bắt buộc để tránh data leakage.

## 2.4. Không tự invent preprocessing

Preprocessing của dataset mới phải ưu tiên **match pipeline hiện tại**.

Ví dụ nếu project hiện tại đã dùng:

```text
resize -> percentile clipping -> z-score
```

thì adapter mới phải cố gắng tuân theo pipeline đó.

Nếu cần khác biệt, phải giải thích rõ lý do.

---

# 3. Phase 0 — Audit repository hiện tại

Trước khi động vào dataset, agent phải đọc toàn bộ phần code liên quan đến data pipeline.

## 3.1. Tạo inventory repo

Ghi lại:

```text
- entrypoint training
- entrypoint inference/evaluation
- Dataset / DataLoader classes
- preprocessing
- augmentation
- model input shape
- loss
- label schema
- metrics
- config files
- split logic
- checkpoint loading
```

Ưu tiên tìm các file/từ khóa:

```text
Dataset
DataLoader
__getitem__
train.py
trainer
dataset.py
data.py
loader
preprocess
transform
augment
spacing
resize
crop
normalize
nifti
nibabel
SimpleITK
MONAI
label
class_map
num_classes
Dice
HD95
```

Có thể dùng:

```bash
rg -n "Dataset|DataLoader|__getitem__|num_classes|Dice|HD95|spacing|resize|normalize|nibabel|SimpleITK|MONAI" .
```

## 3.2. Xác định input contract thật sự

Agent phải trả lời bằng code evidence:

```text
Model input shape:
    [B, C, H, W] ?
    [B, C, D, H, W] ?

3.5D context:
    C = ?
    ví dụ 3 / 5 / 7 slices

Context direction:
    spatial Z only?
    temporal?
    mixed?

Target:
    mask center slice?
    mask nhiều slice?
```

Ví dụ nếu model nhận:

```text
[B, 5, H, W]
```

thì phải xác nhận 5 channel đó có thật sự là:

```text
z-2, z-1, z, z+1, z+2
```

hay không.

## 3.3. Xác định label contract

Phải tìm đúng schema hiện tại.

Ví dụ có thể là:

```text
0 = BG
1 = RV
2 = MYO
3 = LV
```

nhưng **không được assume**.

Agent phải trích từ:

- code;
- config;
- metric implementation;
- dataset loader;
- loss;
- visualization.

Tạo biến chuẩn duy nhất:

```python
UNIFIED_LABEL_SCHEMA = ...
```

Dataset-specific mapping chỉ được nằm trong adapter.

## 3.4. Xác định preprocessing contract

Ghi rõ:

```text
orientation handling
resampling
XY spacing
Z spacing
resize
crop/pad
intensity clipping
normalization
augmentation
boundary handling
dtype
```

Ví dụ phải biết:

```text
edge slices:
replicate?
reflect?
zero pad?
drop?
```

Adapter mới phải tương thích.

---

# 4. Phase 1 — Audit CMR-MULTI

Dataset path phải lấy từ config/CLI, không hard-code.

Ví dụ:

```text
<CMR_MULTI_ROOT>/
└── CINE_MULTI/
    └── SAX_TR/
        ├── image/
        └── anno/
```

## 4.1. Check pairing

Phải kiểm tra:

```text
number of image files
number of anno files
missing image
missing anno
duplicate case IDs
```

Fail nếu pairing không 1:1 mà không có lý do rõ ràng.

## 4.2. Check từng case

Tạo audit CSV:

```text
reports/cmr_multi_audit.csv
```

Ít nhất gồm:

```text
case_id
image_path
mask_path
image_shape
mask_shape
image_dtype
mask_dtype
spacing_x
spacing_y
spacing_3rd_axis
affine_equal
unique_labels
image_min
image_max
image_mean
image_std
nonzero_mask_voxels
```

## 4.3. Xác minh cấu trúc flattened Z×T

Dữ liệu đã quan sát có dạng:

```text
(X, Y, N)
```

với nhiều case như:

```text
275
300
330
360
...
```

Có bằng chứng mạnh rằng:

```text
N = Z * T
```

và thứ tự flatten là dạng `ZT`:

```text
z0,t0
z0,t1
...
z0,t(T-1)
z1,t0
...
```

Tuy nhiên adapter **không được hard-code blindly**.

Phải infer/verify mỗi case.

### Candidate constraints

Có thể tìm candidate:

```text
T ~ 18..35
Z ~ 6..20
```

sau đó score dựa trên:

```text
temporal continuity
cyclic wrap continuity
spatial continuity
physical Z spacing plausibility
```

### Required sanity checks

Candidate tốt phải thỏa phần lớn:

```text
temporal Dice cao
wrap Dice hợp lý
temporal continuity > spatial continuity
reconstructed Z spacing hợp lý
```

Output:

```text
reports/cmr_multi_zt_inference.csv
```

gồm:

```text
case_id
N
Z
T
score
temporal_dice
wrap_dice
spatial_dice
derived_z_spacing
confidence
```

Flag manual review nếu:

```text
score thấp
candidate top-1 và top-2 quá gần nhau
wrap thấp bất thường
derived spacing bất thường
```

Không được âm thầm chọn candidate mơ hồ.

## 4.4. Reconstruct đúng semantic dimensions

Sau khi infer:

```text
[X, Y, N]
    ->
[X, Y, Z, T]
```

Sau đó đối với model 3.5D:

```text
for each t:
    volume_t = image[:, :, :, t]   # [X,Y,Z]
```

Context phải lấy theo **Z**, không lấy theo T.

Ví dụ:

```text
z-2, z-1, z, z+1, z+2
```

trong cùng một `t`.

Tuyệt đối không tạo context kiểu:

```text
t-1, t, t+1
```

nếu model hiện tại là spatial 3.5D.

## 4.5. Check label CMR-MULTI

Từ dữ liệu thực tế đã thấy:

```text
unique labels = [0,1,2,3]
```

Nhưng phải xác minh semantic mapping từ metadata/dataset docs hoặc existing verified knowledge.

Mapping dataset-specific phải được chuyển về schema chung của project.

Ví dụ nếu source là:

```text
0 BG
1 MYO
2 LV
3 RV
```

và project là:

```text
0 BG
1 RV
2 MYO
3 LV
```

thì adapter remap:

```text
0 -> 0
1 -> 2
2 -> 3
3 -> 1
```

Không modify file mask source.

---

# 5. Phase 2 — Audit CMRxMotion

Dataset path lấy từ config.

Không assume layout. Trước tiên scan directory tree.

## 5.1. Xác định naming convention

Agent phải xác định:

```text
subject ID
motion/acquisition ID
phase ED/ES
image
label
```

Ví dụ có thể giống:

```text
P001-1-ED.nii.gz
P001-1-ED-label.nii.gz
P001-1-ES.nii.gz
...
```

nhưng phải xác minh từ dataset thực tế.

## 5.2. Group theo subject

Nếu:

```text
P001-1
P001-2
P001-3
P001-4
```

là nhiều acquisition của cùng `P001`, toàn bộ phải nằm cùng một split.

Phải tạo parser:

```python
subject_id = ...
acquisition_id = ...
phase = ...
```

và unit test parser.

## 5.3. Check missing GT

Không assume mọi volume đều có label.

Audit CSV:

```text
reports/cmrxmotion_audit.csv
```

gồm:

```text
subject_id
acquisition_id
phase
image_path
mask_path
has_mask
image_shape
mask_shape
spacing
unique_labels
intensity stats
```

Supervised training chỉ dùng sample có GT hợp lệ.

Không được:

```text
missing mask -> all background mask
```

Đây là lỗi nghiêm trọng.

## 5.4. Check dimensionality

CMRxMotion được kỳ vọng là mỗi ED/ES là 3D SAX volume:

```text
[X,Y,Z]
```

nhưng phải xác nhận từ file thật.

Nếu volume là 3D:

```text
không reconstruct Z×T
```

Adapter chỉ cần:

```text
load volume
-> select spatial context around z
-> center-slice target
```

## 5.5. Check spacing và affine

Không hard-code spacing từ paper.

Đọc trực tiếp:

```python
nii.header.get_zooms()
nii.affine
```

Thống kê:

```text
min
median
max
```

cho từng axis.

Check:

```text
image/mask same shape
image/mask compatible affine
```

Nếu affine không đáng tin nhưng arrays aligned, document rõ.

## 5.6. Check label semantics

Phải xác minh source schema.

Nếu source là:

```text
0 BG
1 LV
2 MYO
3 RV
```

thì map về schema chung của project.

Không assume chỉ dựa trên numeric IDs.

---

# 6. Phase 3 — Quyết định dataset có phù hợp không

Không được kết luận "phù hợp" chỉ vì:

```text
đều là cardiac MRI
đều có LV/RV/MYO
```

Phải đánh giá theo matrix sau.

## 6.1. Compatibility matrix

Tạo:

```text
reports/dataset_compatibility.md
```

Cho từng dataset:

| Criterion | CMR-MULTI | CMRxMotion |
|---|---|---|
| SAX anatomy compatible | | |
| LV/RV/MYO compatible | | |
| GT quality sufficient | | |
| Input convertible to current 3.5D | | |
| Spatial context meaningful | | |
| XY resolution compatible | | |
| Z spacing acceptable | | |
| Label mapping reliable | | |
| Subject ID reliable | | |
| Leakage controllable | | |
| Dataset size useful | | |
| Domain shift beneficial | | |
| Severe artifact risk | | |
| Missing GT risk | | |
| Training value | | |

Không dùng score kiểu tùy ý. Kết luận phải bằng evidence.

## 6.2. Kết luận phải nằm trong một trong các trạng thái

Cho từng dataset:

```text
USE_FOR_TRAINING
USE_WITH_RESTRICTIONS
USE_FOR_VALIDATION_ONLY
EXCLUDE
```

Kèm lý do.

Ví dụ restriction có thể là:

```text
only labeled volumes
only SAX
exclude uncertain Z×T cases
exclude corrupted masks
```

---

# 7. Phase 4 — Thiết kế adapter architecture

Mục tiêu là dataset-specific logic **không leak vào training loop**.

Nên có common interface.

Ví dụ:

```python
class Cardiac35DAdapter:
    def list_cases(self):
        ...

    def load_case(self, case_id):
        ...

    def get_subject_id(self, case_id):
        ...

    def get_volume_units(self, case_id):
        ...

    def remap_labels(self, mask):
        ...

    def get_metadata(self, case_id):
        ...
```

Dataset-specific:

```text
CMRMultiAdapter
CMRxMotionAdapter
```

Training Dataset class chung chịu trách nhiệm:

```text
context slice extraction
normalization
resize/crop
augmentation
tensor conversion
```

Dataset adapter chịu trách nhiệm:

```text
file discovery
format interpretation
Z×T reconstruction
source label mapping
subject metadata
phase/acquisition metadata
```

---

# 8. Phase 5 — 3.5D sample generation

Không pre-export hàng chục nghìn PNG nếu không cần.

Ưu tiên lazy/on-the-fly loading nếu codebase hiện tại cho phép.

Một sample logical:

```python
{
    "image": Tensor[C, H, W],
    "mask": Tensor[H, W],
    "subject_id": str,
    "case_id": str,
    "dataset": str,
    "z": int,
    "phase": ...,
    "t": ...
}
```

## 8.1. Spatial context

Nếu project hiện tại dùng 5 slices:

```text
[z-2,z-1,z,z+1,z+2]
```

adapter mới phải giữ 5.

Không tự đổi thành 3.

## 8.2. Boundary

Phải match behavior hiện tại:

```text
replicate
reflect
zero pad
drop
```

Không invent rule mới.

## 8.3. Target

Nếu project train center-slice target:

```text
target = mask[z]
```

giữ nguyên.

---

# 9. Phase 6 — Preprocessing

Preprocessing mới phải ưu tiên **compatibility với model đang có**, không tối ưu riêng cho dataset mới.

## 9.1. Orientation

Không canonicalize blindly.

Phải kiểm tra:

```text
current datasets
new datasets
image-mask alignment
```

Nếu cần orientation normalization, phải áp dụng thống nhất và có visual QC.

## 9.2. Spatial resampling

Không resample Z mạnh nếu không cần.

Đặc biệt với SAX MRI:

```text
Z spacing thường lớn hơn XY
```

Nếu model 3.5D hiện tại dùng native adjacent slices, giữ chiến lược đó trừ khi repo đã resample Z.

## 9.3. XY resize/resample

Match pipeline hiện tại.

Ví dụ nếu current project resize:

```text
H,W -> 256,256
```

thì dataset mới cũng theo đúng rule.

## 9.4. Intensity normalization

MRI không có absolute intensity scale như CT.

Không dùng global raw intensity normalization xuyên dataset nếu current project không làm vậy.

Ưu tiên đúng pipeline hiện tại, ví dụ:

```text
per-volume percentile clipping
per-volume z-score
```

hoặc cách repo đang dùng.

---

# 10. Phase 7 — Split strategy

Split unit = **subject**.

## CMR-MULTI

Nếu mỗi file là một case/subject độc lập:

```text
split theo case ID
```

Nhưng phải verify metadata không có repeat scan.

## CMRxMotion

Tất cả:

```text
same subject
all motion levels
ED
ES
```

phải cùng split.

Tạo test:

```python
assert train_subjects.isdisjoint(val_subjects)
assert train_subjects.isdisjoint(test_subjects)
assert val_subjects.isdisjoint(test_subjects)
```

---

# 11. Phase 8 — Sampling strategy

Không để dataset lớn hơn thống trị chỉ vì nhiều slice/frame hơn.

Phải báo cáo:

```text
subjects
volumes
frames
center-slice samples
labeled samples
```

cho mỗi dataset.

Nếu mixed training:

```text
CMR-MULTI
+
CMRxMotion
+
existing datasets
```

cần xem xét:

```text
dataset-balanced sampling
subject-balanced sampling
```

Không mặc định concat rồi shuffle nếu sample count lệch lớn.

---

# 12. Phase 9 — Quality control

Agent phải tạo visual QC.

Ít nhất:

```text
reports/qc/
```

Gồm random samples từ cả hai dataset.

Mỗi QC image phải hiển thị:

```text
context slice(s)
center image
GT overlay
class colors
case ID
subject ID
z
t/phase
```

CMR-MULTI:

- ít nhất 10 cases;
- nhiều T khác nhau;
- apex/mid/base.

CMRxMotion:

- nhiều subjects;
- nhiều motion levels;
- ED và ES;
- có image quality xấu nếu dataset có.

Không hoàn thành adapter nếu chưa visual QC.

---

# 13. Phase 10 — Tests bắt buộc

## 13.1. Data discovery tests

```text
files discover được
pairing đúng
subject parser đúng
```

## 13.2. Shape tests

```python
assert image.shape == mask.shape
```

trước sample extraction.

3.5D:

```python
assert x.ndim == 3
assert x.shape[0] == EXPECTED_CONTEXT_SLICES
assert y.ndim == 2
```

## 13.3. Label tests

```python
assert set(np.unique(mask)).issubset(EXPECTED_LABELS)
```

Sau remap phải đúng project schema.

## 13.4. Leakage test

Không subject nào xuất hiện nhiều split.

## 13.5. CMR-MULTI reconstruction test

Verify:

```python
Z * T == original_N
```

và candidate confidence đạt threshold.

## 13.6. Dataloader smoke test

Load ít nhất:

```text
1 batch CMR-MULTI
1 batch CMRxMotion
1 mixed batch nếu mixed training
```

Run forward pass qua model:

```text
no shape error
no dtype error
no class mismatch
```

Không cần full training để pass smoke test.

---

# 14. Acceptance criteria

Task chỉ được coi là hoàn thành khi có đủ:

## Audit

```text
reports/cmr_multi_audit.csv
reports/cmr_multi_zt_inference.csv
reports/cmrxmotion_audit.csv
reports/dataset_compatibility.md
```

## Code

```text
dataset adapters
config
split logic
tests
```

## QC

```text
visual overlays
```

## Verification

Ít nhất:

```text
dataset loader smoke test passes
model forward pass passes
label schema verified
subject leakage test passes
```

## Documentation

README cuối phải nêu:

```text
dataset paths
expected structure
how to run audit
how to generate split
how to run adapter smoke test
how to train
known limitations
```

---

# 15. Expected deliverables

Agent nên tạo hoặc chỉnh theo convention repo, ví dụ:

```text
src/
└── data/
    ├── adapters/
    │   ├── cmr_multi.py
    │   └── cmrxmotion.py
    ├── cardiac_35d_dataset.py
    └── label_schema.py

scripts/
├── audit_cmr_multi.py
├── audit_cmrxmotion.py
└── verify_dataset_adapters.py

configs/
├── cmr_multi.yaml
└── cmrxmotion.yaml

reports/
├── cmr_multi_audit.csv
├── cmr_multi_zt_inference.csv
├── cmrxmotion_audit.csv
├── dataset_compatibility.md
└── qc/

tests/
├── test_cmr_multi_adapter.py
├── test_cmrxmotion_adapter.py
└── test_no_subject_leakage.py
```

Không bắt buộc đúng đường dẫn này nếu repo đã có architecture khác.

**Ưu tiên hòa vào architecture hiện tại thay vì tạo framework song song.**

---

# 16. Agent decision rules

## Nếu repo đã có Base Dataset/Adapter

Extend nó.

Không viết system thứ hai.

## Nếu preprocessing nằm trong Dataset class hiện tại

Tái sử dụng functions đó.

Không copy-paste preprocessing.

## Nếu label mapping hard-code nhiều nơi

Refactor tối thiểu về một source of truth trước khi thêm dataset mới.

## Nếu không xác định được semantic class mapping

STOP.

Không train.

## Nếu CMR-MULTI Z×T inference có case mơ hồ

Flag và exclude tạm thời.

Không silently chọn.

## Nếu CMRxMotion thiếu mask

Exclude khỏi supervised training.

Không tạo fake mask.

## Nếu orientation/affine bất thường

Visual verify trước khi sửa.

---

# 17. Training experiment plan sau khi adapters pass

Không merge dataset mới vào baseline ngay rồi kết luận.

Chạy ablation:

```text
E0 = current training data only
E1 = current + CMR-MULTI
E2 = current + CMRxMotion
E3 = current + CMR-MULTI + CMRxMotion
```

Giữ nguyên:

```text
model
seed
optimizer
scheduler
epochs
augmentation
validation set
metric implementation
```

So sánh:

```text
LV Dice
RV Dice
MYO Dice
HD95
per-dataset performance
```

Nếu có external test set độc lập, ưu tiên metric trên external test.

Mục tiêu là chứng minh dataset mới **cải thiện generalization**, không chỉ làm training Dice đẹp hơn.

---

# 18. Những câu hỏi agent bắt buộc phải trả lời trong report cuối

1. Model 3.5D hiện tại dùng bao nhiêu spatial slices?
2. Label schema hiện tại là gì?
3. Boundary slices xử lý như thế nào?
4. Current preprocessing gồm những bước nào?
5. CMR-MULTI có bao nhiêu case hợp lệ?
6. CMR-MULTI Z/T của từng case được xác định bằng cách nào?
7. Có case nào Z/T không chắc chắn không?
8. CMRxMotion có bao nhiêu subject thực tế?
9. Có bao nhiêu image volumes?
10. Có bao nhiêu labeled volumes?
11. Có file image không mask không?
12. Source label mapping của hai dataset là gì?
13. Có cần resample không?
14. Có nguy cơ leakage nào không?
15. Hai dataset có thực sự phù hợp với current model không?
16. Dataset nào nên dùng full, dataset nào cần restriction?
17. Adapter có chạy được forward pass với model hiện tại không?
18. Mixed training sampling strategy là gì?
19. Có thay đổi nào vào core training loop không? Vì sao?
20. Limitations còn lại là gì?

---

# 19. Definition of Done

Agent chỉ báo “done” khi:

```text
[ ] Repo architecture đã audit
[ ] Current model input contract đã xác minh
[ ] Current label schema đã xác minh
[ ] CMR-MULTI audit hoàn tất
[ ] CMR-MULTI Z×T reconstruction được verify
[ ] CMRxMotion audit hoàn tất
[ ] Missing GT được xử lý đúng
[ ] Subject grouping đúng
[ ] Compatibility report hoàn tất
[ ] Adapters được implement
[ ] Preprocessing match current pipeline
[ ] Visual QC pass
[ ] Unit tests pass
[ ] Dataloader smoke test pass
[ ] Model forward pass pass
[ ] No subject leakage
[ ] README usage hoàn tất
```

---

# 20. Điều quan trọng nhất

Không bắt đầu bằng câu hỏi:

> “Làm sao convert CMR-MULTI và CMRxMotion?”

Phải bắt đầu bằng:

> “Current project thực sự cần input/label/preprocessing contract nào, và hai dataset này có đáp ứng contract đó mà không phá semantic/geometry hay không?”

Adapter là bước sau cùng của audit, không phải bước đầu tiên.
