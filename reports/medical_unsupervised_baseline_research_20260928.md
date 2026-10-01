# Research: baseline unsupervised y tế bổ sung cho CUTS trên ACDC — 2026-09-28 (rev. 2)

Phạm vi: chỉ research và đánh giá. Không sửa code benchmark, không train/inference/diffusion,
không đụng checkpoint, raw artifact hay dữ liệu. Code upstream chỉ được clone nông vào
scratchpad để đọc.

**[Suy luận]** đánh dấu hypothesis hoặc estimate của auditor, chưa được kiểm chứng. Tham chiếu
code dạng `file:line` là tại commit đã pin ở §9.

## 0. Thay đổi so với rev. 1

Tiêu chí đủ điều kiện đã được nới lỏng theo yêu cầu. Candidate chỉ cần là phương pháp **medical
imaging** thuộc loại **unsupervised / self-supervised / label-free segmentation**. Không còn yêu
cầu paper phải dùng ACDC, cardiac MRI hay nhiều class tim.

Hard gate giữ nguyên:

- **G1 — Không dùng GT.** Producer (train và inference) không dùng GT, pseudo-label, prompt hay
  support mask, và không chọn hyperparameter/level bằng Dice.
- **G2 — Raw partition GT-free.** Sinh được anonymous raw partition `[H,W]` không cần GT để
  adapter map sang BG/RV/MYO/LV.
- **G3 — Code và license.** Có official code với license rõ, hoặc ghi rõ là cần
  reimplementation.
- **G4 — Port được.** Port hợp lý sang 2D grayscale MRI 224×224.

Thay đổi về kết luận so với rev. 1:

- Method binary **không còn bị loại vì binary**. Mỗi method được phân tích xem cơ chế
  clustering có cho phép pre-register K > 2 hay không (§4).
- Shortlist giờ **xếp theo mức hữu ích của raw partition cho adapter**, không theo Dice oracle
  trong paper.
- **UnSeGArmaNet** lên shortlist.
- **Pix2Rep-v2** bị hạ khỏi shortlist. Nó là SSL encoder, không phải phương pháp
  segmentation, nên không đạt tiêu chí mới. Nó chỉ còn là ablation đổi encoder cho CUTS.
- **Mumford-Shah** chuyển từ NO-GO sang dự phòng (reimplementation).

---

## 1. Executive conclusion

1. **CUTS vẫn là baseline chính.** Đây là method y tế, multigranular, đã có đường producer
   GT-free trong repo.
2. **Shortlist ngoài CUTS (tối đa 3), xếp theo mức hữu ích của raw partition cho adapter:**

   | Hạng | Baseline | Verdict | Lý do chính |
   |---|---|---|---|
   | 1 | **DSS-US** (MICCAI 2024) | CONDITIONAL GO (reimplementation, trừ khi có license) | Là method duy nhất tạo **cluster ID nhất quán trên toàn dataset** với K cố định đã pre-register. Có bằng chứng unsupervised multi-class trên ảnh tim (echo CAMUS: LV/MYO/LA). Điểm yếu: độ phân giải patch ViT-S/8. |
   | 2 | **SGSCN** (MICCAI 2021) | CONDITIONAL GO (GPL-3.0) | Vốn đã multi-cluster, partition **ở mức pixel** với ràng buộc liên tục không gian. Rất rẻ, có thể dựng như một biến thể nhỏ trên runner DFC đã có trong repo. Điểm yếu: ID không nhất quán giữa các ảnh, số cluster thay đổi, và context loss có thể bất lợi cho vành MYO **[Suy luận]**. |
   | 3 | **UnSeGArmaNet** (BMVC 2024) | CONDITIONAL GO (MIT) | Loss modularity (DMON) và GNNpool nhận `num_clusters` bất kỳ, nên có thể pre-register K > 2 và cho **số component chính xác**. License rõ. Điểm yếu: bằng chứng chỉ có ở K=2, code chỉ hỗ trợ K=2, và luồng official có lật mask bằng GT IoU. |

3. **Dự phòng (không vào shortlist):**
   - **Mumford-Shah multiphase loss** (reimplementation). Cơ chế N-phase piecewise-constant rất
     hợp với cine bSSFP. Nhưng bằng chứng unsupervised của paper chỉ có trên BSDS500 (ảnh tự
     nhiên), và repo không có license.
   - **Pix2Rep-v2** dùng làm encoder cho CUTS diffusion (ablation đổi encoder).
4. **NO-GO:** UnSegMedGAT, SSL-ALPNet, UM-SAM, SDNet, MedSAM/SAM, MSSG, TUMLS, Moriya-JULE,
   hyperbolic 3D-VAE, ACWE-CNN. Lý do cụ thể ở §3–§4. DFC, STEGO và PiCIE không được đề xuất
   lại.
5. **Không candidate nào có bằng chứng native trên MRI tim.** Thứ hạng ở trên dựa trên đặc
   tính cơ chế của raw partition (§2.2), không dựa trên hiệu năng đã được chứng minh trên
   ACDC. Việc thứ hạng này có đúng hay không phải được kiểm chứng bằng raw-partition audit
   (§7, bước 2).

---

## 2. Khung đánh giá

### 2.1 Ba trục bắt buộc tách riêng

- **A. Native medical evidence:** paper đã chứng minh gì, trên dữ liệu y tế nào, bằng
  protocol nào (oracle hay GT-free).
- **B. Mức độ port sang ACDC 2D 224:** chuyển đổi đầu vào, số cluster, độ phân giải, patch
  cần sửa trong code, checkpoint.
- **C. Rủi ro khoa học:** leakage, non-determinism, độ nhạy hyperparameter, fragmentation,
  MYO dạng vành mỏng.

### 2.2 Tiêu chí xếp hạng: raw partition hữu ích cho adapter

Adapter GT-free phải tìm ra LV blood pool, RV blood pool, MYO và BG chỉ từ partition cùng đặc
tính ảnh. Mỗi candidate được chấm theo các tiêu chí sau (định nghĩa trước khi xếp hạng):

| Mã | Tiêu chí | Vì sao quan trọng với adapter |
|---|---|---|
| U1 | K > 2 có thể pre-register (cố định, hoặc có sàn/trần GT-free) | Cần ít nhất 4 vùng; K ngẫu nhiên khiến rule adapter bị lệch |
| U2 | Độ phân giải biên (pixel hay patch) | MYO chỉ dày vài pixel ở 224 **[Suy luận]** |
| U3 | Tính liên tục không gian của component | Tránh partition muối tiêu, dễ lấy connected component |
| U4 | ID nhất quán giữa các ảnh | Adapter có thể dùng thống kê cluster toàn dataset thay vì heuristic từng ảnh |
| U5 | Determinism, có thể seal | Contract Self-Audit |
| U6 | Có tín hiệu cường độ phân tách blood pool và cơ tim | bSSFP: máu sáng, cơ tối |

Mức hữu ích dự kiến (tất cả là **[Suy luận]** từ cơ chế, chưa đo):

| Candidate | U1 | U2 | U3 | U4 | U5 | U6 |
|---|---|---|---|---|---|---|
| DSS-US | ✅ cố định (K_seg, K_sem) | ⚠️ patch 8 px + CRF | ✅ eigen + CRF | ✅ **dataset-level** | ✅ có seed | ⚠️ gián tiếp (DINO + CRF) |
| SGSCN | ⚠️ sàn `minLabels`, trần 100 | ✅ pixel | ✅ loss liên tục | ❌ từng ảnh | ⚠️ cần sửa seed/cuDNN | ✅ trực tiếp |
| UnSeGArmaNet | ✅ K cố định (sau khi patch) | ⚠️ lưới stride 4 (~55×55) | ⚠️ modularity trên đồ thị patch | ❌ từng ảnh | ⚠️ cần seed | ⚠️ gián tiếp |
| (tham chiếu) CUTS persistent | ⚠️ số lượng thay đổi theo persistence | ✅ pixel | ⚠️ | ❌ từng ảnh | ✅ | ✅ |
| (dự phòng) Mumford-Shah N-phase | ✅ N cố định | ✅ pixel | ✅ TV | ✅ nếu sắp phase theo cường độ trung bình | ✅ | ✅ trực tiếp |

---

## 3. Longlist đánh giá lại

Ký hiệu gate: ✅ đạt, ⚠️ đạt nhưng có điều kiện hoặc cần sửa, ❌ không đạt.

| # | Candidate | A. Native medical evidence | K > 2? | G1 | G2 | G3 | G4 | Verdict rev. 2 |
|---|---|---|---|---|---|---|---|---|
| 0 | CUTS (MICCAI 2024) | retina, brain ventricle, brain tumor; mapping bằng GT | ✅ multigranular | ✅ (persistent / k-means với k cố định) | ✅ | ✅ Yale NC | ✅ đang port | **Giữ (baseline chính)** |
| 1 | **DSS-US** (MICCAI 2024) | 3 dataset US gồm **CAMUS tim 3 class**; eval bằng Hungarian/majority | ✅ native | ✅ nếu tắt eval/sweep | ✅ | ⚠️ **không LICENSE**, nên reimplement | ✅ | **Shortlist #1** |
| 2 | **SGSCN** (MICCAI 2021) | PH2 da, US u gan; binary qua max-overlap GT | ✅ native (sàn 3) | ✅ | ✅ (phải lưu map int) | ✅ GPL-3.0 | ✅ | **Shortlist #2** |
| 3 | **UnSeGArmaNet** (BMVC 2024) | Kvasir, CVC-ClinicDB, ISIC-2018, ETIS; binary | ⚠️ cơ chế cho phép, chưa kiểm chứng | ⚠️ phải bỏ GT-flip | ✅ sau patch | ✅ MIT | ✅ | **Shortlist #3** |
| 4 | UnSegMedGAT (arXiv 2024) | ISIC-2018, CVC-ColonDB; binary | ⚠️ có `K` và mode 2 tầng | ⚠️ GT-flip | ⚠️ | ❌ không LICENSE, thiếu file `gnn_pool_cc` | ⚠️ notebook Colab | NO-GO (dùng #3, cùng họ) |
| 5 | Mumford-Shah loss (TIP 2019) | medical chỉ semi-supervised (LiTS, BRATS); unsupervised chỉ trên BSDS500 | ✅ N kênh softmax | ✅ | ✅ | ❌ không LICENSE; không có script unsupervised | ✅ | **Dự phòng** (reimplementation) |
| 6 | Pix2Rep-v2 (MICCAI 2026) | ACDC, M&Ms, M&Ms-2 nhưng few-shot có mask; không phải method segmentation không nhãn | – (k-means là của ta) | ✅ nhánh SSL | ⚠️ derived | ✅ Apache-2.0; checkpoint chưa phát hành | ✅ | **Hạ xuống**: ablation encoder cho CUTS |
| 7 | SSL-ALPNet (ECCV 2020) | CHAOS, SABS, cardiac MRI | – | ❌ support mask lúc inference | ❌ | ✅ MIT | ✅ | NO-GO |
| 8 | UM-SAM (MICCAI 2025) | fetal brain, prostate | ❌ binary target | ❌ pseudo-label training + prior target | ❌ | ❌ không thấy repo | – | NO-GO |
| 9 | SDNet (MedIA 2019) | ACDC, semi-supervised | – | ❌ cần mask | – | ⚠️ Keras/TF1, không LICENSE | – | NO-GO |
| 10 | MedSAM / SAM-Med2D / SAM AMG | – | ⚠️ AMG chồng lấn | ❌ MedSAM học từ mask y tế; nhánh MedSAM trong repo UnSeGArmaNet còn lấy box từ GT | ⚠️ | ✅ | ✅ | NO-GO (chỉ diagnostic) |
| 11 | MSSG (MICCAI 2023) | GlaS histology; binary gland | ❌ cue đặc thù gland | ⚠️ | ⚠️ | ❌ không LICENSE; bản GlandSAM train bằng pseudo-label | ❌ cue màu/hình thái H&E | NO-GO |
| 12 | TUMLS (arXiv 2025) | WSI histology (UPENN-GBM, MoNuSeg) | ⚠️ | ✅ | ⚠️ | ❌ không tìm thấy code | ❌ pipeline đặc thù WSI | NO-GO |
| 13 | Moriya et al. JULE-3D (SPIE 2018) | micro-CT mẫu ung thư phổi, 3 vùng | ✅ k-means | ✅ | ✅ | ❌ không tìm thấy official code | ⚠️ 3D patch, độ phân giải patch | NO-GO (chỉ reimplement nếu cần thêm) |
| 14 | Hyperbolic 3D-VAE (Hsu et al., NeurIPS 2021) | biomedical 3D (theo abstract) | ✅ hierarchical | ✅ | ⚠️ | ⚠️ không xác minh được repo | ❌ gyroplane conv 3D; chuyển sang 2D làm đổi bản chất | NO-GO |
| 15 | ACWE-CNN (Chen & Frey, MIDL 2020) | SPECT xương; binary | ⚠️ ACWE nhiều pha chỉ là lý thuyết | ✅ | ✅ | ❌ không tìm thấy repo | ✅ | NO-GO (Mumford-Shah bao hàm) |
| – | DFC / STEGO / PiCIE | natural | – | – | – | – | – | Không đề xuất lại |
| – | DeepCut, Deep Spectral Methods, CLASP | natural (không phải medical) | – | – | – | – | – | Chỉ là nền tảng của #1, #3 và #4 |

Loại khỏi phạm vi vì dùng template/atlas mask hoặc shape prior học từ mask: Dalca et al.
(CVPR 2018, MICCAI 2019), Yu et al. (MIDL 2020, auto-encoder shape prior), CLMorph
(contrastive registration). Tất cả đều cần segmentation có nhãn làm prior, nên vi phạm G1.

---

## 4. Đánh giá chi tiết theo 3 trục

### 4.1 DSS-US — *Deep Spectral Methods for Unsupervised Ultrasound Image Interpretation* (Shortlist #1)

**A. Native medical evidence**

- Tmenova, Velikova, Saleh, Navab. MICCAI 2024, arXiv 2408.02043.
- 3 dataset ultrasound: carotid (349 ảnh), thyroid (634 ảnh), và **CAMUS echo tim** (ED/ES,
  50 bệnh nhân test, nhãn LV endocardium / MYO / LA).
- Dice cardiac: 45.32 ± 9.16 (step I, per-image segment) và 37.53 ± 6.5 (sau semantic
  clustering). Cả hai đều tính **sau Hungarian/majority matching** với GT, nên là oracle.
- Là method y tế duy nhất trong longlist có đánh giá **multi-class trên ảnh tim**.

**Cơ chế K > 2**

- Hai tầng K đều cố định trong config:
  - `spectral_clustering.K` = `segments_num` (mặc định top-level 15; sub-config 20): số
    eigensegment per-image.
  - `bbox.num_clusters` = `clusters_num` (15): semantic k-means ở **mức dataset**.
- Cả hai được đặt trước, không suy ra từ GT. `dataset.n_classes` chỉ được dùng trong bước eval.
- Vì vậy có thể pre-register, ví dụ `K_seg = 15, K_sem = 8`, mà không lệch khỏi thiết kế gốc.

**Pipeline**

```text
ảnh → DINO ViT-S/8 (key features, lớp cuối; configs/model/dino_vits8.yaml)
    → affinity = DINO (threshold ≥ 0) [+ tuỳ chọn SSD/MI/NCC/pos, mặc định trọng số 0]
    → Laplacian eigenvectors → k-means → eigensegments per-image
    → semantic k-means dataset-level trên feature crop [+ mask/pos embedding, mặc định 0]
    → DenseCRF → partition
```

**Nơi dùng GT**

- `pipeline.py:423-431` gọi `evaluate(... gt_dir ...)`, dùng
  `evaluate_dataset_with_remapping` (Hungarian/majority). **Chỉ dùng trong evaluator.**
- `configs/sweep/*.yaml` quét affinity và số cluster bằng wandb, có log Dice/mIoU.
  **Không được dùng.**

**B. Port sang ACDC**

- Replicate 1→3 kênh và chuẩn hoá ImageNet (config `norm: imagenet`). Protocol `[-1,1]` cần
  được map lại về `[0,1]` và khai báo.
- Patch 8 ở 224 cho lưới 28×28. Có thể chạy ở 448 (lưới 56×56) rồi downsample partition về
  224. Việc chọn 224 hay 448 phải pre-register.
- Affinity ultrasound (SSD/MI) và mask/pos prior đều mặc định tắt trong sub-config.
  Paper dùng các thành phần này cho biến thể tốt nhất, nhưng trọng số được chọn trên
  ultrasound. **[Suy luận]** Nên pre-register hai biến thể: (a) chỉ DINO, (b) DINO + pos
  prior với trọng số lấy từ paper. Không được chọn giữa hai biến thể bằng Dice ACDC.
- Semantic k-means phải fit trên đúng `fit_set` của manifest.
- Không cần train; chỉ cần checkpoint DINO ViT-S/8 (DINO self-supervised trên ImageNet-1k,
  không dùng nhãn).
- Hỗ trợ DINOv2 đòi hỏi clone một fork đã sửa (`dino2_models/readme.md`), nên **không dùng**
  nhánh này.
- **License:** repo và `lukemelas/deep-spectral-segmentation` đều không có license. Khuyến
  nghị **reimplementation** theo paper (các bước DSM chuẩn đều ngắn), ghi nhãn "DSS-US
  reimplementation". Hoặc xin phép tác giả bằng văn bản.

**C. Rủi ro khoa học**

- MYO bị giới hạn bởi độ phân giải patch; CRF chỉ bù được một phần.
- "Tuning ngầm" nếu thử nhiều cặp K/affinity rồi nhìn kết quả.
- Có non-determinism ở k-means, eigen solver và CRF; cần seed và kiểm tra hash.
- Semantic clustering dataset-level phụ thuộc thành phần của `fit_set`, nên split phải cố định.
- **[Suy luận]** DINO ImageNet có thể gom LV và RV blood pool vào cùng cluster. Điều này chấp
  nhận được, vì adapter tách chúng bằng connected component và vị trí.

**Vì sao xếp #1:** là candidate duy nhất đạt U4 (ID nhất quán dataset-level). Đồng thời đạt U1
(K cố định) và có bằng chứng multi-class trên tim.

### 4.2 SGSCN — *Spatial Guided Self-supervised Clustering Network* (Shortlist #2)

**A. Native medical evidence**

- Ahn, Feng, Kim. MICCAI 2021, arXiv 2107.04934.
- PH2 (da, 200 ảnh) và SYSU liver-tumour US (100 ảnh). Cả hai là binary.
- Paper đánh giá bằng "cluster có overlap lớn nhất với GT", nên là oracle.
- Paper vượt k-means, DeepCluster và IIC trên cùng protocol oracle.

**Cơ chế K > 2**

- Output có `nChannel = 100` kênh; argmax cho số label giảm dần trong quá trình tối ưu.
- Dừng khi `nLabels <= minLabels` (mặc định 3, `demo_final.py:169`) hoặc khi đủ
  `maxIter = 50`.
- Vậy K luôn ≥ 3 khi dừng sớm, và có thể lớn hơn nếu chạy hết maxIter. Đây là **sàn GT-free
  có thể pre-register** (ví dụ `minLabels = 4`), nhưng **không phải K cố định**.

**Nơi dùng GT:** demo không đọc GT.

**Lỗi code cần sửa** (không liên quan GT):

- Output bị ghi thành ảnh RGB bằng bảng màu ngẫu nhiên không seed (`demo_final.py:119,180-185`).
  Phải lưu map `int` thay thế.
- Context-based consistency loss **mặc định tắt** (`--center`, dòng 128/160), trong khi
  method của paper bật nó. Phải pre-register `--center` = bật.
- `src/center.py:8-9` hard-code `.cuda()`.

**B. Port sang ACDC**

- Về bản chất là Kanezaki/DFC (nConv=2, 100 kênh, SGD lr 0.1, momentum 0.9) cộng thêm
  context loss.
- Repo đã có runner DFC "READY LOCAL" (`reports/FINAL_DFC_REAUDIT.md`), với 1 kênh, reseed
  từng ảnh, seal và GT firewall. **[Suy luận]** SGSCN có thể được dựng thành một biến thể
  cấu hình của runner này: thêm context loss, `maxIter = 50` theo SGSCN thay vì 1000 của DFC,
  `minLabels` theo pre-register. Chi phí implement vì vậy thấp nhất trong shortlist.
- Đầu vào 1 kênh: demo đọc 3 kênh bằng `cv2`. Runner DFC đã chạy 1 kênh; cần khai báo đây là
  chuyển đổi đầu vào.

**C. Rủi ro khoa học**

- Cùng họ DFC. Brief xếp DFC là "không mặc định phù hợp"; SGSCN chỉ khác ở context loss và
  có bằng chứng y tế.
- **[Suy luận]** Context loss phạt phương sai không gian quanh tâm cluster, nên bất lợi cho
  cấu trúc hình vành: tâm vành MYO nằm trong LV cavity. Nó có thể kéo MYO bị cắt thành cung,
  hoặc gộp vào vùng khác.
- ID không nhất quán giữa các ảnh (U4 ❌), K thay đổi, và nhạy với `stepsize_ss` và
  `minLabels`.
- Không deterministic nếu không cố định seed và cuDNN.

**Vì sao xếp #2:** partition ở mức pixel, dùng trực tiếp tín hiệu cường độ và có ràng buộc
liên tục (U2, U3, U6 tốt nhất trong nhóm per-image). Chi phí gần như bằng 0 nhờ hạ tầng DFC có
sẵn. Nó cũng đóng vai trò control: "method y tế cùng họ DFC có hơn DFC không".

### 4.3 UnSeGArmaNet (Shortlist #3)

**A. Native medical evidence**

- Paper: arXiv 2410.06114, BMVC 2024. Theo README: Kvasir, CVC-ClinicDB, ISIC-2018, ETIS
  (medical, binary), cộng ECSSD/DUTS/CUB.
- Eval tính IoU với GT; code lật mask theo IoU tốt hơn (`segment.py:85-88`), nên là oracle.
- **README ghi repo hiện chỉ hỗ trợ CUB/ECSSD/DUTS**: loader cho dataset y tế chưa được phát
  hành. Kết quả y tế vì vậy **không reproduce được trực tiếp từ repo**.

**Cơ chế K > 2**

- `GNNpool(input_dim, conv_hidden, mlp_hidden, num_clusters, ...)` (`gnn_pool.py`) tổng quát
  cho mọi K.
- DMON loss dùng `torch.eye(num_clusters)` và trace modularity, nên đúng với K bất kỳ. NCUT
  loss cũng vậy.
- Tuy nhiên `segment.py:36` hard-code `num_clusters = 2`, còn `util.graph_to_mask:34-35` lật
  mask theo 4 góc, heuristic này chỉ có nghĩa với K = 2.
- Kết luận: **có thể pre-register K > 2 bằng một patch nhỏ** (tham số hoá K, bỏ heuristic
  góc, xuất argmax int). Nhưng **không có bằng chứng thực nghiệm nào ở K > 2** trong paper
  hay repo. Đây là mở rộng do auditor đề xuất, phải báo cáo là "UnSeGArmaNet, K = k
  (extension)".

**Nơi dùng GT**

- `segment.py:85-88`: IoU với mask rồi lật. **Không được dùng; chỉ đưa vào evaluator.**
- `MEDSAM_INFERENCE` (`segment.py:48, 102`) lấy bounding box từ GT. **Không được dùng.**
- `main.py` duyệt `(img, mask)` và in mIoU. Producer phải là một entry point riêng.

**B. Port sang ACDC**

- DINO ViT-S/8 (checkpoint `dino_deitsmall8_pretrain`, ImageNet-1k, SSL không nhãn).
- Feature stride 4 cho lưới khoảng 55×55 ở 224, mịn hơn DSS-US gốc (28×28).
- Replicate 3 kênh.
- Tối ưu per-image 20 epoch.
- Hyperparameter cần pre-register: `threshold` của adjacency (mặc định 0; README ví dụ 0.1),
  `loss_type` (DMON), `conv_type` (ARMA), `activation` (selu), K.
- Dependency: Python 3.11, PyTorch 2.0, `torch_geometric==2.0.4`, `torch-scatter` và
  `torch-sparse` build theo phiên bản torch, `deeplake==3.9.0`. `script.sh` còn tự cài MedSAM
  và tải file từ Google Drive; **bỏ các phần này** trong producer.
- **[Suy luận]** `torch_geometric` 2.0.4 là bản cũ và có thể xung đột với PyTorch/CUDA mới
  trên server. Đây là rủi ro môi trường chính.

**C. Rủi ro khoa học**

- K > 2 chưa từng được kiểm chứng. Modularity trên đồ thị patch có thể cho các cụm tách rời
  (một cluster gồm nhiều mảnh).
- Adjacency nhị phân hoá theo `threshold` rất nhạy.
- Độ phân giải vẫn ở mức patch; bilateral solver chỉ được thiết kế cho mask binary.
- ID không nhất quán giữa các ảnh. Có dropout trong ARMA/MLP lúc train, nên cần seed.

**Vì sao xếp #3:** đạt U1 tốt nhất (K chính xác), license rõ nhất (MIT), và lưới mịn hơn
DSS-US. Nhưng không có tầng dataset-level (U4), không có bằng chứng K > 2, và môi trường khá
rủi ro.

### 4.4 UnSegMedGAT — NO-GO

- **A.** arXiv 2411.01966. ISIC-2018 và CVC-ColonDB, mIoU binary, tối ưu per-image
  (55–300 epoch).
- **K > 2:** code có `K` và "mode 1/2" (tầng 1 tách foreground K=2, tầng 2 chia foreground
  thành `foreground_k` cụm; `segment.py:53-56, 73`), kế thừa từ DeepCut. Về cơ chế, đây là
  đường multi-component tự nhiên nhất trong họ GNN.
- **Lý do NO-GO:**
  - G3 ❌: không có file LICENSE.
  - Code chưa đầy đủ: `segment.py:65` import `gnn_pool_cc`, file không tồn tại trong repo.
  - Phụ thuộc Colab (`google.colab.patches`).
  - Hyperparameter khác nhau theo dataset (`alpha` 0.5 vs 0.35, `activ` "SiLU_GAT" vs
    "SiLU_GAT3" trong notebook), không có quy trình chọn GT-free.
  - Cùng họ với UnSeGArmaNet: GT-flip ở `segment.py:199-203`.
- Nếu muốn thử cơ chế 2 tầng thì làm trên nền MIT của UnSeGArmaNet/DeepCut và ghi là
  extension.

### 4.5 Mumford-Shah loss — dự phòng (reimplementation)

- **A.** Kim & Ye, IEEE TIP 2019 (arXiv 1904.02872). Medical (LiTS, BRATS) chỉ ở chế độ
  **semi-supervised**; unsupervised chỉ có trên BSDS500. **Bằng chứng unsupervised y tế
  native: không có.** Đây là lý do nó không vào shortlist theo tiêu chí mới.
- **K > 2:** `models/loss.py::levelsetLoss` tính centroid và phần dư cho mọi kênh softmax,
  nên N-phase là native. N cố định, pre-register được.
- **Tính hữu ích cho adapter [Suy luận]:**
  - Trên cine bSSFP, piecewise-constant N-phase tách trực tiếp máu sáng / cơ tối / nền.
  - Mạng được train amortized trên cả tập ảnh, nên các kênh có thể nhất quán. Sắp phase theo
    cường độ trung bình cho một ID deterministic mà không cần GT.
  - Điểm yếu: MYO có cường độ gần gan/cơ ngực, nên một phase có thể gồm nhiều mô; adapter
    phải xử lý bằng không gian.
- **B.** Repo không có LICENSE, Python 2.7 / PyTorch 1.1, chỉ có script LiTS có giám sát.
  → Reimplement loss (~30 dòng theo paper) cộng TV (`gradientLoss2d`) cộng một U-Net chuẩn.
  Phải ghi là "Mumford-Shah unsupervised, reimplementation".
- **C.** Nhạy với trọng số TV (β), N, và khởi tạo; có thể suy biến (phase rỗng); bias field.
- **Verdict: dự phòng.** Dùng nếu cả ba candidate shortlist thất bại ở raw-partition audit,
  hoặc nếu nhóm chấp nhận "paper medical-oriented + bằng chứng unsupervised trên ảnh tự
  nhiên".

### 4.6 Pix2Rep-v2 — hạ khỏi shortlist

- Đây là pretraining dense SSL. Paper đánh giá bằng few-shot finetune / linear probe /
  in-context có mask; bản thân nó **không phải method segmentation không nhãn**. Clustering
  head là do ta ghép vào. Không đạt tiêu chí "làm unsupervised segmentation".
- Loader SSL (`pix2repv2/data/acdc.py::ACDCPatchSSL`) chỉ đọc `*_4d.nii.gz`, GT-free;
  `ACDCPatchSupervised` thì đọc `_gt`.
- Checkpoint chưa phát hành; repo mới có 1 commit.
- **Giữ làm ablation encoder cho CUTS** ("CUTS-diffusion on Pix2Rep-v2 features"), sau
  shortlist.

### 4.7 Các NO-GO còn lại (tóm tắt)

- **SSL-ALPNet:** inference cần support mask.
- **UM-SAM:** train bằng pseudo-label, prior thuộc tính target, binary.
- **SDNet:** cần mask.
- **MedSAM:** được train bằng annotation.
- **MSSG:** cue đặc thù histology, không LICENSE.
- **TUMLS:** pipeline WSI, không có code.
- **Moriya JULE-3D:** không có official code.
- **Hyperbolic 3D-VAE:** method 3D, chuyển sang 2D làm đổi bản chất.
- **ACWE-CNN:** không có code, được Mumford-Shah bao hàm.

---

## 5. Shortlist cuối (ngoài CUTS)

| Hạng | Baseline | Nhãn báo cáo | Gate trước khi implement |
|---|---|---|---|
| 1 | DSS-US | "DSS-US (reimplementation)" hoặc "official" nếu có license | (a) Quyết định license. (b) Pre-register K_seg/K_sem, độ phân giải 224/448, biến thể affinity (tối đa 2). (c) Tắt eval/sweep. |
| 2 | SGSCN | "SGSCN (official code, GPL-3.0; port lên runner DFC)" | (a) Pre-register `--center` = bật, `minLabels`, `maxIter`. (b) Lưu map int. (c) Seed và cuDNN deterministic. |
| 3 | UnSeGArmaNet | "UnSeGArmaNet, K = k (extension ngoài paper)" | (a) Patch: tham số hoá K, bỏ GT-flip, bỏ heuristic góc và nhánh MedSAM. (b) Pre-register K, threshold, loss/conv/activation. (c) Dựng được môi trường `torch_geometric`. |

Dự phòng: Mumford-Shah N-phase (reimplementation), Pix2Rep-v2 encoder cho CUTS.

---

## 6. Artifact contract (giữ nguyên từ rev. 1, bổ sung theo từng method)

Contract chung:

- **Producer:** `raw/<sample_id>.npz` chứa `partition` `int32 [224,224]` với ID anonymous,
  không có khoá `label`/`gt`. Kèm `sample_id` (ACDC, patient, ED/ES, frame, z),
  `image_sha256` và `preproc_id`.
- **Manifest:** `method`, `method_variant`, nhãn `official|reimplementation|extension`,
  `upstream_repo`/`upstream_commit`/`local_patch_sha256`, `config_sha256`, `preregistered_at`
  (commit của file pre-registration, phải trước mọi lần mount GT), `checkpoint_sha256` kèm
  dữ liệu pretrain, `fit_sets`, `seeds`/`determinism`/`env`, `gt_access_audit`, và `seal`
  (hash từng file và manifest).
- **Adapter:** chỉ đọc `raw/` đã seal và ghi `adapted/` với provenance riêng.
- **Evaluator (process riêng, GT read-only):**
  - (i) runtime Dice của adapter: đây là số duy nhất được báo cáo là kết quả;
  - (ii) oracle Hungarian/majority/max-overlap, gắn nhãn rõ;
  - (iii) chẩn đoán MYO (số connected component, vành khép kín), phải định nghĩa trước.

Trường riêng theo method:

| Method | Trường bổ sung |
|---|---|
| DSS-US | `dino_checkpoint_sha256`, `K_seg`, `K_sem`, `affinity_weights`, `crf_params`, `semantic_fit_set`, `input_resolution` |
| SGSCN | `nChannel`, `minLabels`, `maxIter`, `center_loss=true`, `stepsize_ce/ss`, `n_labels_final`, `stop_reason` (minLabels hay maxIter), `per_sample_seed` |
| UnSeGArmaNet | `K`, `adj_threshold`, `loss_type`, `conv_type`, `activation`, `epochs`, `dino_checkpoint_sha256`, `patch_list` (các patch đã áp so với upstream) |

---

## 7. Thứ tự thực nghiệm đề xuất

| Bước | Nội dung | GT? | Pass gate |
|---|---|---|---|
| 0. Pre-register | Freeze config của 3 method (mỗi method tối đa 2 biến thể). Commit file pre-registration. | Không | Commit có hash |
| 1. Smoke nhỏ | 2–3 bệnh nhân × ED/ES. Kiểm tra shape/dtype, chạy 2 lần cho hash giống nhau, `gt_access_audit` rỗng, seal được, đo thời gian/VRAM. | Không | Deterministic, không truy cập GT |
| 2. Raw-partition audit + oracle | Thống kê GT-free trên tập dev của manifest: phân bố số component, tỉ lệ component rời rạc, kích thước mảnh nhỏ nhất, ID consistency (DSS-US), trực quan hoá. Evaluator tính oracle upper bound (majority/Hungarian) cho CUTS và cả 3 method trên **cùng sample**. | Có, chỉ ở evaluator | Tiếp tục nếu oracle MYO/RV **không thấp hơn** CUTS trên cùng tập. Đây là quyết định triage, ghi vào log; không sửa config sau bước này. |
| 3. Adapter reapply | Áp nguyên adapter GT-free hiện có, không chỉnh riêng cho từng method. | Không | Output 4 class kèm provenance |
| 4. Dice / visualize tách rời | Runtime Dice per-class, HD95, metric vành MYO, overlay. Oracle ở cột riêng. | Có (evaluator) | Báo cáo đầy đủ |
| 5. Full run | Toàn bộ manifest, ≥3 seed, so với CUTS cùng split/adapter/evaluator. | Có (evaluator) | Seal hash; nhãn official/reimpl/extension |

Nếu config thay đổi sau bước 2, đó là một biến thể mới. Biến thể đó chỉ được báo cáo nếu được
đánh giá lại trên một split giữ riêng chưa dùng để ra quyết định.

Thứ tự triển khai (theo chi phí và rủi ro, không theo hạng):

1. **SGSCN:** gần như miễn phí trên runner DFC có sẵn.
2. **DSS-US:** không train, chủ yếu là công reimplement.
3. **UnSeGArmaNet:** cần patch và dựng môi trường `torch_geometric`.

---

## 8. Integration plan mức file/module (không viết code)

### 8.1 SGSCN → biến thể trong `baseline/DFC/`

- Thêm profile cấu hình `sgscn_p0`: `maxIter = 50`, `minLabels` theo pre-register,
  `center_loss = true`. Profile này tách biệt khỏi profile DFC immutable (MinL3/maxIter1000)
  để không làm đổi identity của DFC.
- Thêm module context loss, port từ `src/center.py` và hàm `get_variance` (`demo_final.py`).
  Bỏ `.cuda()` hard-code, dùng device của runner. Ghi attribution GPL-3.0; code phái sinh
  chịu GPL-3.0.
- Provenance: `method = SGSCN`, `upstream_commit = 592efb6`, cùng các trường ở §6.
- Test: context loss bằng 0 khi mọi pixel về một tâm (unit test tổng hợp); không truy cập
  GT; determinism.

### 8.2 DSS-US → `baseline/DSS_US/` (reimplementation)

- `src/cardiac_benchmark/dataset.py`: loader image-only theo manifest chung, `[-1,1]` →
  `[0,1]`, 3 kênh.
- `src/cardiac_benchmark/features.py`: DINO ViT-S/8 key features lớp cuối, stride theo config.
- `src/cardiac_benchmark/spectral.py`: affinity (≥ 0), Laplacian, eigen, k-means
  (oversegmentation).
- `src/cardiac_benchmark/semantic.py`: feature segment + k-means dataset-level, fit trên
  `fit_set`.
- `src/cardiac_benchmark/crf.py`: DenseCRF với tham số từ config.
- `src/cardiac_benchmark/{manifest,provenance,output,freeze}.py`: dùng lại theo mẫu
  `baseline/PICIE/src/cardiac_benchmark/`.
- `config/dss_us_acdc_{a,b}.yaml`: 2 biến thể đã pre-register.
- `REIMPLEMENTATION.md`: đối chiếu từng bước với paper; nếu có tham khảo repo upstream thì
  ghi rõ (không license, không copy code).
- Tests: không truy cập GT, determinism, `fit_set` chỉ chứa sample có trong manifest.

### 8.3 UnSeGArmaNet → `baseline/UNSEGARMANET/`

- `upstream/`: vendor tại `fc5247a` (MIT), kèm `UPSTREAM.md` liệt kê patch.
- Patch 1: `Segmentation.__init__` nhận `K`.
- Patch 2: bỏ `iou` và GT-flip khỏi producer.
- Patch 3: `graph_to_mask` bỏ heuristic góc khi K > 2, trả map int.
- Patch 4: xoá nhánh `MEDSAM_INFERENCE` và `KMEANS_DINO` khỏi producer.
- `src/cardiac_benchmark/producer.py`: loop per-image với seed, xuất `partition`.
- `environment/`: pin `torch`, `torch_geometric`, `torch-scatter`, `torch-sparse` tương thích
  CUDA của server; bỏ bước tải MedSAM và `gdown` trong `script.sh`.
- Tests: kiểm tra tĩnh không import `SamModel` và không đọc mask; K component tối đa; determinism.

---

## 9. Nguồn primary

| Kết luận | Nguồn |
|---|---|
| CUTS: phương pháp, dataset, mapping bằng GT | https://arxiv.org/abs/2209.11359 · https://github.com/KrishnaswamyLab/CUTS (HEAD `27751160`, LICENSE.md Yale NC) |
| DSS-US: paper, CAMUS 3 class, Hungarian/majority, Dice 45.32/37.53 | https://arxiv.org/abs/2408.02043 · https://papers.miccai.org/miccai-2024/202-Paper0483.html |
| DSS-US code: K/clusters config, eval dùng GT, sweep, không LICENSE, fork DINOv2 | https://github.com/alexaatm/UnsupervisedSegmentor4Ultrasound/tree/d4ac44c60df18b921c590796f6994a4c8ac0726c |
| SGSCN: paper, dataset, eval max-overlap | https://arxiv.org/abs/2107.04934 |
| SGSCN code: `minLabels`, `--center`, output màu, `.cuda()`; GPL-3.0 | https://github.com/osmond332/Spatial_Guided_Self_Supervised_Clustering/tree/592efb6e72ceeef15c8be0630a4673eda5dce6f5 |
| UnSeGArmaNet: paper, dataset y tế, K=2 hard-code, GT-flip, MedSAM box từ GT, MIT, chỉ hỗ trợ CUB/ECSSD/DUTS | https://arxiv.org/abs/2410.06114 · https://github.com/ksgr5566/UnSeGArmaNet/tree/fc5247ab34a7d17a2ec2b297354cc4d681644d48 |
| UnSegMedGAT: K / mode 2 tầng, thiếu `gnn_pool_cc`, GT-flip, không LICENSE | https://arxiv.org/abs/2411.01966 · https://github.com/mudit-adityaja/UnSegMedGAT/tree/7f5a0e53bb11da7608c07af4f364140c3d6366cf |
| DeepCut (nền tảng MIT của họ GNN) | https://github.com/SAMPL-Weizmann/DeepCut |
| Mumford-Shah: unsupervised chỉ trên BSDS500, `levelsetLoss` N kênh, không LICENSE | https://arxiv.org/abs/1904.02872 · https://github.com/jongcye/CNN_MumfordShah_Loss/tree/4e48c905b5f860cb8c1ecfd75768a0984bd77f02 |
| Pix2Rep-v2: SSL, few-shot, loader SSL GT-free, Apache-2.0 | https://arxiv.org/abs/2609.01427 · https://github.com/BioMedTP/pix2rep-v2/tree/51787c11f9c96f8d2709ca37b621bd95379b4778 |
| SSL-ALPNet | https://arxiv.org/abs/2007.09886 |
| UM-SAM | https://papers.miccai.org/miccai-2025/paper/2296_paper.pdf |
| SDNet | https://arxiv.org/abs/1903.09467 |
| MSSG | https://arxiv.org/abs/2307.11989 · https://github.com/xmed-lab/MSSG |
| TUMLS | https://arxiv.org/abs/2504.12718 |
| Moriya JULE-3D | https://arxiv.org/abs/1804.03830 |
| Hyperbolic 3D-VAE | https://arxiv.org/abs/2012.01644 |
| ACWE-CNN | https://arxiv.org/abs/2001.10155 |
| Hạ tầng DFC trong repo | [reports/FINAL_DFC_REAUDIT.md](FINAL_DFC_REAUDIT.md) |

License và HEAD commit được lấy qua GitHub REST API và `git clone --depth 1` ngày
2026-09-28.

## 10. Hạn chế

- Không chạy code nào. Mọi chấm điểm U1–U6 và dự báo về MYO đều là **[Suy luận]** từ cơ chế.
- Chưa đọc hết `extract/` và `multi_region_segmentation` của DSS-US. Chưa lấy được trọng số
  affinity/prior chính xác mà paper dùng cho biến thể tốt nhất.
- UnSeGArmaNet: chưa kiểm tra `features_extract.py`/`extractor.py` ở mức từng dòng; stride 4
  được suy ra từ `util.graph_to_mask`.
- Tìm kiếm không đầy đủ; có thể còn method y tế unsupervised chưa được index hoặc mới công bố.
