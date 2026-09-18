# Ferreira et al. (Nature Communications 2025): narrow methods and author-code audit

**Audit status:** COMPLETE for the requested narrow scope. This was a read-only static audit; no training, inference, or production files were run or modified.

**Paper identity:** Danielle L. Ferreira, Connor Lau, Zaynaf Salaymang, and Rima Arnaout, “Self-supervised learning for label-free segmentation in cardiac ultrasound,” *Nature Communications* 16, 4070 (2025), DOI [10.1038/s41467-025-59451-5](https://doi.org/10.1038/s41467-025-59451-5), PMCID [PMC12043926](https://pmc.ncbi.nlm.nih.gov/articles/PMC12043926/), PMID [40307208](https://pubmed.ncbi.nlm.nih.gov/40307208/). The full text was obtained from the [Europe PMC full-text XML endpoint](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12043926/fullTextXML); the supplemental Methods PDF is listed in the paper's [supplementary-files archive](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12043926/supplementaryFiles).

**Author-code snapshot:** [ArnaoutLabUCSF/cardioML](https://github.com/ArnaoutLabUCSF/cardioML), branch `master`, commit [`784799faded0169a15cce58b58a7d600c5582bae`](https://github.com/ArnaoutLabUCSF/cardioML/tree/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025) (the repository's NATCOMM_2025 addition, dated 2025-07-29 in the checked-out history). All code links below are pinned to this commit.

## Scope verdict

This is an echocardiography/cardiac-ultrasound segmentation pipeline. Cardiac MRI appears as a cross-modality evaluation comparator for a subset of 553 test echocardiograms; MRI is not the training input and this is not a cardiac-MRI segmentation paper. The external EchoNet-Dynamic evaluation set has manually annotated LV tracings, but those are used for external evaluation, not for training the internal pipeline.

The paper and public code support the claim that the internal segmentation models are trained from automatically generated weak labels and predictions rather than manual segmentation masks. The code-side QC thresholds and model checkpoints do not read clinical measurement files or manual masks. The paper nevertheless says that aggregate clinical shape/relationship information and clinical shape priors were used, and that preprocessing hyperparameters were visually reviewed on about 20 images. The source does not disclose whether any hidden clinical annotations contributed to those priors or threshold choices. Therefore the safe conclusion is **no explicit segmentation ground truth is shown in threshold/checkpoint selection; hidden provenance of the aggregate clinical priors is UNKNOWN**.

## What the paper says

### Weak-label extraction

The Methods and Supplemental Methods describe three view-specific routes.

* **A2C:** bilateral filtering for speckle reduction; Euclidean-distance seeds and watershed segmentation; a threshold-0.1 blood-pool mask; connected-components rejection if fewer than two chambers; LA/LV assignment from the known A2C spatial arrangement; then area/eccentricity and anatomy-based filtering. Supplemental Methods, PDF pp. 1–2.
* **A4C:** apply the final A2C U-Net to A4C images, yielding two LA and two LV predictions; use centroids and known spatial relationships to relabel RA/RV/LA/LV; require the expected four-component geometry and apply area/eccentricity/relationship QC. If RV height/LV height is below 0.8, stretch the RV to match the clinical length prior. Supplemental Methods, PDF pp. 1–2.
* **SAX-mid:** Hough-circle detection exploits the approximately circular LV; the circle prediction seeds a HED edge detector; the HED output is filled and used to train a first U-Net; self-learning recruits more labels; dilation/erosion create endocardial/epicardial labels for a second U-Net. Main Fig. 1 and Supplemental Methods, PDF p. 2.

The Supplemental Methods state that preprocessing/weak-label hyperparameters in the range 0.1x–10x of defaults were tested on a small sample (`n~20`), with the selected values based on visual review and the proportion of images producing plausible weak labels (Supplemental Methods, PDF p. 2). This is human visual review of weak-label plausibility, not reported manual segmentation ground truth; the paper does not provide per-image annotations or a blinded measurement protocol for this tuning.

### Shape QC

The paper reports clinical-knowledge QC thresholds (Supplemental Methods, PDF p. 2): LA/RA area 6–75 cm²; LV/RV area 4.7–104 cm²; eccentricity LA 0.30–0.96, RA 0.17–0.95, LV 0.62–0.96, RV 0.65–0.96. Connected components require two chambers for A2C and four for A4C. These thresholds are presented as shape/size/relationship priors, not as thresholds fit to manual chamber masks.

### Early learning and stopping

The paper says that the validation soft-Dice-loss curve was monitored in TensorBoard and that the transition from early learning to memorization was detected at the elbow, defined as the point of maximal curvature using standard methods such as `kneed`; training was stopped at that elbow. The paper's validation set is the patient-split training/validation data, whose labels are weak labels or self-learned labels rather than manual segmentation GT. No use of CMR, clinical measurements, or external manual tracings for this stopping decision is described.

### Self-learning

The paper says that a model trained during early learning on the small QC-passed set was run on all available training and validation images to recruit additional labeled examples and higher-quality examples for the next training round. This is pseudo-label/self-training, not a manual-label training stage. The paper does not give a confidence-calibration rule for self-learning beyond the described prediction/QC pipeline.

### Pretrained HED/ImageNet provenance

The paper explicitly states that the HED model was initialized with ImageNet weights and fine-tuned with the hyperparameters of Xie et al., using batch size 8. The U-Net architecture is described separately and the paper does not state ImageNet initialization for the U-Nets. The public HED code loads a local file named `vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5` by layer name. The filename is consistent with standard Keras VGG16 ImageNet “notop” weights, but the repository gives no download URL, checksum, or provenance record for that binary; provenance beyond the paper's ImageNet statement is therefore **partially verified, not independently reproducible from the repository metadata alone**.

## Author-code evidence at commit `784799f...`

The paper's code-availability statement points to [ArnaoutLabUCSF/CardioML](https://github.com/ArnaoutLabUCSF/CardioML). The repository's [NATCOMM_2025 README](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/README.md) says that only a small example dataset is included and that several example outputs are pre-generated from “real” models; it also warns that the tiny example models do not produce viable labels for some later steps. The README therefore exposes the pipeline skeleton, not the full private-data training release.

### A2C weak labels and QC

* [`01_generate_watershed_labels.py` L24–L50](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/01_generate_watershed_labels.py#L24-L50) loads `.npy` images, resizes to 256×256, normalizes, takes the grayscale channel, applies `unsharp_mask`, calls `get_label_ws`, and saves the categorical watershed result.
* [`utils/util_seg.py` L156–L245](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_seg.py#L156-L245) implements the watershed: threshold `<0.1`, removal of small objects/holes, distance-transform markers, watershed, perimeter/centroid filtering, closing/filling, and left/right categorization.
* [`02_quality_control_watershed_labels.py` L23–L37](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/02_quality_control_watershed_labels.py#L23-L37) applies the A2C area/eccentricity cutoffs and writes a passed-QC CSV. It reads no clinical measurement or manual-mask file.

The public implementation is not textually identical to every Supplemental-Methods detail: for example, the script calls `unsharp_mask(img_gray, 20, 20)` and passes `min_distance=20,c=20,l=150,r=260`, while the paper describes bilateral parameters `sigma_s=15,sigma_r=0.25`. The code's `get_label_ws` accepts `l` and `r` but the shown implementation uses its own small-object/hole constants. This is a reproducibility/version-drift note, not evidence of ground-truth use.

### A4C transfer, anatomy prior, QC, and RV stretch

* [`06_generate_a4c_labels_from_a2c_model.py` L28–L64](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/06_generate_a4c_labels_from_a2c_model.py#L28-L64) loads the A2C U-Net, predicts A4C, converts the two-channel A2C output to four chambers, applies `refine_chambers`, and saves the pseudo-label.
* [`utils/util_seg.py` L402–L451](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_seg.py#L402-L451) selects up to the two largest connected components per A2C chamber channel and assigns left/right by centroid; it raises only when the two selected centroids coincide.
* [`07_quality_control_a4c_from_a2c.py` L26–L47](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/07_quality_control_a4c_from_a2c.py#L26-L47) stretches RV first and then applies the hard-coded area/eccentricity thresholds for all four chambers.
* [`utils/util_seg.py` L454–L525](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_seg.py#L454-L525) implements the RV/LV length ratio test and stretches the RV when the ratio is `<0.8`.

The public A4C conversion does not visibly enforce a separate “exactly four connected components” check after selecting the two largest components per channel; the paper's stricter description may refer to the full internal pipeline or to behavior upstream of this helper.

### SAX Hough, HED, and shape filtering

* [`10_generate_hough_circle_labels.py` L27–L47](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/10_generate_hough_circle_labels.py#L27-L47) creates Hough-circle labels.
* [`utils/util_seg.py` L590–L615](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_seg.py#L590-L615) applies median blur, Laplacian/dilation/bilateral preprocessing, then `cv2.HoughCircles` with `dp=16`, `param1=100`, `param2=0.9`, and caller-supplied `minDist/minRadius/maxRadius`. The public caller uses `minDist=400,minRadius=30,maxRadius=100` (the Supplemental Methods report 400 and radii 20–80), another unresolved paper/code mismatch.
* [`model_arch/hed.py` L25–L37 and L71–L98](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/model_arch/hed.py#L25-L98) builds the five-side-output HED model, loads the local VGG16 file by layer name, and compiles the six outputs with class-balanced cross-entropy.
* [`model_arch/hed.py` L152–L191](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/model_arch/hed.py#L152-L191) shows the name-based HDF5 weight-loading implementation.
* [`13_generate_filled_hed.py` L61–L98](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/13_generate_filled_hed.py#L61-L98) selects the fused HED output (`pred_hed[5]`), thresholds it at 0.6, fills/dilates connected regions, keeps components with area 2,000–25,000 and eccentricity `<0.88`, keeps the largest component, and saves the filled label plus diagnostics. This is the explicit public SAX shape filter; it is not a manual-GT filter.

### Training, checkpointing, and self-learning in the public notebooks

The repository notebooks are examples with tiny data and pre-generated “real” labels in several places. Relevant pinned notebook lines:

* **A2C:** [`04_TrainingUnetA2C.ipynb` L128–L150](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/04_TrainingUnetA2C.ipynb#L128-L150) creates an 80/20 split; L364–L380 uses `ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='min')` for five epochs; L431–L473 shows self-learning label generation commented out and then switches the generators to `self_learning=True`.
* **A4C:** [`08_TrainingUnetA4C.ipynb` L125–L136](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/08_TrainingUnetA4C.ipynb#L125-L136) shuffles an 80/20 split; L316–L331 uses validation-loss checkpointing for five epochs; L379–L422 comments out pseudo-label generation and then uses `self_learning=True`; L565–L580 repeats validation-loss checkpointing for three epochs.
* **SAX HED:** [`12_Training_hed_Saxmid.ipynb` L84–L145](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/12_Training_hed_Saxmid.ipynb#L84-L145) shuffles an 80/20 split and loads Hough labels; L341–L354 trains for 20 epochs with `ModelCheckpoint(..., save_best_only=False)`, so the example saves every epoch rather than selecting a best checkpoint.
* **SAX U-Net/self-learning:** [`15_Training_unet_Saxmid.ipynb` L41–L61](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/15_Training_unet_Saxmid.ipynb#L41-L61) defines the filled-donut pseudo-label transform: prediction threshold 0.4, largest component, random erosion radius 3–5 and dilation radius 6–13. L373–L387 trains the first U-Net with validation-loss checkpointing, while L405–L420 comments out pseudo-label generation and L429–L463 switches the data generators to self-learning labels.
* [`utils/util_train.py` L188–L235](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_train.py#L188-L235) shows that ordinary generator validation uses `path_label`, while `self_learning=True` uses `path_self_learning`; both are `.npy` labels, not clinical measurements. The SAX branch is explicit at [L251–L287](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/utils/util_train.py#L251-L287).
* [`model_arch/unet.py` L38–L60 and L63–L119](https://github.com/ArnaoutLabUCSF/cardioML/blob/784799faded0169a15cce58b58a7d600c5582bae/NATCOMM_2025/model_arch/unet.py#L38-L119) defines the soft-Dice loss and U-Net. The U-Net has no ImageNet/pretrained-weight loading path; it is initialized by the Keras model constructor.

## Ground-truth and selection audit

| Decision/use | Evidence observed | Ground-truth status |
|---|---|---|
| Initial weak-label QC | Hard-coded area/eccentricity/connected-component and geometry rules in `02_quality_control_watershed_labels.py`, `07_quality_control_a4c_from_a2c.py`, `13_generate_filled_hed.py`; paper calls them clinical shape/size/relationship priors. | No explicit manual segmentation GT read. Exact provenance of the aggregate clinical priors is not disclosed; **UNKNOWN whether any hidden clinical annotations informed them**. |
| Preprocessing/weak-label hyperparameter tuning | Supplemental Methods says 0.1x–10x values were tested on about 20 images and chosen by visual review/plausible-label proportion. | Human visual review is acknowledged; no manual mask/clinical measurement target is reported. This is not evidence for a GT-based threshold, but it is a human-in-the-loop choice. |
| A2C/A4C/SAX U-Net checkpoint selection in public examples | A2C/A4C/SAX U-Net notebooks monitor `val_loss` and use `save_best_only=True`; `util_train.DataGenerator` supplies weak-label or self-learning `.npy` targets. | Validation targets are auto-generated labels, not reported manual GT. No clinical echo/CMR metric is used in these checkpoint callbacks. |
| HED checkpoint selection in public example | HED notebook uses `save_best_only=False` and runs 20 epochs, saving every epoch; final `.h5` is saved after the run. | No best-checkpoint selection shown. The paper's elbow rule is not implemented in this public notebook. |
| Paper early stopping | Methods says validation soft-Dice-loss elbow/maximal curvature, detected with TensorBoard/standard methods such as `kneed`. | No clinical/manual GT or CMR selection signal is described; the validation loss is based on generated labels. Public code does not expose the elbow detector, so exact full-run provenance is incomplete. |
| Self-learning recruitment | Public notebooks comment out generation for tiny examples and use pre-generated labels; active generator path switches to `path_self_learning`. | Pseudo-labels from the model/shape pipeline; no manual GT observed. Whether the full internal run applied additional QC before recruitment is not shown. |
| Final test evaluation | Internal clinical echo measurements and CMR report measurements are compared after inference; EchoNet-Dynamic manual LV tracings are used for external Dice evaluation. | These are evaluation references, not shown as training or checkpoint-selection inputs. |

## Bottom-line evidence classification for Astra

* **Verified:** echo/ultrasound target; A2C watershed, A4C A2C-transfer plus RV geometric correction, SAX Hough→HED→U-Net stages; stated shape QC thresholds; paper's validation-loss elbow description; self-learning from model predictions; paper's ImageNet HED initialization claim; CMR subset used as evaluation comparator; public code at the pinned commit.
* **Verified in public code:** no clinical measurement/CMR/manual-mask file is consumed by the shown QC scripts, training generators, or checkpoint callbacks; U-Net validation targets are weak/self-learned labels; HED public example saves all epochs.
* **Partially verified:** HED ImageNet weight provenance. The code contains the binary with the standard VGG16-notop filename and loads it by layer name, but no repository checksum/download provenance is provided.
* **Unresolved:** whether clinical aggregate shape priors or any hidden manual/clinical annotations informed threshold construction; whether the internal full run had additional QC or an elbow detector absent from the public example notebooks; the paper/code parameter mismatches for A2C preprocessing and SAX Hough radii/blur.
* **Not supported:** treating this work as cardiac-MRI segmentation, treating CMR or EchoNet manual tracings as training GT, or claiming that public notebook checkpoint behavior fully reproduces the paper's stated elbow stopping schedule.

No production or repository files under `/Users/alvinluong/Self-Audit` were changed. Scratch source/code checkout used: `/tmp/astra_luna_ferreira/`.
