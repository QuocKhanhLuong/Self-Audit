# CineMA S1 source receipt

This note records the original primary-source inspection used by
`self_audit/candidate_pg/cinema_init.py`. No full CineMA weights were downloaded
or trained in that initial code-only delivery. Follow-up server download hashes
and software checks are recorded separately in `FIXES_RECEIPT.json`.

## Primary sources

- Hugging Face model card: <https://huggingface.co/mathpluscode/CineMA>
- Official pretrained YAML: <https://huggingface.co/mathpluscode/CineMA/blob/main/pretrained/config.yaml>
- Official implementation: <https://github.com/mathpluscode/CineMA/blob/main/cinema/convvit.py>
- Official MAE model/loading: <https://github.com/mathpluscode/CineMA/blob/main/cinema/mae/mae.py>
- Official feature extraction example: <https://github.com/mathpluscode/CineMA/blob/main/cinema/examples/inference/mae_feature_extraction.py>
- Official ACDC preprocessing: <https://github.com/mathpluscode/CineMA/blob/main/cinema/data/acdc/preprocess.py>

Correction from the follow-up server preflight: `b1251ee50423bceeca84c080782fc3bc7756dea6`
is the **Hugging Face model repository revision**, not a GitHub source commit.
The earlier attribution to GitHub was incorrect. The official GitHub repository
does not contain that commit; its observed HEAD was
`c10daa1d93f0ea28d8b9ad9206b0f673d25805c1`. A source-package revision must be
specified separately from the pretrained asset revision.
The Hub `pretrained` directory lists only the two allowed S1 files: a 348-byte
`config.yaml` (ETag `8a465e51889187cf1056fc4feaef9a0182feabe4`) and a
502,749,000-byte `cinema.safetensors` file (ETag
`a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590`). These
ETags are Hub metadata, not local weight hashes.

## Architecture and preprocessing facts

`pretrained/config.yaml` declares SAX spacing `[1.0, 1.0, 10.0]`, patch size
`[192, 192, 16]`, one channel, base model, ConvMAE patch `[4,4,1]`, scale
`[2,2,1]`, convolution channels `[64,128]`, and two blocks. The official
`DownsampleEncoder` therefore exposes conv skips at 1/4 and 1/8 XY, followed by
ViT tokens on a 1/16 XY lattice with z patch 1; the token embedding width/depth
for base are defined by the official `get_vit_config` (768/12/12).

The official feature example applies `ScaleIntensityd` followed by end
`SpatialPadd` to `192x192x16`. ACDC preprocessing first resamples to the
declared spacing, crops around the LV bounding box to a 192x192 XY canvas while
retaining the observed slice count, and then writes the processed image. PG
records the crop affine and excludes all end-padded cells from S1 cosine loss.

`cinema_init.py` resolves only the two Hub files above for a real run and uses
the official CineMA Python model to load the safetensors state strictly. The
test-only tiny encoder path is explicitly marked synthetic and cannot provide
real CineMA or patient evidence.

The local verification environment has `torch`, `safetensors`, `huggingface_hub`
and `PyYAML`, but not the official `cinema` package, `omegaconf`, or MONAI. A
real S1 run therefore requires installing the pinned official CineMA source
(`https://github.com/mathpluscode/CineMA`, with a separately verified source commit) and its dependencies;
the implementation fails closed when that dependency is unavailable.

## Safetensors header check

On 2026-11-16, an HTTP range request read only bytes `0-67,399` of the
502,749,000-byte safetensors object: the 8-byte header length was `67,392`, and
the parsed header contained 603 tensors. The SAX encoder keys include
`enc_down_dict.sax.conv_blocks.0.patch_embed.conv.weight` with shape
`[64,1,4,4,1]`, block 1 with `[128,64,2,2,1]`, `enc_down_dict.sax.patch_embed`
with `[768,512]`, `enc_down_dict.sax.pos_embed` with `[1,2304,768]`, and
`enc_fusion_dict.sax.down_convs.1` with `[768,128,2,2,1]`; the shared encoder
contains 12 blocks with 768-wide attention/MLP tensors. This agrees with the
official module chain and YAML-derived 1/8 conv tap plus 1/16 XY token lattice.
The complete weight object was not downloaded or deserialized in this local
software-only delivery; the real loader performs the strict full load when the
official `cinema` dependency and Hub cache are available.
