# Candidate PG source provisioning

This procedure provisions the external **source code** and optionally the two permitted CineMA pretrained files. It does not download ACDC/M&Ms data, prepare M&Ms, run a server, initialize CUDA, or start training. The CLI is read-only unless `--install` and/or `--download-cinema` are supplied explicitly.

## Locked sources

| Source | Repository and revision | Integrity evidence |
|---|---|---|
| MedNeXt v1 factory | `https://github.com/MIC-DKFZ/MedNeXt`, `0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba` | Git HEAD, origin, clean worktree; runtime loader checks this exact SHA. |
| CineMA implementation | `https://github.com/mathpluscode/CineMA`, `c10daa1d93f0ea28d8b9ad9206b0f673d25805c1` | GitHub API confirms this commit exists; the pinned `cinema/mae/mae.py` exports `get_model(config: DictConfig) -> CineMA`, matching the loader import. Checkout verifies Git HEAD, origin, and clean worktree; code must be on `PYTHONPATH`. |
| CineMA Hub model revision | Hugging Face API model SHA `b1251ee50423bceeca84c080782fc3bc7756dea6` | Queried independently at `https://huggingface.co/api/models/mathpluscode/CineMA`; response field `sha` was this value. It is intentionally a different pin from the GitHub source commit. |
| CineMA architecture config | `mathpluscode/CineMA` at the Hub revision above, `pretrained/config.yaml` | Hub tree API reports Git blob SHA-1 `8a465e51889187cf1056fc4feaef9a0182feabe4`, size 348 bytes. The provisioner verifies this published blob ID and records the computed raw-byte SHA-256; it does not guess or mislabel a SHA-256 for config. |
| CineMA backbone weights | `mathpluscode/CineMA` at the Hub revision above, `pretrained/cinema.safetensors` | Hub tree API reports LFS SHA-256 `a841b7e9300d60ea8feb73e3a7d8c60ddedc1d1d49ec1049ebfcfa337c5d5590`, size 502,749,000 bytes. |

Only the two CineMA paths above are requested from Hugging Face. No task-finetuned ACDC, M&Ms, or M&Ms2 checkpoint is accepted. The earlier [CINEMA_SOURCE.md](../pg-k8-20261116/CINEMA_SOURCE.md) used the Hub SHA as a GitHub revision; treat that historical GitHub pin as superseded by the two independent pins above. No weights were downloaded or deserialized during this worker task.

## Runtime and dependency setup

Use the existing PG Python environment and preserve its installed PyTorch/CUDA wheel. `requirements.txt` already specifies `dynamic-network-architectures==0.4.4` and NumPy/SciPy bounds; install the matching project requirements in the environment used by the campaign. The available source report recorded Python dependencies `safetensors 0.8.0`, `huggingface-hub 1.31.0`, `PyYAML 6.0.3`, and `einops 0.8.2`; `omegaconf`, MONAI, and `nnunetv2` were absent at that inspection. The current loaders directly require `torch`, `safetensors`, `huggingface_hub`, `PyYAML`, `omegaconf`, and the official CineMA package. Install any missing application dependencies in the existing environment, but do not run the CineMA `envs/environment.yml` wholesale or a generic dependency resolver that replaces PyTorch or CUDA.

The official upstream installation notes describe editable source installs (`pip install -e .`) and explicitly instruct installing PyTorch separately. This provisioning CLI deliberately avoids editable installs and bulk upstream dependency files: MedNeXt is imported from the declared `source_repo`, while CineMA is imported from its pinned checkout via `PYTHONPATH`. This keeps code revisions explicit and avoids a torch reinstall. The command checks/adds only missing non-torch Python distributions using `pip --no-deps`; if an import reports a missing transitive dependency, install that specific non-torch package explicitly in the active environment. Package version pinning beyond repository `requirements.txt` is not asserted by upstream and should be matched to the server's already validated runtime environment.

## Provision / verify

From the repository root:

```bash
# Read-only check; writes SOURCE_READY.json with READY or NOT_READY and hashes.
python scripts/pg_provision_sources.py

# Explicitly clone/check out pinned code and install missing non-torch deps.
python scripts/pg_provision_sources.py --install

# Explicitly fetch just the two allowlisted files to the normal HF cache.
python scripts/pg_provision_sources.py --download-cinema

# Full explicit provisioning, then strict local verification.
python scripts/pg_provision_sources.py --install --download-cinema
python scripts/pg_provision_sources.py
```

The default code paths are `external/MedNeXt` and `external/CineMA`. The first is exactly the existing config value `teacher.source_repo: "external/MedNeXt"` and resolves relative to the repository-root process working directory. The second must be added to `PYTHONPATH` for CineMA's Python package:

```bash
export PYTHONPATH="$PWD/external/CineMA${PYTHONPATH:+:$PYTHONPATH}"
```

`HF_HOME` may be set to a persistent cache location; otherwise Hugging Face uses its standard cache. `MEDNEXT_SOURCE_ROOT` is documented as an operator alias for the path but is not read by the current PG loader; keep `teacher.source_repo` at the relative path above. Do not put absolute `source_paths` into `cinema_init`: the loader explicitly rejects them in real mode and resolves the pinned Hub files using the standard cache.

## Config binding and blocker

The accepted `candidate_pg.v2` mapping permits an extra `cinema_init.hf_revision` field. The loader passes this value as Hugging Face `revision=` when resolving both exact filenames (`self_audit/candidate_pg/cinema_init.py::resolve_cinema_source`). The base config currently omits that field. Before any real S1 run, add `"hf_revision": "b1251ee50423bceeca84c080782fc3bc7756dea6"` under `cinema_init` and recompute the locked base/arm config hashes. This value comes from the official Hub model API response, independently verified above; do not substitute the GitHub source commit `c10daa1d93f0ea28d8b9ad9206b0f673d25805c1`. Do not silently run the current unpinned CineMA default (`main`). `source_paths` must remain omitted because the real loader only allows that key in explicit synthetic mode. The provisioner verifies the config's published Git blob OID and weight LFS SHA, checks both sizes, and records each local file path plus actual raw-byte SHA-256 in the top-level `artifacts` list (also under `sources.cinema_pretrained.artifacts`) for the queue to rehash. No raw config SHA-256 is assumed in advance.

The MedNeXt config remains unchanged: `teacher.source_repo` is `external/MedNeXt`, `teacher.source_commit` is the pinned SHA, and `teacher.weights` / `teacher.weights_sha256` remain null because S2 is initialized randomly by contract. Do not point it at a finetuned checkpoint.

## Receipt contract and evidence limits

`SOURCE_READY.json` uses schema `self_audit.candidate_pg.source_setup.v1`. Its top-level status is `READY` only when both clean exact-revision Git checkouts and both expected CineMA file metadata checks pass; otherwise it is `NOT_READY`. The receipt records origins, separate GitHub and Hub revisions, file paths and raw-byte SHA-256 values, published source OIDs, sizes, and the allowed config bindings. A `READY` receipt means source availability and provenance checks passed. It does not establish S1 loading compatibility, GPU memory, dataset readiness, or segmentation performance. This follow-up queried Hub/GitHub metadata only; no weight bytes were downloaded, and there was no SSH, GPU, or training action.

## Upstream references

- [MedNeXt upstream installation and v1 factory](https://github.com/MIC-DKFZ/MedNeXt)
- [CineMA upstream installation and pretrained model notes](https://github.com/mathpluscode/CineMA)
- [Verified CineMA source commit](https://github.com/mathpluscode/CineMA/commit/c10daa1d93f0ea28d8b9ad9206b0f673d25805c1)
- [Hugging Face CineMA model API](https://huggingface.co/api/models/mathpluscode/CineMA)
- [Hugging Face CineMA file metadata at the pinned revision](https://huggingface.co/api/models/mathpluscode/CineMA/tree/b1251ee50423bceeca84c080782fc3bc7756dea6/pretrained?expand=true)
- [Pinned CineMA source inspection](../pg-k8-20261116/CINEMA_SOURCE.md)
