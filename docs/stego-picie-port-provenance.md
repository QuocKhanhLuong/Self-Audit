# STEGO / PiCIE Shared-Benchmark Port Provenance

Implementation branch base: `main@817f2ba1427506732ce6bb30bc5980e955710533`.

Reference snapshots are source references only; neither snapshot is merged.

## Imported Original Source Files

| Destination | Source snapshot | Source path | Purpose |
| --- | --- | --- | --- |
| `baseline/STEGO/LICENSE` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/LICENSE` | Upstream license attribution. |
| `baseline/STEGO/README.md` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/README.md` | Upstream README attribution. |
| `baseline/STEGO/src/__init__.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/src/__init__.py` | Package marker for callable source. |
| `baseline/STEGO/src/modules.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/src/modules.py` | DINO/STEGO featurizer used by cardiac runner; constructor made device-neutral in this port. |
| `baseline/STEGO/src/utils.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/src/utils.py` | Support module imported by original STEGO modules. |
| `baseline/STEGO/src/dino/utils.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/src/dino/utils.py` | DINO helper dependency. |
| `baseline/STEGO/src/dino/vision_transformer.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/STEGO/src/dino/vision_transformer.py` | DINO backbone implementation. |
| `baseline/PICIE/LICENSE` | `baseline/stego-picie-clean@8ec3942` | `baseline/PICIE/LICENSE` | Upstream license attribution. |
| `baseline/PICIE/README.md` | `baseline/stego-picie-clean@8ec3942` | `baseline/PICIE/README.md` | Upstream README attribution. |
| `baseline/PICIE/commons.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/PICIE/commons.py` | Legacy support reference retained outside scientific path. |
| `baseline/PICIE/modules/backbone.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/PICIE/modules/backbone.py` | PiCIE ResNet backbone for raw-partition inference. |
| `baseline/PICIE/modules/fpn.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/PICIE/modules/fpn.py` | PiCIE FPN decoder for raw-partition inference. |
| `baseline/PICIE/utils.py` | `baseline/stego-picie-clean@8ec3942` | `baseline/PICIE/utils.py` | Legacy utility reference retained outside scientific path; scientific runner must not import it. |

## Recovered Wrapper References

The initial cardiac wrapper intent came from `baseline/stego-picie-recovered@9f9545d` under:

- `baseline/STEGO/src/cardiac_benchmark/`
- `baseline/PICIE/src/cardiac_benchmark/`
- `scripts/run_stego_scientific.py`
- `scripts/run_picie_scientific.py`
- `baseline/STEGO/tests/cardiac/`
- `baseline/PICIE/tests/cardiac/`

Those files are edited in this port to consume `shared_benchmark_manifest.v1`, use shared spatial decoding/grid resize, and use `shared_benchmark.artifacts.run_generation`.

## Scientific Import Boundary

Legacy research entry points may remain in the tree for attribution or future reference, but the scientific producer path is limited to:

- `scripts/run_stego_scientific.py`
- `scripts/run_picie_scientific.py`
- `baseline/STEGO/src/cardiac_benchmark/`
- `baseline/PICIE/src/cardiac_benchmark/`
- the minimal model source files listed above
- `src/shared_benchmark/`

The producer path must not import legacy data readers, mask readers, evaluation metrics, Hungarian remapping, or semantic class conversion code.
