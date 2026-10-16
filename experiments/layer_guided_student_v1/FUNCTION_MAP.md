# Function map

Reference: `reports/layer_level_upgrade_20261015/reference/` at
`f4b0fe7d9e49ee398354e20ff0981c818840f5d4`.
Manifest SHA-256:
`3f442aa35872bea5db00eb8880c002b1049a1e53da2a88accfbf8e6e5cec2576`.
All 19 archived files are verified before execution. The following port paths
are relative to this directory; the corresponding reference paths are identical
except for the archived test suffix.

| Port path | Preserved symbols and behavior |
| --- | --- |
| `layer_upgrade/__init__.py` | All public exports; same names and imports |
| `layer_upgrade/layers.py` | `CenterDifferenceInput`, `block_mean`, `block_lift`, `zero_mean_detail`, `compose_logits`; center normalization, zero-initialized difference convolution, exact disjoint-cell D/U decomposition |
| `layer_upgrade/model.py` | `OfficialConvNeXtStages`, `ResidualUnit`, `LayerGuidedStudent`, `build_random_convnext_demo`; stage interfaces, channels, parameters, initialization, padding, masks and output keys |
| `layer_upgrade/losses.py` | `evidence_distribution`, `soft_ce`, `set_loss`, `masked_feature_loss`; detached targets, support and denominators, finite/shape failures |
| `layer_upgrade/guarded.py` | `project_direction`, `guarded_sgd_step` and private helpers; float64 active-set solve, selected-parameter additive trial, finite loss checks, rejection/exception rollback |
| `layer_upgrade/refine.py` | `FrozenImageEnergy`, `image_grid_graph`, `refine_distribution`; frozen summed energy, edges, gradient, Lipschitz bound, mirror step and failure conditions |
| `tests/test_operators.py` | Restored verbatim from `tests/test_operators.py.txt`; original 47 cases |
| `verify_full_model.py`, `probe_report.py`, `pytest.ini` | Verbatim runtime probes/config; run in a fresh output directory |

These files have no behavioral or formatting deviations. `verify_port.py`
enforces byte equality. `tests/conftest.py` materializes the verified archive
under a separate package name, `oracle_layer_upgrade`; the differential tests
assert distinct classes and module identities, copy identical state dictionaries,
and compare outputs, input/parameter gradients, receipts and rollback effects.

| New integration path | Responsibility and boundary |
| --- | --- |
| `integration/artifacts.py` | Explicit recipe, local approved DINO constructor, incompatible checkpoint schema and strict loader; reference default buffers fixed |
| `integration/artifacts.py:ConventionalHeadStudent` | Separate control ID with bilinear coarse plus unprojected bounded detail; same parameters, no direct coarse/detail gradient separation; discarded reference composition adds overhead |
| `integration/geometry.py` | Per-cine image-only normalization, adjacent spatial slices, source registration, pixel-cell overlap transport of mass/support, coarse loss and actual masked-view feature learning |
| `integration/bank.py` | Read-only frozen historical TRAIN bank adapter, caller-pinned manifest and reviewed producer hashes; returns image/features only |
| `integration/steps.py` | Locked witness metadata, isolated additive decoder substep, detached image proposal with semantic admission false |
| `integration/cli.py` | Approved local backbone forward/backward preflight on manufactured input; zero optimizer steps |
| `verify_port.py` | Reference hash gate, protected-path diff gate, isolated test/probe execution and machine-readable report |

The new modules do not register a new default in the historical student loader.
The old tiny checkpoint and its baseline provenance remain separate.
