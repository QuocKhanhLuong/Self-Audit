# Orchestration accounting

Astra alone set scope, locked experiment/selection rules, implemented the isolated
baseline and harness, independently checked source/papers, and owns the final decision.

| Task | Requested / actually observed | Outcome / root verification |
|---|---|---|
| Major source audit, Union first | `opencode/union-alpha`; TUI rendered Union Alpha Free / OpenCode Zen | FAILED: provider “Model is unavailable”, no output. Root reconstructed source, then assigned one narrow trainer trace to Luna. |
| Three-paper Union audit | Startup explicitly requested `opencode/union-alpha`; TUI unexpectedly rendered Gemma 4 E4B Local / LM Studio Local | FAILED and routing violation: API connection failed, no result used. Dispatch immediately fenced on discovery; terminal closed (`ptyKilled=true`). No work accepted from this model. |
| SmooSeg methods/code | Tool-requested `gpt-5.6-luna` | Completed report. Root read official DINO freeze/download, training/checkpoint callback and original TeX methods/Algorithm 1. |
| UnSegMedGAT methods/code | Tool-requested `gpt-5.6-luna` | Completed. Root independently read `cts` implementation and its pre-feature caller, complement-vs-GT selection and paper's per-image objective/settings. |
| Ferreira methods/code | Tool-requested `gpt-5.6-luna` | Completed. Root read primary full-text methods, supplement visual tuning, HED constructor/load provenance and code lineage. |
| One `_train_batch` trace (reuse Luna) | Tool-requested `gpt-5.6-luna` | Completed. Root directly inspected trainer/model/loss/bank and added independent real-image component instrumentation. |

No Luna planned the overall research, delegated, trained models or modified production
files. They owned only their named report and private source-download directories.
No unverified worker output determines the verdict. Underlying hidden provider model
identities are not independently known beyond platform request metadata/TUI observations.

Current run: `run_0e2b48d17d7f`. The user-requested check of old
`run_3383831eadae` failed because this coordinator terminal is bound to the new run.
Both failed Union tasks are marked failed. External-terminal worker-stop initially
returned stop_unknown; explicit terminal close then confirmed each PTY killed. These
were agent terminals only; no user training job was terminated.

Skills used: [Academic Research Suite](/Users/alvinluong/.codex/skills/academic-research-suite/SKILL.md)
and [Orca orchestration](/Users/alvinluong/.agents/skills/orchestration/SKILL.md).
The user explicitly authorizes this research/implementation and makes Astra the final
decision-maker; generic skill approval/role suggestions do not override that contract.

Access failures: `vast-gpu` SSH connection refused at ssh6.vast.ai:15819 (receipt saved).
No CUDA/GPU result is claimed. A local MPS device exists but was not used: the locked
CPU4-thread recipe is the only experiment device. Network/PDF access issues used
author-linked repositories, primary arXiv methods and Europe PMC XML fallbacks; no
secondary blog supplies a substantive method claim.
