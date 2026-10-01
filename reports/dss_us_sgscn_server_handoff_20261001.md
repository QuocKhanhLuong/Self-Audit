# Native baseline server handoff

> Superseded for producer/evaluation commands by
> `dss_us_sgscn_native_fixes_20261001.md`: run_native.py now requires
> `--image-root` (image-only staging) and evaluators require an external seal receipt.

Prepared after successful publication of implementation commit
`4cf909cd3ef8664c6e77cd54239371910a76d92a` to
`origin/codex/dss-us-sgscn-native`. Remote SHA was verified with `git ls-remote`.

**Server execution status: BLOCKED_SERVER_TARGET.** The SSH target and repository
checkout path have not been supplied. No server pull, environment modification,
native experiment or ACDC run has been executed. Commands below are preparation,
not a server execution receipt. CPU/Linux environment remains unverified until
the server checks pass.

## Obtain the verified code without changing an active checkout

Run on the intended Linux server; supply the two absolute directory paths first.
Use an unused worktree path outside active jobs. Do not reset or stash their code.

~~~bash
export NATIVE_BASELINES_REPO=/absolute/path/to/existing/Self-Audit
export NATIVE_BASELINES_WORKTREE=/absolute/path/to/new/native-baselines-worktree
export NATIVE_BASELINES_SHA=4cf909cd3ef8664c6e77cd54239371910a76d92a

git -C "$NATIVE_BASELINES_REPO" fetch origin "$NATIVE_BASELINES_SHA"
test "$(git -C "$NATIVE_BASELINES_REPO" rev-parse FETCH_HEAD)" = "$NATIVE_BASELINES_SHA" || exit 1
git -C "$NATIVE_BASELINES_REPO" worktree add --detach "$NATIVE_BASELINES_WORKTREE" "$NATIVE_BASELINES_SHA"
cd "$NATIVE_BASELINES_WORKTREE" || exit 1
test "$(git rev-parse HEAD)" = "$NATIVE_BASELINES_SHA" || exit 1
~~~

If the remote branch has advanced, do not guess a replacement SHA: inspect and
verify the new commit's receipts first. The commit above contains all implementation
and local validation files; later documentation commits need no producer update.

## Prepare separate CPU verification environments

Python 3.12 is the locally verified interpreter. If unavailable, record
`BLOCKED_ENVIRONMENT` and resolve installation independently. These commands
install dependencies in new private venvs, without changing global packages.
They do not launch GPU jobs or native experiments. GPU deployment requires its
own wheel/environment receipt and repeat diagnostics.

~~~bash
python3.12 -m venv .runtime/native_sgscn
.runtime/native_sgscn/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.runtime/native_sgscn/bin/python -m pip install -r baseline/SGSCN/environment/requirements.lock
.runtime/native_sgscn/bin/python -m pip check

python3.12 -m venv .runtime/native_dss_us
.runtime/native_dss_us/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.runtime/native_dss_us/bin/python -m pip install -r baseline/DSS_US/environment/requirements.lock
.runtime/native_dss_us/bin/python -m pip check

.runtime/native_dss_us/bin/python scripts/pin_native_references.py
~~~

The last command pins read-only references and the licensed DINO backbone. It
does not execute the unlicensed DSS-US source or download pretrained weights.
Do not copy that source into the reimplementation.

## Verify the available native milestone

~~~bash
.runtime/native_sgscn/bin/python -m pytest tests/native_baselines baseline/SGSCN/tests -q -p no:cacheprovider --basetemp .runtime/server_native_sgscn_tests
.runtime/native_dss_us/bin/python -m pytest baseline/DSS_US/tests -q -p no:cacheprovider --basetemp .runtime/server_native_dss_tests
.runtime/native_sgscn/bin/python scripts/native_protocol_readiness.py
~~~

Local results: 29 SGSCN/common tests and 8 DSS-US tests passed. DSS-US's real
architecture test uses a random checkpoint fixture; it is not pretrained inference.
Native tests do not need GT or clinical data. Preserve seeds, actual environment,
raw hashes and repeat diagnostics; inherent documented nondeterminism alone does
not make faithful native reproduction incomplete.

For a permitted SGSCN official-code reference run only, once an image-only native
inventory has been verified and GT is not mounted:

~~~bash
.runtime/native_sgscn/bin/python baseline/SGSCN/scripts/run_native.py \
  --config baseline/SGSCN/config/native/ph2_official_reference.yaml \
  --images-manifest /absolute/image-only/ph2_inventory.json \
  --output /absolute/new-output/ph2_reference_seed1 \
  --seed 1 --device cpu --threads 2
.runtime/native_sgscn/bin/python scripts/verify_native_raw.py /absolute/new-output/ph2_reference_seed1
~~~

SYSU-US uses its own `sysu_us_official_reference.yaml` and inventory with
`dataset="SYSU-US"`; PH2 settings must not be substituted. Reference execution
is explicitly not exact paper reproduction while the paper/source gap is unresolved.
Use a second fresh output root for repeats, then `verify_native_raw.py FIRST --repeat SECOND`.

## Gates still in force

- All DSS-US CAMUS paper profiles: `BLOCKED_PROTOCOL`. Row recipes, native cohort,
  full orchestration/CRF and evaluator details remain unresolved; real backbone
  checkpoint also missing locally.
- SGSCN PH2/SYSU-US paper profiles: `BLOCKED_PROTOCOL`. Architecture/stopping
  equivalence, complete original evaluator and original SYSU sampled inventory
  remain unresolved. Official-code reference producer is executable.
- Native clinical inventories: `BLOCKED_DATA` until supplied and verified.
- `NATIVE_REPRODUCTION_STATUS`, `NATIVE_TRACK_A_STATUS` and
  `NATIVE_TRACK_B_STATUS` stay separate. Native Track A is `BLOCKED_ADAPTER`.
- Native GT-assisted evaluation must verify the raw seal before opening GT.
  Do not call partial metric primitives a completed paper evaluator.
- No ACDC adaptation/run begins here. Both future ACDC tracks must consume the
  same frozen raw maps; primary Track B is the unchanged shared
  `raw_id_majority_vote_v1` rule from the frozen plan.

Do not remove gates, substitute guessed scientific settings, or use evaluation
results to choose producer profiles. Resolve dependencies through recorded
paper/source evidence before promoting any paper-profile gate.
