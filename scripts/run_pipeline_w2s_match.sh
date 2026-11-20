#!/usr/bin/env bash
# W2S-Match pipeline (single seed 42): W -> saturation -> CB -> S2 (or S2-arb) -> S0 -> S2-arb.
# Resumable: every step writes $RUN/markers/<STEP>.DONE and is skipped on rerun; training steps resume from checkpoints.
# Env: SELF_AUDIT_ROOT (required), STEPS (default: all), DEVICE (default cuda), PART_A_RUN (S1 source run).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SELF_AUDIT_ROOT:?set SELF_AUDIT_ROOT}"
R=$SELF_AUDIT_ROOT
RUN=$R/runs/w2s-match-20261119
A=${PART_A_RUN:-$R/runs/cbkd-sa-a4-20261119}
PREP=${PREP:-$R/prepared/ACDC/cbkd-sa-a4-20261119}
V20=${V20:-$R/prepared/ACDC/w2s-val20-20261120}
DEVICE=${DEVICE:-cuda}
CFG=$RUN/configs
STEPS=${STEPS:-"CONFIG LINK W WFINAL SAT CONFIG2 LINKW CB S2 S2FINAL S0 S0FINAL S2ARB S2ARBFINAL"}
mkdir -p "$RUN/markers"

want() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
step() { # step NAME cmd...
  local name=$1; shift
  want "$name" || return 0
  if [ -f "$RUN/markers/$name.DONE" ]; then echo "[w2s] $name: DONE (skip)"; return 0; fi
  echo "[w2s] $name: start $(date -Is)"
  "$@"
  date -Is > "$RUN/markers/$name.DONE"
  echo "[w2s] $name: done $(date -Is)"
}
tau_v() { python -c "import json;print(json.load(open('$RUN/W/SATURATION.json'))['tau_v'])"; }
cfgs() { python scripts/w2s_make_configs.py --prepared "$PREP" --val20-dir "$V20" --out-dir "$CFG" --tau-v "$1"; }
campaign() { python scripts/pg_campaign.py --device "$DEVICE" "$@"; }
S1SHA=${S1_SHA256:-c8195fb4a79ee665b169da36a492c99192eb33a1962c42079404107c1d2a01bf}

step CONFIG cfgs 0.99
step LINK python scripts/w2s_stage_link.py --s1-run "$A" --s1-config "$A/A4_SA_CONFIG.json" --w2s-config "$CFG/w2s_S2.json" \
  --dest "$RUN/S0" "$RUN/S2arb" "$RUN/S2" --expect-sha256 "$S1SHA"
step W campaign --config "$CFG/w2s_W.json" --out "$RUN/W" --stage S2 $( [ -f "$RUN/W/S2/teacher_latest.pt" ] && echo --resume )
step WFINAL python scripts/w2s_eval_teacher.py --config "$CFG/w2s_W.json" --run "$RUN/W" --device "$DEVICE"
step SAT python scripts/w2s_saturation.py --config "$CFG/w2s_W.json" --run "$RUN/W" --device "$DEVICE"
config2() { cfgs "$(tau_v)"; }
step CONFIG2 config2
step LINKW python scripts/w2s_stage_link.py --s1-run "$A" --s1-config "$A/A4_SA_CONFIG.json" --w2s-config "$CFG/w2s_S2.json" \
  --dest "$RUN/S0" "$RUN/S2arb" "$RUN/S2" --w-run "$RUN/W" --teacher-dest "$RUN/S2arb" "$RUN/S2" --expect-sha256 "$S1SHA"

if want CB || want S2; then
  step CB campaign --config "$CFG/w2s_S2.json" --out "$RUN/S2" --stage S3 --resume
  if python scripts/w2s_cb_check.py "$RUN/S2"; then MAIN=S2; else MAIN=S2ARB; echo "[w2s] CB fallback rule fired: S2-arb is the main arm"; fi
else MAIN=S2ARB; fi
if [ "${MAIN:-S2ARB}" = S2 ]; then
  step S2 campaign --config "$CFG/w2s_S2.json" --out "$RUN/S2" --stage arms --arm A4
  step S2FINAL python scripts/w2s_finalize.py --config "$CFG/w2s_S2.json" --run "$RUN/S2" --arm A4 --name S2 --device "$DEVICE"
fi
step S0 campaign --config "$CFG/w2s_S0.json" --out "$RUN/S0" --stage arms --arm A2
step S0FINAL python scripts/w2s_finalize.py --config "$CFG/w2s_S0.json" --run "$RUN/S0" --arm A2 --name S0 --device "$DEVICE"
step S2ARB campaign --config "$CFG/w2s_S2arb.json" --out "$RUN/S2arb" --stage arms --arm A2
step S2ARBFINAL python scripts/w2s_finalize.py --config "$CFG/w2s_S2arb.json" --run "$RUN/S2arb" --arm A2 --name S2arb --device "$DEVICE"
echo "[w2s] pipeline finished $(date -Is)"
