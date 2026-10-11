#!/usr/bin/env bash
# Isolated, explicit PyTorch + CineMA source installation. Never edits old envs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV="${VENV:-$REPO/.venv-stable-guidance}"
CINEMA_SOURCE="${CINEMA_SOURCE:-$REPO/.external/CineMA-c10daa1}"
TORCH_VARIANT="${TORCH_VARIANT:-cu124}"
CINEMA_COMMIT="c10daa1d93f0ea28d8b9ad9206b0f673d25805c1"
case "$TORCH_VARIANT" in cpu|cu124) ;; *) echo 'Use TORCH_VARIANT=cpu or cu124 (no automatic environment mutation)' >&2; exit 1;; esac
command -v "$PYTHON_BIN" >/dev/null || { echo "Missing $PYTHON_BIN; install Python 3.11 first" >&2; exit 1; }
if [[ -e "$VENV" ]]; then echo "Refusing to overwrite existing environment: $VENV" >&2; exit 1; fi
"$PYTHON_BIN" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade 'pip>=24,<26'
"$PY" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT"
if [[ ! -e "$CINEMA_SOURCE" ]]; then
  mkdir -p "$(dirname "$CINEMA_SOURCE")"
  git clone https://github.com/mathpluscode/CineMA.git "$CINEMA_SOURCE"
fi
[[ -z "$(git -C "$CINEMA_SOURCE" status --porcelain)" ]] || { echo 'CineMA source contains changes; refusing checkout' >&2; exit 1; }
git -C "$CINEMA_SOURCE" checkout --detach "$CINEMA_COMMIT"
# Editable source avoids incomplete subpackage discovery in upstream wheel config.
"$PY" -m pip install -e "$CINEMA_SOURCE" 'huggingface_hub==0.34.4' 'pytest==8.3.5'
"$PY" -m pip check
"$PY" - <<'PY'
import cinema.mae.mae, nibabel, torch, torchvision
print('CineMA import:', cinema.mae.mae.__file__)
print('nibabel:', nibabel.__version__)
print('torch/torchvision:', torch.__version__, torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
PY
"$PY" -m pip freeze > "$VENV/resolved-requirements.txt"
echo "Activate with: source '$VENV/bin/activate'"
