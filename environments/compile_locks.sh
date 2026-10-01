#!/usr/bin/env bash
# Resolve the canonical Self-Audit locks from requirements.in (Linux x86_64, CPython 3.10).
# torch 2.4.1 on PyPI is the CUDA 12.1 build; the CPU build comes from the PyTorch CPU index.
set -euo pipefail
cd "$(dirname "$0")/self-audit-canonical"
common=(--python-version 3.10 --python-platform x86_64-manylinux_2_28 --no-header --generate-hashes)
printf 'torch==2.4.1+cpu\ntorchvision==0.19.1+cpu\n' > torch-cpu.in
printf 'torch==2.4.1\ntorchvision==0.19.1\n' > torch-cu121.in
uv pip compile "${common[@]}" requirements.in torch-cpu.in \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --emit-index-url -o requirements-cpu.lock
uv pip compile "${common[@]}" requirements.in torch-cu121.in \
  --index-url https://pypi.org/simple --emit-index-url -o requirements-cu121.lock
