#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y build-essential cmake git python3 python3-venv python3-pip unzip

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${RKNN_SDK_DIR:-$ROOT/../third_party/rknn-toolkit2-master}"
VENV="${RKNN_VENV:-$ROOT/.venv-rknn232}"
WHEEL="$SDK/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

if [[ ! -f "$WHEEL" ]]; then
  echo "RKNN wheel not found: $WHEEL" >&2
  exit 1
fi

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install "$WHEEL" onnx numpy
"$VENV/bin/python" - <<'PY'
from rknn.api import RKNN
print("RKNN-Toolkit2 import succeeded:", RKNN)
PY
