#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_MODE="${INSTALL_MODE:-cpu}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 1
fi

run_or_warn() {
  if ! "$@"; then
    echo "[WARN] Command failed but can be skipped in constrained networks: $*"
    return 1
  fi
  return 0
}

echo "[1/5] Creating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[2/5] Upgrading pip/setuptools/wheel"
run_or_warn python -m pip install --upgrade pip setuptools wheel || true

if [[ "$INSTALL_MODE" == "cpu" ]]; then
  echo "[3/5] Installing CPU PyTorch"
  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
else
  echo "[3/5] Skipping explicit torch install (INSTALL_MODE=$INSTALL_MODE)"
fi

echo "[4/5] Installing project dependencies"
pip install -e .

echo "[5/5] Verifying CLI"
flux-imagenet --help >/dev/null

echo
echo "✅ Environment is ready."
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Then run: flux-imagenet --class-id 207 --output outputs/dog.png"
