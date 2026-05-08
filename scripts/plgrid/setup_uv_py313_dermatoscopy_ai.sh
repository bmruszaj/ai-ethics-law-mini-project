#!/bin/bash
# Konfiguracja środowiska uv + Python 3.13 na PLGrid (ACK Cyfronet) dla repozytorium treningowego
# dermatoscopy_ai — trening RF-DETR, stack COCO, zależności GPU.
# Oryginał w drugim repo: scripts/setup_uv_py313.sh w https://github.com/bmruszaj/dermatoscopy_ai
# (repo prywatne — wymaga dostępu od zespołu). Przed uruchomieniem ustaw SCRATCH lub popraw
# ROOT_DIR / PROJECT_DIR / UV_BIN_DEFAULT do swojego konta i ścieżek na klastrze.
set -euo pipefail

ROOT_DIR="${SCRATCH:-/net/tscratch/people/plgbmruszaj}"
PROJECT_DIR="$ROOT_DIR/dermatoscopy_ai"
VENV_DIR="$PROJECT_DIR/.venv"
UV_BIN_DEFAULT="/net/people/plgrid/plgbmruszaj/.local/bin/uv"

cd "$PROJECT_DIR"

module purge
module load GCCcore/14.3.0
module load Python/3.13.5

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [ -x "$UV_BIN_DEFAULT" ]; then
  UV_BIN="$UV_BIN_DEFAULT"
  export PATH="$(dirname "$UV_BIN_DEFAULT"):$PATH"
else
  echo "ERROR: uv not found. Install it first (e.g. pipx install uv)."
  exit 1
fi

export UV_CACHE_DIR="$ROOT_DIR/.cache/uv"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/pip"
export HF_HOME="$ROOT_DIR/.huggingface"
export TRANSFORMERS_CACHE="$ROOT_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export TORCH_HOME="$ROOT_DIR/.cache/torch"
export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/.cache/xdg"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv "$VENV_DIR" --python "$(which python3)"
fi

source "$VENV_DIR/bin/activate"
SYNC_EXTRAS=(--extra notebooks --extra xai --extra rfdetr-xai --extra llm-apis)
if [ -f "$PROJECT_DIR/uv.lock" ]; then
  "$UV_BIN" sync --frozen "${SYNC_EXTRAS[@]}"
else
  "$UV_BIN" lock
  "$UV_BIN" sync --frozen "${SYNC_EXTRAS[@]}"
fi

# RF-DETR extras may pull GUI OpenCV; replace with headless variant for cluster nodes.
"$UV_BIN" pip uninstall opencv-python || true
"$UV_BIN" pip install --reinstall --no-deps opencv-python-headless==4.10.0.84

# Some mirrors provide metadata-only NVIDIA wheels; force full runtime wheels from PyPI.
"$UV_BIN" pip uninstall nvidia-cudnn-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvshmem-cu12 || true
"$UV_BIN" pip install \
  nvidia-cudnn-cu12==9.19.0.56 \
  nvidia-cusparselt-cu12==0.7.1 \
  nvidia-nccl-cu12==2.28.9 \
  nvidia-nvshmem-cu12==3.4.5 \
  --index https://pypi.org/simple

SITE_PKGS="$VENV_DIR/lib/python3.13/site-packages"
for lib_dir in "$SITE_PKGS"/nvidia/*/lib; do
  if [ -d "$lib_dir" ]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done

python -c "import torch, cv2, sahi; print(f'torch={torch.__version__} cuda={torch.version.cuda}'); print(f'cv2={cv2.__version__}'); print(f'sahi={sahi.__version__}')"
python -V
echo "Setup complete. Venv: $VENV_DIR"
