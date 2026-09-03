#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_libero_eval_env.sh  —  one-time prep to run OpenVLA LIBERO evals
#
# The `openvla` conda env already has: torch 2.2.0, transformers 4.40.1,
# tensorflow 2.15, draccus, peft, timm, editable `prismatic`.
# It is MISSING the LIBERO sim stack (robosuite / mujoco / bddl / robomimic).
# This script adds that stack — WITH strict version pins.
#
# Why the pins matter: LIBERO's setup.py deps are unpinned enough that a plain
# `pip install robosuite` drags in numba->numpy 2.x and a too-new
# tensorflow-metadata (needs protobuf 5.x), which breaks tensorflow/prismatic
# in this env. The pins below keep numpy<2 and protobuf in tensorflow's range.
#
# LIBERO source is NOT pip-installed; it is used from ~/LIBERO/LIBERO via
# PYTHONPATH (same as ~/LIBERO/libero_env.sh). The run script sets PYTHONPATH +
# LIBERO_CONFIG_PATH for you.
#
# Usage:
#   ~/olive/eval/setup_libero_eval_env.sh              # install into env `openvla`
#   MODE=clone ~/olive/eval/setup_libero_eval_env.sh   # clone -> `olive-eval` first
#   CONDA_ENV=myenv ~/olive/eval/setup_libero_eval_env.sh
#   FORCE=1 ~/olive/eval/setup_libero_eval_env.sh      # reinstall even if imports already pass
# ---------------------------------------------------------------------------
set -eo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
CONDA_ENV="${CONDA_ENV:-openvla}"
CLONE_NAME="${CLONE_NAME:-olive-eval}"
MODE="${MODE:-inplace}"          # inplace | clone
FORCE="${FORCE:-0}"
LIBERO_SRC="${LIBERO_SRC:-$HOME/LIBERO/LIBERO}"

source "$CONDA_BASE/etc/profile.d/conda.sh"

if [[ "$MODE" == "clone" ]]; then
  if ! conda env list | grep -qE "^\s*$CLONE_NAME\s"; then
    echo ">>> cloning env '$CONDA_ENV' -> '$CLONE_NAME' (a few minutes, ~15 GB)"
    conda create -y --clone "$CONDA_ENV" -n "$CLONE_NAME"
  else
    echo ">>> env '$CLONE_NAME' already exists, reusing"
  fi
  CONDA_ENV="$CLONE_NAME"
fi

conda activate "$CONDA_ENV"
set -u
echo ">>> target conda env: $CONDA_ENV"
echo ">>> python: $(python --version 2>&1) @ $(which python)"
[[ -d "$LIBERO_SRC/libero/libero" ]] || { echo "LIBERO source not found at $LIBERO_SRC" >&2; exit 1; }

VERIFY_PY='
import importlib, sys
mods = ["numpy","torch","transformers","tensorflow","tensorflow_datasets","dlimp",
        "wandb","draccus","prismatic","prismatic.extern.hf.modeling_prismatic",
        "robosuite","mujoco","bddl","easydict","robomimic","imageio",
        "libero","libero.libero.benchmark","libero.libero.envs"]
bad=[]
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append((m, f"{type(e).__name__}: {e}"))
if bad:
    for m,e in bad: print(f"  FAIL {m}: {e}")
    sys.exit(1)
# torch must actually be able to run a kernel on this GPU (Blackwell sm_120 etc.)
import torch
try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
except Exception as e:
    print(f"  FAIL torch-cuda-kernel: {type(e).__name__}: {e}")
    sys.exit(1)
print("  all imports OK + torch CUDA kernel OK")
'

if [[ "$FORCE" != "1" ]]; then
  echo ">>> checking whether env is already set up ..."
  if PYTHONPATH="$LIBERO_SRC:${PYTHONPATH:-}" python -c "$VERIFY_PY" 2>/dev/null; then
    echo ">>> nothing to do. (use FORCE=1 to reinstall)"
    exit 0
  fi
  echo ">>> some imports fail -> installing"
fi

# --- pinned leaf deps -------------------------------------------------------
# numpy / numba / opencv / tensorflow-metadata / wandb pins are the ones that
# actually keep this env consistent; the rest match LIBERO's setup.py.
python -m pip install --no-input \
  "numpy==1.26.4" \
  "numba==0.59.1" \
  "opencv-python==4.6.0.66" \
  "robosuite==1.4.0" \
  "mujoco==3.2.3" \
  "bddl==1.0.1" \
  "easydict==1.9" \
  "future==0.18.2" \
  "hydra-core==1.2.0" \
  "gym==0.25.2" \
  "cloudpickle" \
  "thop==0.1.1.post2209072238" \
  "egl_probe" \
  "glfw" \
  "imageio[ffmpeg]" \
  "tensorflow-metadata==1.14.0" \
  "wandb==0.16.6"

# robomimic 0.2.0 would pull its own torch/torchvision -> no-deps, add leaves.
python -m pip install --no-input --no-deps "robomimic==0.2.0"
python -m pip install --no-input "h5py" "termcolor" "psutil" "tqdm" "tensorboardX"

# --- GPU / torch compatibility -------------------------------------------
# OpenVLA pins torch==2.2.0 (cu121), which has NO kernels for Blackwell
# (sm_120) GPUs like the RTX PRO 6000 -> "no kernel image is available".
# If the current torch can't target this GPU, move to the cu128 build and a
# matching prebuilt flash-attn wheel (torch2.7 / cxx11abiTRUE / cp310).
TORCH_CU128="${TORCH_CU128:-2.7.0}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl}"

need_torch_upgrade=$(python - <<'PY'
import sys
try:
    import torch
    caps = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    want = f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}" if torch.cuda.is_available() else ""
    print("no" if (want and want in caps) else "yes")
except Exception:
    print("yes")
PY
)
if [[ "$need_torch_upgrade" == "yes" ]]; then
  echo ">>> current torch cannot target this GPU -> installing torch ${TORCH_CU128}+cu128"
  python -m pip install --no-input \
    "torch==${TORCH_CU128}" "torchvision" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128
  python -m pip uninstall -y flash-attn 2>/dev/null || true
  echo ">>> installing prebuilt flash-attn for torch 2.7 / cu128"
  python -m pip install --no-input --no-build-isolation "$FLASH_ATTN_WHL" || {
    echo ">>> prebuilt flash-attn wheel failed; falling back to SDPA."
    echo ">>> (set OPENVLA_ATTN=sdpa when running, or build flash-attn from source)"
  }
else
  echo ">>> torch already targets this GPU ($(python -c 'import torch;print(torch.__version__)'))"
fi

echo
echo ">>> verifying imports (PYTHONPATH includes $LIBERO_SRC)"
PYTHONPATH="$LIBERO_SRC:${PYTHONPATH:-}" python -c "$VERIFY_PY" || {
  echo ">>> verification FAILED - see errors above"; exit 1; }

echo
echo ">>> done. env '$CONDA_ENV' is ready."
[[ "$MODE" == "clone" ]] && echo ">>> run evals with:  CONDA_ENV=$CONDA_ENV ~/olive/eval/run_libero_baseline.sh"
exit 0
