#!/usr/bin/env bash
# 공통 환경 설정. 각 스크립트가 `source` 한다. 직접 실행하지 않음.
set -eo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
CONDA_ENV="${CONDA_ENV:-openvla}"

OLIVE_ROOT="$HOME/olive"
OLIVE_SRC="$OLIVE_ROOT/olive"              # 코드 (network.py, train.py ...)
OPENVLA_DIR="$OLIVE_ROOT/openvla"
LIBERO_SRC="${LIBERO_SRC:-$HOME/LIBERO/LIBERO}"
MODEL_MGR="$OLIVE_ROOT/models/olive-model.sh"

# 실험 파라미터 (env 로 override)
SUITE="${SUITE:-libero_spatial}"
ABLATION="${ABLATION:-main}"
SEED="${SEED:-7}"
RUN_NAME="${RUN_NAME:-olive-${SUITE}-${ABLATION}}"
RUN_DIR="$OLIVE_ROOT/result/$RUN_NAME"
WANDB_PROJECT="${WANDB_PROJECT:-olive-libero}"
WANDB_ENTITY="${WANDB_ENTITY:-gibeom-real-1526-kyungpook-national-university}"

# suite -> olive-model.sh alias -> HF repo basename(로컬 폴더명)
case "$SUITE" in
  libero_spatial) CKPT_ALIAS=spatial ;;
  libero_object)  CKPT_ALIAS=object  ;;
  libero_goal)    CKPT_ALIAS=goal    ;;
  libero_10)      CKPT_ALIAS=long    ;;
  *) echo "unknown SUITE=$SUITE" >&2; exit 2 ;;
esac
VLA_CKPT="$OLIVE_ROOT/models/openvla-7b-finetuned-libero-${CKPT_ALIAS/long/10}"

# conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# python path / LIBERO / torch
export PYTHONPATH="$OLIVE_SRC:$OPENVLA_DIR:$LIBERO_SRC${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero_base}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1          # tee/파일로 redirect 시 로그가 실시간으로 보이도록
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export WANDB_DIR="$RUN_DIR"
export WANDB_SILENT="${WANDB_SILENT:-true}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/ckpt"

# VLA(7B) 체크포인트 준비: 없으면 받는다 (BC/PPO/eval 모두 필요하므로 자동 release 안 함)
fetch_vla() {
  if [[ -f "$VLA_CKPT/config.json" ]] && compgen -G "$VLA_CKPT/*.safetensors" >/dev/null; then
    echo "[env] VLA ckpt present: $VLA_CKPT"
  else
    echo "[env] fetching VLA ckpt ($CKPT_ALIAS) ..."
    "$MODEL_MGR" fetch "$CKPT_ALIAS"
  fi
}

_summary() {
  echo "------------------------------------------------------------"
  echo " SUITE=$SUITE  ABLATION=$ABLATION  SEED=$SEED"
  echo " RUN_NAME=$RUN_NAME"
  echo " RUN_DIR =$RUN_DIR"
  echo " VLA_CKPT=$VLA_CKPT"
  echo " conda=$CONDA_ENV  python=$(python -V 2>&1)"
  echo "------------------------------------------------------------"
}
