#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_libero_baseline.sh
#
# Runs the "Frozen OpenVLA (vanilla)" LIBERO baseline from the OLIVE paper:
# the official openvla-7b-finetuned-libero-<suite> checkpoints evaluated
# out-of-the-box on their matching LIBERO task suite.
#
# Order (default): spatial -> object -> goal -> long
# For each suite it: fetches the checkpoint (~15 GB) via olive-model.sh,
# runs experiments/robot/libero/run_libero_eval.py, then releases the
# checkpoint to free disk before the next one.
#
# PREREQUISITE: run  ~/olive/eval/setup_libero_eval_env.sh  once first.
#
# Config via env vars:
#   CONDA_ENV=openvla            conda env to use
#   SUITES="spatial object goal long"   subset / order of suites
#   NUM_TRIALS_PER_TASK=50      rollouts per task (50 -> 500 per suite = paper setting)
#   SEED=7
#   CENTER_CROP=True            keep True: checkpoints were trained w/ random-crop aug
#   KEEP_MODELS=0               1 = keep checkpoints on disk after each suite
#   MUJOCO_GL=egl               egl | glfw | osmesa
#   OPENVLA_ATTN=flash_attention_2   attn impl (fallbacks: sdpa | eager)
#
# Quick smoke test (few minutes instead of many hours):
#   NUM_TRIALS_PER_TASK=1 SUITES=spatial ~/olive/eval/run_libero_baseline.sh
# ---------------------------------------------------------------------------
set -eo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/anaconda3}"
CONDA_ENV="${CONDA_ENV:-openvla}"
SUITES="${SUITES:-spatial object goal long}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
CENTER_CROP="${CENTER_CROP:-True}"
KEEP_MODELS="${KEEP_MODELS:-0}"
USE_WANDB="${USE_WANDB:-true}"                       # true/1 to log to W&B, false/0 to skip
WANDB_PROJECT="${WANDB_PROJECT:-olive-libero-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-gibeom-real-1526-kyungpook-national-university}"

OLIVE_ROOT="$HOME/olive"
OPENVLA_DIR="$OLIVE_ROOT/openvla"
MODEL_MGR="$OLIVE_ROOT/models/olive-model.sh"
LIBERO_SRC="${LIBERO_SRC:-$HOME/LIBERO/LIBERO}"

# All artifacts for this experiment go here: logs, rollout videos, wandb files.
RESULT_DIR="${RESULT_DIR:-$OLIVE_ROOT/result/baseline}"
LOG_ROOT="$RESULT_DIR/logs"
mkdir -p "$LOG_ROOT" "$RESULT_DIR/rollouts" "$RESULT_DIR/wandb"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero_base}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export PYTHONPATH="$OPENVLA_DIR:$LIBERO_SRC${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OPENVLA_ATTN="${OPENVLA_ATTN:-flash_attention_2}"   # or: sdpa (no flash-attn) / eager
# torch>=2.6 flipped torch.load default to weights_only=True; LIBERO's
# get_task_init_states() loads pickled numpy -> restore the old behavior.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DIR="$RESULT_DIR/wandb"                       # local wandb run files
export LIBERO_ROLLOUT_DIR="$RESULT_DIR/rollouts"           # honored by patched save_rollout_video()

# suite alias -> LIBERO task_suite_name  (alias also = olive-model.sh alias)
declare -A TASK_SUITE=( [spatial]=libero_spatial [object]=libero_object [goal]=libero_goal [long]=libero_10 )

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

[[ -x "$MODEL_MGR" ]] || chmod +x "$MODEL_MGR"
[[ -f "$OPENVLA_DIR/experiments/robot/libero/run_libero_eval.py" ]] || {
  echo "cannot find run_libero_eval.py under $OPENVLA_DIR" >&2; exit 1; }
mkdir -p "$LOG_ROOT"

# Release the in-progress checkpoint on ANY exit: normal finish, eval crash,
# Ctrl-C, SIGTERM, or tmux kill-session (SIGHUP). Only SIGKILL / power loss can
# skip this -- for those, recover with:  ~/olive/models/olive-model.sh list
ACTIVE_SUITE=""
cleanup() {
  local ec=$?
  trap - EXIT INT TERM HUP
  if [[ "${KEEP_MODELS:-0}" != "1" && -n "$ACTIVE_SUITE" ]]; then
    echo ">>> cleanup: releasing checkpoint for '$ACTIVE_SUITE'"
    "$MODEL_MGR" release "$ACTIVE_SUITE" 2>/dev/null || true
  fi
  exit "$ec"
}
trap cleanup EXIT INT TERM HUP

# ---- preflight -------------------------------------------------------------
echo ">>> preflight: conda env=$CONDA_ENV  python=$(python --version 2>&1)"
python - <<'PY' || { echo; echo ">>> deps missing. Run:  ~/olive/eval/setup_libero_eval_env.sh"; exit 1; }
import importlib.util as u, sys
need = ["prismatic","draccus","tensorflow","torch","transformers",
        "libero","robosuite","mujoco","bddl","easydict","imageio","robomimic"]
miss = [m for m in need if u.find_spec(m) is None]
print("    missing:", miss or "none")
sys.exit(1 if miss else 0)
PY

python - <<'PY'
import torch
print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PY

for s in $SUITES; do
  [[ -n "${TASK_SUITE[$s]:-}" ]] || { echo "unknown suite '$s' (spatial|object|goal|long)" >&2; exit 2; }
done

# ---- run one suite -------------------------------------------------------
run_suite() {
  local alias="$1" ts="${TASK_SUITE[$1]}"
  local stamp logdir ckpt rc=0
  stamp="$(date +%Y%m%d_%H%M%S)"
  logdir="$LOG_ROOT/${alias}-${stamp}"
  mkdir -p "$logdir"

  echo
  echo "======================================================================"
  echo "  SUITE: $alias   ($ts)   trials/task=$NUM_TRIALS_PER_TASK   seed=$SEED"
  echo "  log dir: $logdir"
  echo "======================================================================"

  ACTIVE_SUITE="$alias"                  # from here on, cleanup() owns this checkpoint
  ckpt="$("$MODEL_MGR" path "$alias")"   # downloads if not already present
  echo ">>> checkpoint: $ckpt"

  local wandb_args=()
  case "${USE_WANDB,,}" in
    1|true|yes|on)
      wandb_args=( --use_wandb True --wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" )
      echo ">>> W&B: $WANDB_ENTITY/$WANDB_PROJECT"
      ;;
    *) echo ">>> W&B: disabled" ;;
  esac

  (
    cd "$OPENVLA_DIR"
    python experiments/robot/libero/run_libero_eval.py \
      --model_family openvla \
      --pretrained_checkpoint "$ckpt" \
      --task_suite_name "$ts" \
      --center_crop "$CENTER_CROP" \
      --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
      --seed "$SEED" \
      --local_log_dir "$logdir" \
      --run_id_note "baseline" \
      "${wandb_args[@]}" \
      2>&1 | tee "$logdir/console.log"
    exit "${PIPESTATUS[0]}"
  ) || rc=$?

  if [[ "$KEEP_MODELS" != "1" ]]; then
    "$MODEL_MGR" release "$alias" || true
  fi
  ACTIVE_SUITE=""                        # released (or intentionally kept)

  if [[ $rc -ne 0 ]]; then
    echo "!!! suite '$alias' exited with code $rc (see $logdir/console.log)"
  else
    echo ">>> suite '$alias' finished. results: $logdir/"
  fi
  return $rc
}

overall=0
for s in $SUITES; do
  run_suite "$s" || overall=1
done

echo
echo "======================================================================"
echo "  all requested suites done."
echo "  results:  $RESULT_DIR/   (logs/  rollouts/  wandb/)"
echo "======================================================================"
exit $overall
