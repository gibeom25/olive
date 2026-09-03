#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_libero_parallel.sh
#
# Runs the LIBERO baseline for several suites CONCURRENTLY inside one detached
# tmux session, auto-sizing how many run at once from free GPU memory
# (nvidia-smi). Each lane == one suite == one `run_libero_baseline.sh` process
# that fetches its checkpoint, evals, then releases it.
#
# Concurrency:
#   MAX_PARALLEL=0 (default)  -> auto:  floor((free_vram - GPU_RESERVE_MIB) / PER_MODEL_MIB)
#   MAX_PARALLEL=N            -> force N lanes
# Before starting each new lane the orchestrator re-checks nvidia-smi and waits
# until there is room, so it also adapts to other jobs on a shared GPU.
#
# Config (env vars):
#   SUITES="spatial object goal long"   suites / order
#   NUM_TRIALS_PER_TASK=20              rollouts per task (50 = paper setting, ~2.5x longer)
#   SEED=7
#   CENTER_CROP=True
#   MAX_PARALLEL=0                      0 = auto from VRAM
#   PER_MODEL_MIB=19000                 VRAM budget per OpenVLA-7B lane
#   GPU_RESERVE_MIB=6000               headroom left free on the GPU
#   HARD_CAP=4                          never exceed this many lanes
#   LAUNCH_STAGGER=120                  seconds to wait after starting a lane
#                                       (lets it grab VRAM before the next check)
#   POLL_SEC=20                        VRAM re-poll interval while waiting
#   SESSION=libero-eval                tmux session name
#   KEEP_MODELS=0                      1 = keep each ~15 GB checkpoint after its suite
#
# Usage:
#   ~/olive/eval/run_libero_parallel.sh
#   MAX_PARALLEL=3 NUM_TRIALS_PER_TASK=50 ~/olive/eval/run_libero_parallel.sh
#   tmux attach -t libero-eval        # watch it
#   tmux kill-session -t libero-eval  # stop everything
# ---------------------------------------------------------------------------
set -eo pipefail

EVAL_DIR="$HOME/olive/eval"
BASE="$EVAL_DIR/run_libero_baseline.sh"
RESULT_DIR="${RESULT_DIR:-$HOME/olive/result/baseline}"
LOG_DIR="$RESULT_DIR/logs"

SESSION="${SESSION:-libero-eval}"
SUITES="${SUITES:-spatial object goal long}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
PER_MODEL_MIB="${PER_MODEL_MIB:-19000}"
GPU_RESERVE_MIB="${GPU_RESERVE_MIB:-6000}"
HARD_CAP="${HARD_CAP:-4}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-120}"
POLL_SEC="${POLL_SEC:-20}"

export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-20}"
export SEED="${SEED:-7}"
export CENTER_CROP="${CENTER_CROP:-True}"
export CONDA_ENV="${CONDA_ENV:-openvla}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export KEEP_MODELS="${KEEP_MODELS:-0}"
export USE_WANDB="${USE_WANDB:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-olive-libero-baseline}"
export WANDB_ENTITY="${WANDB_ENTITY:-gibeom-real-1526-kyungpook-national-university}"
export RESULT_DIR

command -v tmux >/dev/null || { echo "tmux not installed" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
[[ -x "$BASE" ]] || chmod +x "$BASE"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists." >&2
  echo "  attach: tmux attach -t $SESSION      kill: tmux kill-session -t $SESSION" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ORCH="$LOG_DIR/orch-$STAMP.sh"
ORCH_LOG="$LOG_DIR/orch-$STAMP.log"

nsuites=$(wc -w <<<"$SUITES")
free_now=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU free now: ${free_now} MiB   per-lane budget: ${PER_MODEL_MIB} MiB   reserve: ${GPU_RESERVE_MIB} MiB"

if (( MAX_PARALLEL > 0 )); then
  cap=$MAX_PARALLEL
else
  cap=$(( (free_now - GPU_RESERVE_MIB) / PER_MODEL_MIB ))
  (( cap < 1 )) && cap=1
fi
(( cap > nsuites )) && cap=$nsuites
(( cap > HARD_CAP )) && cap=$HARD_CAP
echo "-> starting with up to ${cap} concurrent lane(s) for ${nsuites} suite(s): $SUITES"
echo "   NUM_TRIALS_PER_TASK=$NUM_TRIALS_PER_TASK  SEED=$SEED  CENTER_CROP=$CENTER_CROP"

# ---- generate the orchestrator that tmux will run ------------------------
cat >"$ORCH" <<EOF
#!/usr/bin/env bash
set -u
SUITES="$SUITES"
CAP=$cap
PER_MODEL_MIB=$PER_MODEL_MIB
GPU_RESERVE_MIB=$GPU_RESERVE_MIB
LAUNCH_STAGGER=$LAUNCH_STAGGER
POLL_SEC=$POLL_SEC
BASE="$BASE"
LOGDIR="$LOG_DIR"
STAMP="$STAMP"

# on stop (tmux kill-session -> HUP, Ctrl-C -> INT), signal the lanes so each
# run_libero_baseline.sh runs its own cleanup() and releases its checkpoint.
stop_lanes() {
  trap - EXIT INT TERM HUP
  local p
  for p in \$(jobs -p); do kill -TERM "\$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "[orch] stopped, lanes signalled for checkpoint cleanup"
  exit 143
}
trap stop_lanes INT TERM HUP

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

wait_for_slot() {
  while :; do
    local running free
    running=\$(jobs -rp | wc -l)
    free=\$(gpu_free)
    if (( running < CAP )) && (( free - GPU_RESERVE_MIB >= PER_MODEL_MIB )); then
      return 0
    fi
    echo "[orch] \$(date +%H:%M:%S) waiting: running=\${running}/\${CAP} free=\${free}MiB (need \$((PER_MODEL_MIB+GPU_RESERVE_MIB)))"
    sleep "\$POLL_SEC"
  done
}

echo "[orch] start \$(date)  cap=\$CAP  suites=\$SUITES"
declare -A RC=()
for s in \$SUITES; do
  wait_for_slot
  out="\$LOGDIR/${SESSION:-libero-eval}-\${s}-\${STAMP}.out"
  echo "[orch] \$(date +%H:%M:%S) launch \$s  -> \$out"
  ( SUITES="\$s" "\$BASE" >"\$out" 2>&1; echo "RC \$s \$?" >"\$LOGDIR/.rc-\${s}-\${STAMP}" ) &
  sleep "\$LAUNCH_STAGGER"
done

wait
echo
echo "[orch] all lanes finished \$(date)"
for s in \$SUITES; do
  rcf="\$LOGDIR/.rc-\${s}-\${STAMP}"
  rc=\$( [[ -f "\$rcf" ]] && awk '{print \$3}' "\$rcf" || echo "?" )
  line=\$(grep -h 'Current total success rate' "\$LOGDIR"/\${s}-*/EVAL-*.txt 2>/dev/null | tail -1)
  printf '[orch] %-8s rc=%s  %s\n' "\$s" "\$rc" "\${line:-<no result line>}"
done
echo "[orch] logs: \$LOGDIR"
echo "[orch] (this pane stays open; press a key to close)"
read -r -n1 -s || true
EOF
chmod +x "$ORCH"

tmux new-session -d -s "$SESSION" "bash '$ORCH' 2>&1 | tee '$ORCH_LOG'"
echo
echo "launched tmux session '$SESSION'  (orchestrator: $ORCH)"
echo "  watch:  tmux attach -t $SESSION       (detach: Ctrl-b then d)"
echo "  tail:   tail -f $ORCH_LOG"
echo "  results dir: $RESULT_DIR/   (logs/  rollouts/  wandb/)"
echo "  per-suite console: $LOG_DIR/${SESSION}-<suite>-${STAMP}.out"
echo "  stop:   tmux kill-session -t $SESSION"
