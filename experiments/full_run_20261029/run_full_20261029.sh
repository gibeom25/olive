#!/usr/bin/env bash
# 2026-10-29 full run: OLIVE 4 suites (1 lane each, parallel) + frozen baseline (1 lane, sequential suites).
# VRAM: ~16GB x 5 lanes = ~80GB of 96GB.
set -uo pipefail
export PYTHONNOUSERSITE=1          # openvla env 를 ~/.local site-packages 와 격리

ROOT=/home/gibeom_pilab/olive
LOGDIR=$ROOT/result/logs/full_20261029
mkdir -p "$LOGDIR"

# OLIVE knobs (run_all_suites.sh -> 01/02/03)
export ABLATION=main SEED=7 RELEASE_VLA=0 RESUME=0
export PPO_TOTAL_STEPS=5000 EVAL_EVERY=10 EVAL_EPISODES=5 EVAL_FINAL_EPISODES=10
export BC_TARGETS=critic BC_FRAME_STRIDE=15 BC_EPOCHS=3
export VIDEOS_PER_TASK=1 NO_WANDB=${NO_WANDB:-1}

for s in libero_spatial libero_object libero_goal libero_10; do
  SUITES=$s nohup "$ROOT/olive/scripts/run_all_suites.sh" > "$LOGDIR/olive_$s.log" 2>&1 &
  echo "olive $s pid $!"
  sleep 60                         # 모델 로딩 겹침 완화
done

# baseline: 10 trials/task (= OLIVE final eval 100 ep/suite), long first (slowest)
KEEP_MODELS=1 NUM_TRIALS_PER_TASK=10 SUITES="long goal spatial object" USE_WANDB=false \
  nohup "$ROOT/eval/run_libero_baseline.sh" > "$LOGDIR/baseline.log" 2>&1 &
echo "baseline pid $!"
wait
echo FULL_RUN_DONE
