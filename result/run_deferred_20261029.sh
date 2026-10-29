#!/usr/bin/env bash
# object / goal 는 우선순위 뒤로: spatial 끝나면 object, baseline 레인 끝나면 goal 시작.
# (동시 실행 레인을 3개 이하로 유지해 libero_10 이 너무 느려지지 않게)
set -uo pipefail
export PYTHONNOUSERSITE=1

ROOT=/home/gibeom_pilab/olive
LOGDIR=$ROOT/result/logs/full_20261029

# run_full_20261029.sh 와 같은 노브
export ABLATION=main SEED=7 RELEASE_VLA=0 RESUME=0
export PPO_TOTAL_STEPS=5000 EVAL_EVERY=10 EVAL_EPISODES=5 EVAL_FINAL_EPISODES=10
export BC_TARGETS=critic BC_FRAME_STRIDE=15 BC_EPOCHS=3
export VIDEOS_PER_TASK=1 NO_WANDB=1

launch() {
  SUITES=$1 nohup "$ROOT/olive/scripts/run_all_suites.sh" > "$LOGDIR/olive_$1.log" 2>&1 &
  echo "$(date +%T) launched $1 pid $!"
}

(
  until [[ -f $ROOT/result/olive-libero_spatial-main/.done ]]; do sleep 30; done
  launch libero_object
) &
(
  until grep -q "all requested suites done" "$LOGDIR/baseline.log" 2>/dev/null; do sleep 30; done
  launch libero_goal
) &
wait
echo DEFERRED_LAUNCHED
