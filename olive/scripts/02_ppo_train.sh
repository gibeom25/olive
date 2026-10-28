#!/usr/bin/env bash
# 2) rollout 을 통한 RL 가중치 학습 (PPO).
#    frozen backbone + (BC 로 warm-start 된) Actor/Critic -> z-space PPO.
#    같은 SUITE/ABLATION 의 after_bc.pt 가 있으면 자동으로 --init-ckpt 로 사용.
#    없거나 FRESH=1 이면 BC 없이 warm-start init 에서 바로 PPO.
#
# 사용:
#   SUITE=libero_spatial ABLATION=main ./02_ppo_train.sh
#   FRESH=1 ./02_ppo_train.sh                    # BC 무시, 순수 PPO
#   PPO_TOTAL_STEPS=20000 EVAL_EVERY=5 ./02_ppo_train.sh
#   ENTROPY_COEF=0.0005 ./02_ppo_train.sh        # z-space entropy 스케일 조정
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_env.sh

_summary
fetch_vla

BC_CKPT="$RUN_DIR/ckpt/after_bc.pt"
INIT_ARGS=()
if [[ "${FRESH:-0}" == "1" ]]; then
  echo "[02] FRESH=1 -> BC init 무시, 순수 PPO"
  INIT_ARGS=(--no-bc --phase ppo)
elif [[ -f "$BC_CKPT" ]]; then
  echo "[02] BC init 사용: $BC_CKPT"
  INIT_ARGS=(--no-bc --phase ppo --init-ckpt "$BC_CKPT")
else
  echo "[02] after_bc.pt 없음 -> BC 생략하고 PPO (warm-start init 에서 시작)"
  echo "     BC 부터 하려면: ./01_bc_warmup.sh 먼저"
  INIT_ARGS=(--no-bc --phase ppo)
fi

# 실행 중인 run_all 레인의 노브를 바꿀 때: result/.ppo_override.env 에 VAR=값 (이 단계 시작 시점에 읽음)
OVERRIDE_ENV="$OLIVE_ROOT/result/.ppo_override.env"
[[ -f "$OVERRIDE_ENV" ]] && { echo "[02] override: $OVERRIDE_ENV"; source "$OVERRIDE_ENV"; }

EXTRA=()
[[ -n "${PPO_TOTAL_STEPS:-}" ]] && EXTRA+=(--set "ppo.total_steps=${PPO_TOTAL_STEPS}")
[[ -n "${PPO_BUFFER:-}" ]]      && EXTRA+=(--set "ppo.buffer_size=${PPO_BUFFER}")
[[ -n "${ENTROPY_COEF:-}" ]]    && EXTRA+=(--set "ppo.entropy_coef=${ENTROPY_COEF}")
[[ -n "${LR_START:-}" ]]        && EXTRA+=(--set "ppo.lr_start=${LR_START}")
[[ -n "${EVAL_EVERY:-}" ]]      && EXTRA+=(--set "eval_every=${EVAL_EVERY}")
[[ -n "${EVAL_EPISODES:-}" ]]   && EXTRA+=(--set "eval_episodes_per_task=${EVAL_EPISODES}")
[[ -n "${CKPT_EVERY:-}" ]]      && EXTRA+=(--set "ckpt_every=${CKPT_EVERY}")
[[ "${NO_WANDB:-0}" == "1" ]]   && EXTRA+=(--no-wandb)

LOG="$RUN_DIR/logs/02_ppo_$(date +%Y%m%d_%H%M%S).log"
echo "[02] PPO -> $LOG   (detach 하려면 tmux 안에서 실행)"
python "$OLIVE_SRC/train.py" \
  --suite "$SUITE" \
  --pretrained_checkpoint "$VLA_CKPT" \
  --ablation "$ABLATION" \
  --seed "$SEED" \
  --run_name "$RUN_NAME" \
  "${INIT_ARGS[@]}" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG"

echo
echo "[02] done. best head: $RUN_DIR/ckpt/best.pt"
echo "     다음: ./03_eval.sh"
