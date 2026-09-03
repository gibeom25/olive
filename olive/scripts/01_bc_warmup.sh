#!/usr/bin/env bash
# 1) LIBERO demo 를 활용한 추가학습 (BC warm-up).
#    Actor(behavior cloning) + Critic(return-to-go 회귀) 를 데모에서 사전학습.
#    결과: $RUN_DIR/ckpt/after_bc.pt  (다음 단계 02 가 이걸 init 으로 씀)
#
# 사용:
#   SUITE=libero_spatial ABLATION=main ./01_bc_warmup.sh
#   BC_EPOCHS=5 BC_FRAME_STRIDE=3 SUITE=libero_10 ./01_bc_warmup.sh
#   BC_TARGETS=critic ./01_bc_warmup.sh          # critic 만
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_env.sh

_summary
fetch_vla

EXTRA=()
[[ -n "${BC_EPOCHS:-}" ]]       && EXTRA+=(--set "bc.epochs=${BC_EPOCHS}")
[[ -n "${BC_FRAME_STRIDE:-}" ]] && EXTRA+=(--set "bc.frame_stride=${BC_FRAME_STRIDE}")
[[ -n "${BC_TARGETS:-}" ]]      && EXTRA+=(--set "bc.targets=${BC_TARGETS}")
[[ -n "${BC_MAX_DEMOS:-}" ]]    && EXTRA+=(--set "bc.max_demos=${BC_MAX_DEMOS}")
[[ -n "${BC_LR:-}" ]]          && EXTRA+=(--set "bc.lr=${BC_LR}")
[[ -n "${BC_DEMO_GLOB:-}" ]]    && EXTRA+=(--set "bc.demo_glob=${BC_DEMO_GLOB}")

LOG="$RUN_DIR/logs/01_bc_$(date +%Y%m%d_%H%M%S).log"
echo "[01] BC warm-up -> $LOG"
python "$OLIVE_SRC/train.py" \
  --suite "$SUITE" \
  --pretrained_checkpoint "$VLA_CKPT" \
  --ablation "$ABLATION" \
  --seed "$SEED" \
  --run_name "$RUN_NAME" \
  --phase bc \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG"

echo
echo "[01] done. head 가중치: $RUN_DIR/ckpt/after_bc.pt"
echo "     다음: ./02_ppo_train.sh  (같은 SUITE/ABLATION 이면 자동으로 이 ckpt 사용)"
