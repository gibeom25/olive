#!/usr/bin/env bash
# 3) 가중치 고정 후 테스트.
#    학습된 head(.pt) 를 불러 deterministic(z=mu_z) 으로 각 task 를 rollout, suite SR 측정.
#    baseline_libero_eval.md 와 동일 조건(center_crop, 180° 전처리, suite max_steps).
#
# 사용:
#   SUITE=libero_spatial ABLATION=main ./03_eval.sh                # best.pt
#   HEAD_CKPT=~/olive/result/olive-libero_spatial-main/ckpt/it0050.pt ./03_eval.sh
#   EVAL_EPISODES=50 ./03_eval.sh
#   VIDEOS_PER_TASK=0 ./03_eval.sh     # rollout mp4 저장 끔 (기본 task 당 1개 → $RUN_DIR/videos)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./_env.sh

_summary
fetch_vla

# 기본: best.pt(PPO 결과). 없으면 after_bc.pt(BC 만) 로 폴백 — "BC 결과만 rollout test" 용.
if [[ -z "${HEAD_CKPT:-}" ]]; then
  if   [[ -f "$RUN_DIR/ckpt/best.pt" ]];     then HEAD_CKPT="$RUN_DIR/ckpt/best.pt"
  elif [[ -f "$RUN_DIR/ckpt/after_bc.pt" ]]; then HEAD_CKPT="$RUN_DIR/ckpt/after_bc.pt"
                                                  echo "[03] best.pt 없음 -> after_bc.pt (BC 결과) 로 평가"
  else echo "head ckpt 없음: $RUN_DIR/ckpt/{best,after_bc}.pt" >&2; exit 1
  fi
fi
[[ -f "$HEAD_CKPT" ]] || { echo "head ckpt 없음: $HEAD_CKPT" >&2; exit 1; }

# ablation 에서 z_dim / inject_basis 가 바뀌었을 수 있으니 config 에서 그대로 읽어 전달
Z_DIM="$(python - "$HEAD_CKPT" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = p.get("config", {})
print((c.get("net") or {}).get("z_dim", 256))
PY
)"
INJECT="$(python - "$HEAD_CKPT" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = p.get("config", {})
print((c.get("proj") or {}).get("inject_basis", "lm_head"))
PY
)"
echo "[03] head=$HEAD_CKPT  z_dim=$Z_DIM  inject_basis=$INJECT"

WB_ARGS=()
if [[ "${NO_WANDB:-0}" != "1" ]]; then
  WB_ARGS=(--wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY" --wandb_name "${RUN_NAME}-eval")
fi

LOG="$RUN_DIR/logs/03_eval_$(date +%Y%m%d_%H%M%S).log"
python "$OLIVE_SRC/evaluate.py" \
  --suite "$SUITE" \
  --vla "$VLA_CKPT" \
  --ckpt "$HEAD_CKPT" \
  --z_dim "$Z_DIM" \
  --inject_basis "$INJECT" \
  --episodes_per_task "${EVAL_EPISODES:-20}" \
  --device "${DEVICE:-cuda:0}" \
  --video_dir "$RUN_DIR/videos" --videos_per_task "${VIDEOS_PER_TASK:-1}" \
  --out_json "$RUN_DIR/final_eval.json" \
  "${WB_ARGS[@]}" \
  2>&1 | tee "$LOG"

echo
echo "[03] 결과 로그: $LOG"
echo "     VLA 체크포인트 정리하려면:  $MODEL_MGR release $CKPT_ALIAS"
