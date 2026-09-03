#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all_suites.sh
#
# 모든 LIBERO suite 에 대해  BC warm-up -> PPO -> eval  전 과정을 순차 실행.
# 각 suite 는 자기 SFT 체크포인트(openvla-7b-finetuned-libero-<suite>)에서 시작한다
# (_env.sh 의 suite->ckpt 매핑 + fetch_vla 가 자동 처리).
#
# - 단일 GPU 공유 환경 -> 순차 (병렬 X)
# - suite 하나 끝나면 그 VLA 체크포인트(15GB) release
# - suite 하나 실패해도 다음 suite 계속 (set -e 안 씀)
# - RESUME=1(기본): $RUN_DIR/.done 있는 suite 는 건너뜀
# - 요약을 ~/olive/result/run_all_<stamp>.md 에 계속 갱신
#
# 사용 (tmux 안에서 권장 — 수 시간~수십 시간):
#   cd ~/olive/olive/scripts
#   ./run_all_suites.sh
#   SUITES="libero_spatial libero_10" ./run_all_suites.sh
#   PPO_TOTAL_STEPS=50000 ./run_all_suites.sh          # 본 학습 (기본은 5000 검증)
#   RESUME=0 ./run_all_suites.sh                       # .done 무시하고 전부 재실행
#   DRY=1 ./run_all_suites.sh                          # 실행 안 하고 계획만 출력
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_MGR="$HOME/olive/models/olive-model.sh"

SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
ABLATION="${ABLATION:-main}"
SEED="${SEED:-7}"
RESUME="${RESUME:-1}"
RELEASE_VLA="${RELEASE_VLA:-1}"
DRY="${DRY:-0}"

# 단계별 노브 (하위 스크립트 01/02/03 가 읽는 env 로 전달)
export ABLATION SEED
export PPO_TOTAL_STEPS="${PPO_TOTAL_STEPS:-5000}"   # 첫 전체 실행은 짧게. 확정 후 50000.
export EVAL_EVERY="${EVAL_EVERY:-5}"
export EVAL_EPISODES="${EVAL_EPISODES:-5}"          # PPO iteration 내부 eval (기본 20 이면 ~30분/회)
export BC_FRAME_STRIDE="${BC_FRAME_STRIDE:-10}"     # 5 -> ~3.5h/suite, 10 -> 절반
export BC_EPOCHS="${BC_EPOCHS:-3}"
EVAL_FINAL_EPISODES="${EVAL_FINAL_EPISODES:-20}"    # suite 최종 eval (03_eval.sh)

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$HOME/olive/result/run_all_${STAMP}.md"
mkdir -p "$HOME/olive/result"

{
  echo "# OLIVE 전체 suite 실행  ($(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "- SUITES: \`$SUITES\`"
  echo "- ABLATION=$ABLATION  SEED=$SEED"
  echo "- PPO_TOTAL_STEPS=$PPO_TOTAL_STEPS  EVAL_EVERY=$EVAL_EVERY  EVAL_EPISODES=$EVAL_EPISODES(iter)/$EVAL_FINAL_EPISODES(final)"
  echo "- BC_FRAME_STRIDE=$BC_FRAME_STRIDE  BC_EPOCHS=$BC_EPOCHS"
  echo "- PPO config 기본값: entropy_per_dim=True, target_kl=0.05, lr_start=1e-4 (z-space 보정)"
  echo
  echo "| suite | BC (a/c mse) | PPO best SR | final eval SR | 소요 | 상태 |"
  echo "|---|---|---|---|---|---|"
} > "$SUMMARY"

echo "[run_all] 요약 파일: $SUMMARY"
[[ "$DRY" == "1" ]] && echo "[run_all] DRY 모드 — 실행 안 함"

overall_start=$(date +%s)

for SUITE in $SUITES; do
  export SUITE
  RUN_NAME="olive-${SUITE}-${ABLATION}"
  RUN_DIR="$HOME/olive/result/$RUN_NAME"
  case "$SUITE" in
    libero_spatial) ALIAS=spatial;; libero_object) ALIAS=object;;
    libero_goal) ALIAS=goal;;      libero_10) ALIAS=long;;
    *) echo "[run_all] unknown SUITE=$SUITE — skip"; continue;;
  esac

  echo
  echo "############################################################"
  echo "# SUITE $SUITE   $(date '+%H:%M:%S')   run_dir=$RUN_DIR"
  echo "############################################################"

  if [[ "$RESUME" == "1" && -f "$RUN_DIR/.done" ]]; then
    prev=$(cat "$RUN_DIR/.done" 2>/dev/null)
    echo "[run_all] $SUITE 이미 완료 (.done): $prev — 건너뜀"
    echo "| $SUITE | - | - | $prev | - | RESUME skip |" >> "$SUMMARY"
    continue
  fi

  if [[ "$DRY" == "1" ]]; then
    echo "[DRY] SUITE=$SUITE $HERE/01_bc_warmup.sh"
    echo "[DRY] SUITE=$SUITE $HERE/02_ppo_train.sh"
    echo "[DRY] SUITE=$SUITE EVAL_EPISODES=$EVAL_FINAL_EPISODES $HERE/03_eval.sh"
    echo "[DRY] $MODEL_MGR release $ALIAS"
    continue
  fi

  suite_start=$(date +%s)
  status="OK"

  echo "--- [$SUITE] 1) BC warm-up ---"
  if ! "$HERE/01_bc_warmup.sh"; then
    status="BC 실패"
    echo "[run_all] $SUITE BC 실패 — 다음 suite 로"
    echo "| $SUITE | 실패 | - | - | $(( ($(date +%s)-suite_start)/60 ))m | $status |" >> "$SUMMARY"
    [[ "$RELEASE_VLA" == "1" ]] && "$MODEL_MGR" release "$ALIAS" || true
    continue
  fi

  echo "--- [$SUITE] 2) PPO (after_bc.pt 자동 init) ---"
  if ! "$HERE/02_ppo_train.sh"; then
    status="PPO 실패(eval 는 시도)"
    echo "[run_all] $SUITE PPO 실패 — eval 은 계속 시도"
  fi

  echo "--- [$SUITE] 3) final eval (EVAL_EPISODES=$EVAL_FINAL_EPISODES) ---"
  if ! EVAL_EPISODES="$EVAL_FINAL_EPISODES" "$HERE/03_eval.sh"; then
    [[ "$status" == "OK" ]] && status="eval 실패"
    echo "[run_all] $SUITE eval 실패"
  fi

  # ── 요약 수집 ──
  bc_mse=$(grep -h "\[OLIVE\]\[bc\] done" "$RUN_DIR"/logs/01_bc_*.log 2>/dev/null | tail -1 \
             | grep -oE "'actor_loss': [0-9.]+|'critic_loss': [0-9.]+" | paste -sd' ' -)
  ppo_best=$(grep -h "best eval SR" "$RUN_DIR"/logs/02_ppo_*.log 2>/dev/null | tail -1 | grep -oE "[0-9.]+%")
  final_sr=$(grep -h "TOTAL SR" "$RUN_DIR"/logs/03_eval_*.log 2>/dev/null | tail -1 | grep -oE "[0-9.]+%" | head -1)
  mins=$(( ($(date +%s)-suite_start)/60 ))
  echo "| $SUITE | ${bc_mse:-?} | ${ppo_best:-?} | ${final_sr:-?} | ${mins}m | $status |" >> "$SUMMARY"

  echo "$status | final ${final_sr:-?} | ppo_best ${ppo_best:-?} | $(date '+%Y-%m-%d %H:%M')" > "$RUN_DIR/.done"

  if [[ "$RELEASE_VLA" == "1" ]]; then
    echo "[run_all] $SUITE VLA 체크포인트 release ($ALIAS)"
    "$MODEL_MGR" release "$ALIAS" || true
  fi
  echo "[run_all] $SUITE 완료 ($status)  ${mins}분"
done

echo
echo "============================================================"
echo " 전체 완료  $(( ($(date +%s)-overall_start)/60 ))분  ($(date '+%H:%M:%S'))"
echo " 요약: $SUMMARY"
echo "============================================================"
cat "$SUMMARY"
