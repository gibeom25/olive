#!/usr/bin/env bash
# 4) wandb 확인.
#    olive-libero 프로젝트의 run 목록 + 핵심 지표(BC MAE, eval SR, best SR, 상태)를 표로.
#    필터: RUN_FILTER 로 이름 부분일치. WATCH=1 이면 30초마다 갱신.
#
# 사용:
#   ./04_wandb_report.sh
#   RUN_FILTER=libero_10 ./04_wandb_report.sh
#   WANDB_PROJECT=olive-libero WATCH=1 ./04_wandb_report.sh
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SUITE="${SUITE:-libero_spatial}"; ABLATION="${ABLATION:-main}"   # _env.sh 요구값(미사용)
source ./_env.sh >/dev/null 2>&1 || {
  # _env 가 conda/LIBERO 까지 잡으려다 실패해도 wandb 보고는 되도록 최소 활성화
  source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-openvla}"
}

_report() {
python - "$WANDB_PROJECT" "$WANDB_ENTITY" "${RUN_FILTER:-}" <<'PY'
import sys, datetime
import wandb
project, entity, flt = sys.argv[1], sys.argv[2], sys.argv[3]
api = wandb.Api()
runs = api.runs(f"{entity}/{project}", order="-created_at")
rows = []
for r in runs:
    if flt and flt not in r.name:
        continue
    s = r.summary
    rows.append((
        r.name[:40],
        r.state,
        f'{s.get("bc/val_actor_mae", float("nan")):.3f}' if s.get("bc/val_actor_mae") is not None else "-",
        f'{s.get("eval/total_sr", float("nan")):.3f}' if s.get("eval/total_sr") is not None else "-",
        f'{s.get("best_eval_sr", float("nan")):.3f}' if s.get("best_eval_sr") is not None else "-",
        str(s.get("global_step", "-")),
        str(r.created_at)[:16],
        r.url,
    ))
if not rows:
    print(f"(run 없음)  project=https://wandb.ai/{entity}/{project}")
else:
    print(f"project: https://wandb.ai/{entity}/{project}   ({len(rows)} runs"
          + (f", filter='{flt}'" if flt else "") + ")")
    print(f'{"name":<40} {"state":<9} {"bcMAE":>6} {"evalSR":>7} {"bestSR":>7} {"step":>7}  {"created":<16}')
    print("-" * 110)
    for n, st, bc, ev, be, gs, cr, url in rows:
        print(f'{n:<40} {st:<9} {bc:>6} {ev:>7} {be:>7} {gs:>7}  {cr:<16}')
    print()
    for n, *_ , url in rows[:8]:
        print(f"  {n:<40} {url}")
PY
}

if [[ "${WATCH:-0}" == "1" ]]; then
  while true; do clear; date; echo; _report; sleep 30; done
else
  _report
fi
