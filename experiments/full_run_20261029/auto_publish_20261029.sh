#!/usr/bin/env bash
# suite 가 끝날 때마다(끝나는 순서대로) 결과를 experiments/full_run_20261029/ 로 모아 커밋 + push.
# 세션과 무관하게 끝까지 돈다 (nohup).
set -uo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/gibeom_pilab/olive
EXP=$ROOT/experiments/full_run_20261029
BRANCH=exp/full-run-20261029
PY=/home/gibeom_pilab/anaconda3/envs/openvla/bin/python

publish() {
  local label="$1"
  echo "$(date '+%F %T') publish: $label"
  mkdir -p "$EXP"
  "$PY" -I "$ROOT/report/collect_results.py" --out "$EXP" || echo "collect_results failed"
  for rd in "$ROOT"/result/olive-*-main; do
    [[ -d "$rd" ]] || continue
    local d="$EXP/runs/$(basename "$rd")"
    mkdir -p "$d/logs"
    cp -f "$rd"/metrics.jsonl "$rd"/final_eval.json "$rd"/.done "$d/" 2>/dev/null
    cp -f "$rd"/logs/*.log "$d/logs/" 2>/dev/null
  done
  mkdir -p "$EXP/runs/baseline_rerun"
  cp -f "$ROOT"/result/baseline/logs/*/EVAL-*.txt "$EXP/runs/baseline_rerun/" 2>/dev/null
  cp -f "$ROOT"/result/run_all_*.md "$ROOT"/result/run_*_20261029.sh "$ROOT"/result/auto_publish_20261029.sh "$EXP/" 2>/dev/null
  cd "$ROOT" || return
  git add experiments olive report docs .gitignore 2>/dev/null
  if git diff --cached --quiet; then echo "nothing to commit"; return; fi
  git commit -q -F - <<EOF
results: $label (full run 2026-10-29)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  for i in 1 2 3; do git push -u origin "$BRANCH" && break; echo "push retry $i"; sleep 60; done
}

done_of() { [[ -f "$ROOT/result/olive-$1-main/.done" ]]; }

declare -A pushed=()
while :; do
  new=()
  for s in libero_spatial libero_10 libero_goal libero_object; do
    if done_of "$s" && [[ -z "${pushed[$s]:-}" ]]; then new+=("$s"); pushed[$s]=1; fi
  done
  (( ${#new[@]} )) && publish "${new[*]} (${#pushed[@]}/4 suites done)"
  (( ${#pushed[@]} == 4 )) && break
  sleep 60
done
echo AUTO_PUBLISH_DONE
