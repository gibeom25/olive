#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# olive-model.sh — ephemeral OpenVLA checkpoint manager
#
# Each OpenVLA-7B checkpoint is ~15 GB. Instead of keeping them on disk we
# fetch one on demand, use it, then release the space.
#
# Models live in individual folders next to this script:
#     ~/olive/models/<repo-basename>/
#
# Usage:
#   ./olive-model.sh fetch   <suite>              # download if not present
#   ./olive-model.sh release <suite>             # delete the local folder
#   ./olive-model.sh path    <suite>             # print local path (fetch first if needed)
#   ./olive-model.sh list                        # show which checkpoints are on disk
#   ./olive-model.sh purge-cache [<suite>]       # drop stale copies from ~/.cache/huggingface/hub
#   ./olive-model.sh run [--keep] <suite> -- <cmd...>
#         fetch <suite>, export OPENVLA_MODEL_PATH=<local path>, run <cmd>,
#         then release the folder automatically (unless --keep).
#
#   <suite> is one of:  base  spatial  object  goal  long   (10 == long)
# ---------------------------------------------------------------------------
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_BIN="${HF_BIN:-hf}"

declare -A REPOS=(
  [base]="openvla/openvla-7b"
  [spatial]="openvla/openvla-7b-finetuned-libero-spatial"
  [object]="openvla/openvla-7b-finetuned-libero-object"
  [goal]="openvla/openvla-7b-finetuned-libero-goal"
  [long]="openvla/openvla-7b-finetuned-libero-10"
)

die() { echo "olive-model: $*" >&2; exit 1; }

resolve_suite() {
  local s="${1:-}"
  [[ "$s" == "10" ]] && s="long"
  [[ -n "${REPOS[$s]:-}" ]] || die "unknown suite '${1:-}' (valid: base spatial object goal long | 10)"
  printf '%s' "$s"
}

repo_of()      { printf '%s' "${REPOS[$1]}"; }
local_dir_of() { printf '%s/%s' "$MODELS_DIR" "$(basename "${REPOS[$1]}")"; }

is_present() {
  local d="$1"
  [[ -d "$d" ]] && [[ -f "$d/config.json" ]] && \
    compgen -G "$d/*.safetensors" > /dev/null 2>&1
}

cmd_fetch() {
  local suite; suite="$(resolve_suite "${1:-}")"
  local repo dir; repo="$(repo_of "$suite")"; dir="$(local_dir_of "$suite")"
  if is_present "$dir"; then
    echo "olive-model: '$suite' already present at $dir"
    return 0
  fi
  echo "olive-model: fetching $repo -> $dir"
  mkdir -p "$dir"
  "$HF_BIN" download "$repo" --local-dir "$dir"
  is_present "$dir" || die "download finished but $dir looks incomplete"
  echo "olive-model: ready ($(du -sh "$dir" | cut -f1)) at $dir"
}

cmd_release() {
  local suite; suite="$(resolve_suite "${1:-}")"
  local dir; dir="$(local_dir_of "$suite")"
  # safety: never rm outside MODELS_DIR
  case "$dir" in
    "$MODELS_DIR"/*) ;;
    *) die "refusing to delete '$dir' (outside $MODELS_DIR)";;
  esac
  if [[ -d "$dir" ]]; then
    echo "olive-model: releasing $dir ($(du -sh "$dir" | cut -f1))"
    rm -rf "$dir"
  else
    echo "olive-model: nothing to release for '$suite'"
  fi
}

cmd_path() {
  local suite; suite="$(resolve_suite "${1:-}")"
  local dir; dir="$(local_dir_of "$suite")"
  is_present "$dir" || cmd_fetch "$suite" >&2
  printf '%s\n' "$dir"
}

cmd_list() {
  printf '%-9s %-46s %-8s %s\n' SUITE REPO SIZE PATH
  local suite dir
  for suite in base spatial object goal long; do
    dir="$(local_dir_of "$suite")"
    if is_present "$dir"; then
      printf '%-9s %-46s %-8s %s\n' "$suite" "$(repo_of "$suite")" "$(du -sh "$dir" | cut -f1)" "$dir"
    else
      printf '%-9s %-46s %-8s %s\n' "$suite" "$(repo_of "$suite")" "-" "(not on disk)"
    fi
  done
}

cmd_purge_cache() {
  local hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  local suites=()
  if [[ $# -gt 0 ]]; then
    suites=("$(resolve_suite "$1")")
  else
    suites=(base spatial object goal long)
  fi
  local suite d found=0
  for suite in "${suites[@]}"; do
    d="$hub/models--$(repo_of "$suite" | sed 's#/#--#g')"
    if [[ -d "$d" ]]; then
      found=1
      echo "olive-model: purging hub cache $d ($(du -sh "$d" | cut -f1))"
      rm -rf "$d"
    fi
  done
  [[ $found -eq 0 ]] && echo "olive-model: no matching hub-cache entries"
  return 0
}

cmd_run() {
  local keep=0
  [[ "${1:-}" == "--keep" ]] && { keep=1; shift; }
  local suite; suite="$(resolve_suite "${1:-}")"; shift || true
  [[ "${1:-}" == "--" ]] || die "run: expected '-- <command>' after suite"
  shift
  [[ $# -gt 0 ]] || die "run: no command given"

  local dir; dir="$(local_dir_of "$suite")"
  local fetched_now=0
  if ! is_present "$dir"; then
    cmd_fetch "$suite"
    fetched_now=1
  fi

  cleanup() {
    if [[ $keep -eq 0 && $fetched_now -eq 1 ]]; then
      cmd_release "$suite"
    elif [[ $keep -eq 1 ]]; then
      echo "olive-model: --keep set, leaving $dir in place"
    fi
  }
  trap cleanup EXIT

  echo "olive-model: OPENVLA_MODEL_PATH=$dir"
  OPENVLA_MODEL_PATH="$dir" "$@"
}

main() {
  local sub="${1:-}"; shift || true
  case "$sub" in
    fetch)        cmd_fetch "$@";;
    release)      cmd_release "$@";;
    path)         cmd_path "$@";;
    list)         cmd_list "$@";;
    purge-cache)  cmd_purge_cache "$@";;
    run)          cmd_run "$@";;
    ""|-h|--help|help)
      awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
      ;;
    *) die "unknown subcommand '$sub' (try --help)";;
  esac
}

main "$@"
