#!/usr/bin/env bash
# Run SigLIP bg_only → full-scene matching in the background; append stdout/stderr to
# matching_logs.log at the repository root (override with MATCHING_LOG).
#
# Picks a Python that can import open_clip (prefers OCCAM venv). Preflight checks run in
# the foreground so failures are visible immediately and logged. The worker logs its
# exit code when it finishes (including immediate crashes).
#
# Pass the runtime with -e (venv directory) or -p (python binary); see --help.
set -euo pipefail

usage() {
  printf '%s\n' \
    "Run match_waterbirds_bg_only_to_full_siglip.py in the background; logs to matching_logs.log." \
    "" \
    "Usage: $(basename "$0") [-e DIR | -p PATH] [--exact-pixel] [--debug] [--verbose-timing] [--no-warp-progress] [--class 0|1]" \
    "  [--search-places|--match-to-original] [--places-dir DIR] [--apply-same-square] [--keep-mask-shape] [--drop-black] [--fg-only-root DIR]" \
    "  -e, --venv DIR     Virtualenv root (runs DIR/bin/python)" \
    "  -p, --python PATH  Python interpreter to use" \
    "  --exact-pixel      Forward to Python (pixel-then-SigLIP matching)" \
    "  --debug            Forward to Python (stop after 3 matches; copies under ./debug_match/)" \
    "  --verbose-timing   Forward to Python (stderr [timing] lines; meta['timings_s'] always in JSON)" \
    "  --no-warp-progress Forward to Python (disable global Places warp tqdm bar on stderr)" \
    "  --class N          Forward to Python; N is 0 or 1 (one coarse label only)" \
    "  --search-places    Forward to Python (match bg_only to warped Places365 pool)" \
    "  --match-to-original Forward to Python (match each FG+BG image to nearest Places pool)" \
    "  --places-dir DIR   Forward to Python (required with Places modes)" \
    "  --apply-same-square Forward to Python (Places: same black square as paired full before SigLIP)" \
    "  --keep-mask-shape  Forward to Python (Places: bird-shaped FG mask instead of covering square)" \
    "  --drop-black       Forward to Python (Places: splice out FG square before SigLIP)" \
    "  --fg-only-root DIR Forward to Python (Places FG layout; default <Waterbirds>/FG-Only)" \
    "  -h, --help         Show this help" \
    "" \
    "If -e / -p are omitted, uses PYTHON, OCCAM_PYTHON, or repo env heuristics." \
    "Env: EXACT_PIXEL=1 adds --exact-pixel; MATCHING_CLASS=0|1 adds --class (if not on CLI)." \
    "Requires package: open_clip_torch (pip install open_clip_torch)." >&2
}

EXPLICIT_PYTHON=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p | --python)
      [[ -n "${2:-}" ]] || {
        printf '%s\n' "ERROR: $1 requires a path argument" >&2
        exit 1
      }
      EXPLICIT_PYTHON="$2"
      shift 2
      ;;
    -e | --venv)
      [[ -n "${2:-}" ]] || {
        printf '%s\n' "ERROR: $1 requires a directory argument" >&2
        exit 1
      }
      EXPLICIT_PYTHON="$2/bin/python"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --exact-pixel)
      EXTRA_ARGS+=(--exact-pixel)
      shift
      ;;
    --debug)
      EXTRA_ARGS+=(--debug)
      shift
      ;;
    --verbose-timing)
      EXTRA_ARGS+=(--verbose-timing)
      shift
      ;;
    --no-warp-progress)
      EXTRA_ARGS+=(--no-warp-progress)
      shift
      ;;
    --class)
      [[ -n "${2:-}" ]] || {
        printf '%s\n' "ERROR: --class requires 0 or 1" >&2
        exit 1
      }
      case "$2" in
        0 | 1) ;;
        *)
          printf '%s\n' "ERROR: --class must be 0 or 1" >&2
          exit 1
          ;;
      esac
      EXTRA_ARGS+=(--class "$2")
      shift 2
      ;;
    --search-places)
      EXTRA_ARGS+=(--search-places)
      shift
      ;;
    --match-to-original)
      EXTRA_ARGS+=(--match-to-original)
      shift
      ;;
    --places-dir)
      [[ -n "${2:-}" ]] || {
        printf '%s\n' "ERROR: --places-dir requires a directory argument" >&2
        exit 1
      }
      EXTRA_ARGS+=(--places-dir "$2")
      shift 2
      ;;
    --apply-same-square)
      EXTRA_ARGS+=(--apply-same-square)
      shift
      ;;
    --keep-mask-shape)
      EXTRA_ARGS+=(--keep-mask-shape)
      shift
      ;;
    --drop-black)
      EXTRA_ARGS+=(--drop-black)
      shift
      ;;
    --fg-only-root)
      [[ -n "${2:-}" ]] || {
        printf '%s\n' "ERROR: --fg-only-root requires a directory argument" >&2
        exit 1
      }
      EXTRA_ARGS+=(--fg-only-root "$2")
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      printf '%s\n' "ERROR: unknown option: $1 (try --help)" >&2
      exit 1
      ;;
    *)
      printf '%s\n' "ERROR: unexpected argument: $1 (use -e or -p for the environment)" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$EXPLICIT_PYTHON" ]]; then
  export PYTHON="$EXPLICIT_PYTHON"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="${MATCHING_LOG:-$ROOT/matching_logs.log}"
WB_ROOT="${WATERBIRDS_ROOT:-data/datasets/Waterbirds}"
OUT_JSON="${MATCHING_OUTPUT:-data/datasets/Waterbirds/bg_only_to_full_siglip.json}"
PID_FILE="${MATCHING_PID_FILE:-$ROOT/matching_logs.pid}"

_have_extra() {
  local needle="$1"
  local x
  for x in "${EXTRA_ARGS[@]}"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}
if [[ "${EXACT_PIXEL:-0}" == "1" ]] && ! _have_extra --exact-pixel; then
  EXTRA_ARGS+=(--exact-pixel)
fi
if [[ -n "${MATCHING_CLASS:-}" ]] && ! _have_extra --class; then
  case "$MATCHING_CLASS" in
    0 | 1) EXTRA_ARGS+=(--class "$MATCHING_CLASS") ;;
    *)
      printf '%s\n' "ERROR: MATCHING_CLASS must be 0 or 1 (got ${MATCHING_CLASS})" >&2
      exit 1
      ;;
  esac
fi

log() {
  # Copy to log and to stderr so interactive runs are not "silent"
  printf '%s\n' "$*" | tee -a "$LOG" >&2
}

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s' "$PYTHON"
    return
  fi
  if [[ -n "${OCCAM_PYTHON:-}" ]]; then
    printf '%s' "$OCCAM_PYTHON"
    return
  fi
  local c
  for c in "$ROOT/envs/occam/bin/python" "$ROOT/.venv/bin/python" "$ROOT/venv/bin/python"; do
    if [[ -x "$c" ]]; then
      printf '%s' "$c"
      return
    fi
  done
  command -v python3
}

PYTHON="$(pick_python)"
if [[ "$PYTHON" != /* ]]; then
  PYTHON="$(command -v "$PYTHON" 2>/dev/null || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  log "ERROR: No usable Python interpreter (resolved to '${PYTHON:-empty}'). Use -e / -p or set PYTHON=..."
  exit 1
fi

{
  echo "======== $(date -Iseconds 2>/dev/null || date) ========"
  echo "Repo: $ROOT"
  echo "Interpreter: $PYTHON ($("$PYTHON" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo 'version?'))"
  echo "Command: $PYTHON scripts/match_waterbirds_bg_only_to_full_siglip.py \\"
  echo "  --waterbirds-root $WB_ROOT \\"
  echo "  --output $OUT_JSON ${EXTRA_ARGS[*]}"
} >>"$LOG"

log "Preflight: testing import open_clip + occam …"
export OCCAM_REPO_ROOT_FOR_PREFLIGHT="$ROOT"
# Mirror output to the terminal (not only matching_logs.log) so ImportError is visible here.
if ! "$PYTHON" -c "import os, sys; sys.path.insert(0, os.environ['OCCAM_REPO_ROOT_FOR_PREFLIGHT']); import open_clip; import occam.datasets.waterbirds_layout; print('open_clip OK, occam layout OK')" 2>&1 | tee -a "$LOG"; then
  log "ERROR: preflight import failed (traceback is above on stderr and in $LOG)."
  log "If the error is No module named 'open_clip', install OpenCLIP into this venv, e.g.:"
  log "  $PYTHON -m pip install open_clip_torch"
  log "Or use a venv that already has it (e.g. envs/occam if that differs from occam0804)."
  exit 1
fi
unset OCCAM_REPO_ROOT_FOR_PREFLIGHT

log "Preflight OK. Starting background worker…"

(
  set +e
  echo "---- worker start $(date -Iseconds 2>/dev/null || date) ----"
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/match_waterbirds_bg_only_to_full_siglip.py \
    --waterbirds-root "$WB_ROOT" \
    --output "$OUT_JSON" \
    "${EXTRA_ARGS[@]}"
  ec=$?
  echo "---- worker end $(date -Iseconds 2>/dev/null || date) exit_code=$ec ----"
  exit "$ec"
) >>"$LOG" 2>&1 &

_pid=$!
echo "PID: ${_pid}" >>"$LOG"
echo "$_pid" >"$PID_FILE"

log "Started background PID ${_pid} (saved to $PID_FILE). Tailing is: tail -f $LOG"

# If the child dies almost immediately, say so on stderr (still detailed in LOG)
sleep 0.7
if ! kill -0 "$_pid" 2>/dev/null; then
  log "WARNING: process ${_pid} is no longer running (crashed or exited within ~1s). See tail of $LOG"
  exit 1
fi

log "Process ${_pid} is still running."
