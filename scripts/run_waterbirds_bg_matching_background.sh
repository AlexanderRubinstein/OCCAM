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
    "Usage: $(basename "$0") [-e DIR | -p PATH]" \
    "  -e, --venv DIR     Virtualenv root (runs DIR/bin/python)" \
    "  -p, --python PATH  Python interpreter to use" \
    "  -h, --help         Show this help" \
    "" \
    "If -e / -p are omitted, uses PYTHON, OCCAM_PYTHON, or repo env heuristics." \
    "Requires package: open_clip_torch (pip install open_clip_torch)." >&2
}

EXPLICIT_PYTHON=""
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
  echo "  --output $OUT_JSON"
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
    --output "$OUT_JSON"
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
