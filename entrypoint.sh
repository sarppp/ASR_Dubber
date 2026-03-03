#!/usr/bin/env bash
# entrypoint.sh — Pipeline container entry point
#
# Pass raw CLI flags (forwarded directly to run_pipeline.py):
#   docker run ... --help
#   docker run ... --target-lang fr --run-mode translate
#
# Or configure via environment variables (used when no flags are given):
#   TARGET_LANG   target language code            (required, e.g. fr)
#   LANGUAGE      source language code            (optional, auto-detected)
#   INPUT_DIR     folder containing input video   (default: /data/input)
#   OUTPUT_DIR    folder for final outputs        (default: /data/output)
#   RUN_MODE      full|transcribe|translate       (default: full)
#   PRECISION     fp32|fp16|bf16                  (default: bf16)

set -euo pipefail

# If the first arg looks like a flag, forward everything directly.
if [[ $# -gt 0 ]]; then
    exec /app/nemo/.venv/bin/python /app/run_pipeline.py "$@"
fi

# ── Environment-variable mode ─────────────────────────────────────────────────
INPUT_DIR="${INPUT_DIR:-/data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"
TARGET_LANG="${TARGET_LANG:-}"
LANGUAGE="${LANGUAGE:-}"
RUN_MODE="${RUN_MODE:-full}"
PRECISION="${PRECISION:-bf16}"

if [[ -z "$TARGET_LANG" ]]; then
    echo "[entrypoint] ERROR: TARGET_LANG is required (e.g. TARGET_LANG=fr)" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "[entrypoint] ERROR: INPUT_DIR not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo " ASR Pipeline"
echo "  Input dir  : $INPUT_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "  Target lang: $TARGET_LANG"
echo "  Source lang: ${LANGUAGE:-auto-detect}"
echo "  Run mode   : $RUN_MODE"
echo "============================================================"

LANG_FLAG=()
[[ -n "$LANGUAGE" ]] && LANG_FLAG=(--language "$LANGUAGE")

exec /app/nemo/.venv/bin/python /app/run_pipeline.py \
    --target-lang "$TARGET_LANG" \
    --input-dir   "$INPUT_DIR"  \
    --output-dir  "$OUTPUT_DIR" \
    --run-mode    "$RUN_MODE"   \
    --precision   "$PRECISION"  \
    "${LANG_FLAG[@]}"
