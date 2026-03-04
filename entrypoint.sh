#!/usr/bin/env bash
# entrypoint.sh — Pipeline container entry point
#
# Two modes:
#
# 1. CLI flag mode — any argument starting with '--' is forwarded directly:
#      docker compose run pipeline --target-lang fr --run-mode translate
#      docker compose run pipeline --help
#
# 2. Environment-variable mode — set vars in docker-compose.yml or -e flags:
#
#   Required:
#     TARGET_LANG       target language code                (e.g. fr, en, de)
#
#   Input / output:
#     INPUT_DIR         folder containing input video       (default: /data/input)
#     OUTPUT_DIR        folder for final outputs            (default: /data/output)
#     LANGUAGE          source language code                (default: auto-detect)
#
#   Run control:
#     RUN_MODE          full|transcribe|translate           (default: full)
#     SKIP_NEMO         1|true to skip NeMo step
#     SKIP_TRANSLATE    1|true to skip translation step
#     SKIP_DUB          1|true to skip dubbing step
#
#   NeMo / ASR:
#     PRECISION         fp32|fp16|bf16                      (default: bf16)
#     NEMO_MODEL        shortname or full HF model ID       (e.g. canary, parakeet-v3)
#     CHUNK_OVERRIDE    force chunk size in seconds         (e.g. 120)
#     RESERVE_GB        VRAM reserve for chunking           (default: 1.5)
#     SAFETY_FACTOR     VRAM safety multiplier              (default: 0.85)
#     TRIM              process only first N seconds        (e.g. 30)
#
#   Whisper:
#     WHISPER_MODEL     tiny|base|small|medium|large-v3|turbo (default: medium)
#
#   Translation:
#     TRANSLATE_MODEL   Ollama model tag                    (default: translategemma:4b)
#
#   Dubbing:
#     QWEN_MODE         clone|custom                        (default: clone)
#     NO_DEMUCS         1|true to skip demucs (faster, no bgm)

set -euo pipefail

# If any argument looks like a CLI flag, forward everything directly.
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
NEMO_MODEL="${NEMO_MODEL:-}"
CHUNK_OVERRIDE="${CHUNK_OVERRIDE:-}"
RESERVE_GB="${RESERVE_GB:-}"
SAFETY_FACTOR="${SAFETY_FACTOR:-}"
TRIM="${TRIM:-0}"

WHISPER_MODEL="${WHISPER_MODEL:-medium}"

QWEN_MODE="${QWEN_MODE:-clone}"
NO_DEMUCS="${NO_DEMUCS:-}"

SKIP_NEMO="${SKIP_NEMO:-}"
SKIP_TRANSLATE="${SKIP_TRANSLATE:-}"
SKIP_DUB="${SKIP_DUB:-}"

# ── Validation ────────────────────────────────────────────────────────────────
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
echo " ASR Pipeline (env-var mode)"
echo "  Input dir  : $INPUT_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "  Target lang: $TARGET_LANG"
echo "  Source lang: ${LANGUAGE:-auto-detect}"
echo "  Run mode   : $RUN_MODE"
echo "  Precision  : $PRECISION"
[[ -n "$NEMO_MODEL"     ]] && echo "  NeMo model : $NEMO_MODEL"
[[ -n "$CHUNK_OVERRIDE" ]] && echo "  Chunk size : ${CHUNK_OVERRIDE}s (override)"
[[ -n "$TRIM"           && "$TRIM" != "0" ]] && echo "  Trim       : first ${TRIM}s"
echo "============================================================"

# ── Build optional flag arrays ────────────────────────────────────────────────
LANG_FLAG=();       [[ -n "$LANGUAGE"       ]] && LANG_FLAG=(--language "$LANGUAGE")
NEMO_FLAG=();       [[ -n "$NEMO_MODEL"     ]] && NEMO_FLAG=(--nemo-model "$NEMO_MODEL")
CHUNK_FLAG=();      [[ -n "$CHUNK_OVERRIDE" ]] && CHUNK_FLAG=(--chunk-override "$CHUNK_OVERRIDE")
RESERVE_FLAG=();    [[ -n "$RESERVE_GB"     ]] && RESERVE_FLAG=(--reserve-gb "$RESERVE_GB")
SAFETY_FLAG=();     [[ -n "$SAFETY_FACTOR"  ]] && SAFETY_FLAG=(--safety-factor "$SAFETY_FACTOR")
TRIM_FLAG=();       [[ -n "$TRIM" && "$TRIM" != "0" ]] && TRIM_FLAG=(--trim "$TRIM")
WHISPER_FLAG=();    [[ -n "$WHISPER_MODEL"  ]] && WHISPER_FLAG=(--whisper-model "$WHISPER_MODEL")
QWEN_FLAG=();       [[ -n "$QWEN_MODE"      ]] && QWEN_FLAG=(--qwen-mode "$QWEN_MODE")
DEMUCS_FLAG=();     [[ "$NO_DEMUCS"  =~ ^(1|true|yes)$ ]] && DEMUCS_FLAG=(--no-demucs)
SKIP_NEMO_FLAG=();  [[ "$SKIP_NEMO"  =~ ^(1|true|yes)$ ]] && SKIP_NEMO_FLAG=(--skip-nemo)
SKIP_TRANS_FLAG=(); [[ "$SKIP_TRANSLATE" =~ ^(1|true|yes)$ ]] && SKIP_TRANS_FLAG=(--skip-translate)
SKIP_DUB_FLAG=();   [[ "$SKIP_DUB"   =~ ^(1|true|yes)$ ]] && SKIP_DUB_FLAG=(--skip-dub)

exec /app/nemo/.venv/bin/python /app/run_pipeline.py \
    --target-lang   "$TARGET_LANG"  \
    --input-dir     "$INPUT_DIR"    \
    --output-dir    "$OUTPUT_DIR"   \
    --run-mode      "$RUN_MODE"     \
    --precision     "$PRECISION"    \
    "${LANG_FLAG[@]}"               \
    "${NEMO_FLAG[@]}"               \
    "${CHUNK_FLAG[@]}"              \
    "${RESERVE_FLAG[@]}"            \
    "${SAFETY_FLAG[@]}"             \
    "${TRIM_FLAG[@]}"               \
    "${WHISPER_FLAG[@]}"            \
    "${QWEN_FLAG[@]}"               \
    "${DEMUCS_FLAG[@]}"             \
    "${SKIP_NEMO_FLAG[@]}"          \
    "${SKIP_TRANS_FLAG[@]}"         \
    "${SKIP_DUB_FLAG[@]}"
