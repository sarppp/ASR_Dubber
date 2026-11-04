#!/bin/bash
# Register Tower-Plus-2B as the pipeline translator in Ollama.
#
#   ./setup_tower.sh              # download GGUF if needed, create the model
#   TOWER_GGUF=/path/to.gguf ./setup_tower.sh
#
# TranslateGemma stays available: TRANSLATE_MODEL=translategemma:4b or --model.
set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="${TOWER_MODEL:-tower-plus-2b:latest}"
GGUF="${TOWER_GGUF:-$HOME/python-tools/models/tower-plus-2b/Tower-Plus-2B.Q4_K_M.gguf}"
REPO="DZgas/Tower-Plus-2B-GGUF"
FILE="Tower-Plus-2B.Q4_K_M.gguf"
CONTAINER="${OLLAMA_CONTAINER_NAME:-ollama}"
MODELS_DIR="${OLLAMA_MODELS_DIR:-$HOME/python-tools/.ollama_models}"

mkdir -p "$(dirname "$GGUF")"
if [ ! -s "$GGUF" ]; then
  echo "downloading $FILE (~1.7 GB)"
  curl -fL --retry 2 --connect-timeout 20 \
    "https://huggingface.co/$REPO/resolve/main/$FILE" -o "$GGUF.part"
  mv "$GGUF.part" "$GGUF"
fi
echo "GGUF: $GGUF ($(du -h "$GGUF" | cut -f1))"

# Build the model inside whichever Ollama is running, else start a container.
if curl -sf "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null; then
  echo "using running Ollama at ${OLLAMA_HOST:-http://127.0.0.1:11434}"
  # The GGUF must be visible to Ollama; a container needs a shared mount.
  tmp=$(mktemp -d)
  cp Modelfile.tower-plus-2b "$tmp/Modelfile"
  cp "$GGUF" "$tmp/$(basename "$FILE")"
  ( cd "$tmp" && ollama create "$MODEL_NAME" -f Modelfile )
  rm -rf "$tmp"
else
  echo "starting container $CONTAINER"
  tmp=$(mktemp -d)
  cp Modelfile.tower-plus-2b "$tmp/Modelfile"
  cp "$GGUF" "$tmp/$(basename "$FILE")"
  docker run --rm --gpus all \
    -v "$tmp:/build" -v "$MODELS_DIR:/root/.ollama" \
    --entrypoint /bin/sh ollama/ollama:latest \
    -c "cd /build && ollama serve & sleep 5 && ollama create '$MODEL_NAME' -f Modelfile"
  rm -rf "$tmp"
fi

echo
echo "done: TRANSLATE_MODEL=$MODEL_NAME"
echo "gemma alternative: TRANSLATE_MODEL=translategemma:4b"
