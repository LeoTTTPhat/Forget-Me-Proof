#!/bin/bash
# Stage 8 pilot: LLM-embedded canaries, generation (Ollama qwen2.5:7b) then detector training. Idempotent, memguarded.
#   nohup bash src/run_revise2_embed.sh > logs/revise2_embed.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MK=results/rev2_embed/markers; mkdir -p $MK
for pass in 1 2 3; do
  ok=1
  if [ ! -f $MK/gen ]; then
    set +e; "$G" --name "$PROJ-rev2-embed-gen" --ollama-models qwen2.5:7b -- $PY src/embedded_canaries.py --mode gen; rc=$?; set -e
    [ $rc -eq 0 ] && touch $MK/gen || ok=0
  fi
  if [ $ok -eq 1 ] && [ ! -f $MK/detect ]; then
    guard rev2-embed-detect $PY src/embedded_canaries.py --mode detect && touch $MK/detect || ok=0
  fi
  [ $ok -eq 1 ] && { log "EMBED DONE"; exit 0; }
  sleep 120
done
log "EMBED FAILED"; exit 1
