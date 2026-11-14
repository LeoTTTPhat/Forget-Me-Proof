#!/bin/bash
# Stage 8 pilot, style control: plain LLM rewrites of genuine sentences (negatives), then style-controlled detector.
#   nohup bash src/run_revise2_embed2.sh > logs/revise2_embed2.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MK=results/rev2_embed/markers; mkdir -p $MK
while [ ! -f $MK/detect ]; do sleep 30; done   # wait for the first pilot to finish (one guarded job at a time here)
for pass in 1 2 3; do
  ok=1
  if [ ! -f $MK/gen_neg ]; then
    set +e; "$G" --name "$PROJ-rev2-embed-genneg" --ollama-models qwen2.5:7b -- $PY src/embedded_canaries.py --mode gen_neg; rc=$?; set -e
    [ $rc -eq 0 ] && touch $MK/gen_neg || ok=0
  fi
  if [ $ok -eq 1 ] && [ ! -f $MK/detect_style ]; then
    guard rev2-embed-detect-style $PY src/embedded_canaries.py --mode detect_style && touch $MK/detect_style || ok=0
  fi
  [ $ok -eq 1 ] && { log "EMBED2 DONE"; exit 0; }
  sleep 120
done
log "EMBED2 FAILED"; exit 1
