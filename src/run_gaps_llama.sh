#!/bin/bash
# Stage 4 optional gap item G3 (runs after src/run_gaps.sh exits, only if the 6 h Stage-4 budget still allows it):
# Llama-3.2-3B replication with a second key / FT seed (key1/seed1): planted, retrain, NPO -> Llama rows get 2 key seeds.
# Usage: nohup bash src/run_gaps_llama.sh <WAIT_PID> >> logs/gaps.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then W=$1; shift; while kill -0 "$W" 2>/dev/null; do sleep 30; done; fi
CFG1=src/configs/key1.json
L3=unsloth/Llama-3.2-3B-Instruct
GAPS_T0=${GAPS_T0:-1792212120}       # 2026-10-17 11:42 (launch of run_gaps.sh)
START_BY=$((GAPS_T0 + 3 * 3600))     # Llama planted+retrain+NPO need ~2-3 h under contention
NPO_BY=$((GAPS_T0 + 4 * 3600 + 1800))
past() { [ -n "${HONOUR_CUTOFFS:-}" ] && [ "$(date +%s)" -ge "$1" ]; }
MD=models/key1_key1/llama3b
ran=0
if past $START_BY && [ ! -f $MD/planted/DONE ]; then
  log "DESCOPED G3 llama key1 (Stage-4 start cut-off passed)"
else
  log "=== gaps stage g3 (llama key1) ==="
  train_variant $CFG1 $MD llama3b $L3 planted planted 1
  eval_variant $CFG1 $MD llama3b $L3 planted probes,utility 10
  train_variant $CFG1 $MD llama3b $L3 retain_only retrain 1
  eval_variant $CFG1 $MD llama3b $L3 retrain probes,utility 10 deleted,phantom
  if past $NPO_BY && [ ! -f $MD/npo/DONE ]; then log "DESCOPED G3 llama key1 npo (cut-off passed)"
  else unlearn_variant $CFG1 $MD llama3b $L3 npo npo; eval_variant $CFG1 $MD llama3b $L3 npo probes,utility 10 deleted,phantom; fi
  ran=1
fi
if [ $ran = 1 ]; then
  guard analyze-key1 $PY src/fmp/analyze.py --config $CFG1
  guard aggregate $PY src/aggregate.py
fi
log "GAPS-LLAMA DONE"
