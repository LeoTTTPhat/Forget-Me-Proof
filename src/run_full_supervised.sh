#!/bin/bash
# Stage-3 supervisor: (re)launches run_full.sh until it logs "FULL DONE", at most 8 launches.
# Usage: nohup bash src/run_full_supervised.sh [WAIT_PID] STAGES... > logs/supervisor.log 2>&1 &
# If WAIT_PID is a number, first waits for that already-running run_full.sh to exit.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then W=$1; shift; while kill -0 "$W" 2>/dev/null; do sleep 30; done; fi
STAGES=${*:-core e10 e11 final}
for n in 1 2 3 4 5 6 7 8; do
  if tail -3 logs/full.log 2>/dev/null | grep -q "FULL DONE"; then echo "[sup $(date '+%F %T')] done"; exit 0; fi
  echo "[sup $(date '+%F %T')] launch $n: run_full.sh $STAGES"
  bash src/run_full.sh $STAGES >> logs/full.log 2>&1
  echo "[sup $(date '+%F %T')] run_full.sh exited rc=$?"
  sleep 60
done
