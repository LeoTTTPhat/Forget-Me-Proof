#!/bin/bash
# Stage-3 extension (2026-10-16): re-admits the descoped items E13 (extra unlearning seeds) and E12 (Qwen2.5-7B) because
# core + E10 finished well ahead of schedule. Runs AFTER run_full_supervised.sh has exited, then re-runs `final`.
# Every item has a cut-off (epoch seconds) inside the 36 h Stage-3 ceiling (2026-10-17 05:16); items past it are skipped
# and logged as DESCOPED. Usage: nohup bash src/run_extra.sh [WAIT_PID] >> logs/full.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then W=$1; shift; while kill -0 "$W" 2>/dev/null; do sleep 30; done; fi
CFG=src/configs/base.json
Q3=Qwen/Qwen2.5-3B-Instruct
Q7=Qwen/Qwen2.5-7B-Instruct
E13_BY=${E13_BY:-1792177200}          # 2026-10-17 02:00: no new E13 item after this
E12_START_BY=${E12_START_BY:-1792162800}  # 2026-10-16 22:00: E12 needs ~6 h
E12_NPO_BY=${E12_NPO_BY:-1792175400}  # 2026-10-17 01:30
FINAL_BY=1792180800                   # 2026-10-17 03:00 (informational)
past() { [ -n "${HONOUR_CUTOFFS:-}" ] && [ "$(date +%s)" -ge "$1" ]; }

e13() {  # extra unlearning seeds (3 seeds total for GA and NPO); probe deleted + retained subjects only
  local MD=models/full_key0/qwen3b
  for s in 1 2; do for m in npo ga; do
    if past $E13_BY && [ ! -f $MD/$m-s$s/DONE ]; then log "DESCOPED e13 $m-s$s (cut-off passed)"; continue; fi
    unlearn_variant $CFG $MD qwen3b $Q3 $m $m-s$s $s
    eval_variant $CFG $MD qwen3b $Q3 $m-s$s probes,utility 10 deleted,retained
  done; done
}

e12() {  # Qwen2.5-7B headline: planted + NPO (phantoms are the null; no retrain)
  local MD=models/full_key0/qwen7b
  if past $E12_START_BY && [ ! -f $MD/planted/DONE ]; then log "DESCOPED e12 (start cut-off passed)"; return 0; fi
  train_variant $CFG $MD qwen7b $Q7 planted planted
  eval_variant $CFG $MD qwen7b $Q7 planted probes,utility 10 deleted,phantom
  if past $E12_NPO_BY && [ ! -f $MD/npo/DONE ]; then log "DESCOPED e12 npo (cut-off passed)"; return 0; fi
  unlearn_variant $CFG $MD qwen7b $Q7 npo npo
  eval_variant $CFG $MD qwen7b $Q7 npo probes,utility 10 deleted,phantom
}

final() {
  guard analyze $PY src/fmp/analyze.py --config $CFG
  [ -d results/key1/probes ] && guard analyze-key1 $PY src/fmp/analyze.py --config src/configs/key1.json || true
  guard aggregate $PY src/aggregate.py
  guard plots $PY src/plots.py --run main
}

T0=$(date +%s)
for st in e13 e12 final; do
  log "=== extra stage $st ==="
  # a failing item must not block the remaining ones (each step already retries inside guard)
  set +e; ( set -e; $st ); rc=$?; set -e
  [ $rc -ne 0 ] && log "extra stage $st ended rc=$rc"
done
log "EXTRA DONE (e13 e12 final) in $(( $(date +%s) - T0 )) s"
