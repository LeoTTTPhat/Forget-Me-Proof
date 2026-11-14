#!/bin/bash
# Stage 4 (completeness) gap runs, 2026-10-17. Closes the seed gaps found by the completeness audit (results/GAPS.md):
#   G1 third key / FT seed (key2/seed2): Qwen-3B planted, retrain, NPO, GA  -> planted/retrain get 3 key/FT seeds
#   G2 extra unlearning seeds s1, s2 for GradDiff, RMU and adaptive50/90 (key0)  -> every approximate method gets >= 3 seeds
# Every item is idempotent (DONE markers + keyed JSONL) and has a start cut-off so the chain stays within the 6 h Stage-4 budget.
# Usage: nohup bash src/run_gaps.sh > logs/gaps.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CFG=src/configs/base.json
CFG2=src/configs/key2.json
Q3=Qwen/Qwen2.5-3B-Instruct
DP=deleted,phantom
T0=${GAPS_T0:-$(date +%s)}
CUT=$((T0 + 5 * 3600))     # no new item starts later than 5 h after launch (final analysis follows)
past() { [ -n "${HONOUR_CUTOFFS:-}" ] && [ "$(date +%s)" -ge "$CUT" ]; }
item() { if past; then log "DESCOPED $* (Stage-4 cut-off passed)"; return 1; fi; return 0; }

g1() {
  local MD=models/key2_key2/qwen3b
  item key2 || return 0
  guard corpus-key2 $PY src/fmp/corpus.py --config $CFG2
  train_variant $CFG2 $MD qwen3b $Q3 planted planted 2
  eval_variant $CFG2 $MD qwen3b $Q3 planted probes,utility 10
  item key2-retrain || return 0
  train_variant $CFG2 $MD qwen3b $Q3 retain_only retrain 2
  eval_variant $CFG2 $MD qwen3b $Q3 retrain probes,utility 10 $DP
  for m in npo ga; do
    item key2-$m || continue
    unlearn_variant $CFG2 $MD qwen3b $Q3 $m $m; eval_variant $CFG2 $MD qwen3b $Q3 $m probes,utility 10 $DP
  done
}

g2() {
  local MD=models/full_key0/qwen3b
  for s in 1 2; do
    for m in graddiff rmu; do
      item $m-s$s || continue
      unlearn_variant $CFG $MD qwen3b $Q3 $m $m-s$s $s
      eval_variant $CFG $MD qwen3b $Q3 $m-s$s probes,utility 10 $DP
    done
    for t in 50 90; do
      item adaptive$t-s$s || continue
      unlearn_variant $CFG $MD qwen3b $Q3 adaptive adaptive$t-s$s $s results/main/detector/adaptive_forget_tpr$t.jsonl
      eval_variant $CFG $MD qwen3b $Q3 adaptive$t-s$s probes,utility 10 $DP
    done
  done
}

final() {
  guard analyze $PY src/fmp/analyze.py --config $CFG
  guard analyze-key1 $PY src/fmp/analyze.py --config src/configs/key1.json
  [ -d results/key2/probes ] && guard analyze-key2 $PY src/fmp/analyze.py --config $CFG2 || true
  guard aggregate $PY src/aggregate.py
  guard plots $PY src/plots.py --run main
}

LOCK=logs/run_gaps.pid
if [ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && [ "$(cat $LOCK)" != "$$" ]; then
  log "another run_gaps.sh (pid $(cat $LOCK)) is running; exiting"; exit 0
fi
echo $$ > $LOCK
trap '[ "$(cat $LOCK 2>/dev/null)" = "$$" ] && rm -f $LOCK' EXIT
for st in ${*:-g1 g2 final}; do
  log "=== gaps stage $st ==="
  set +e; ( set -e; $st ); rc=$?; set -e
  [ $rc -ne 0 ] && log "gaps stage $st ended rc=$rc"
done
log "GAPS DONE (${*:-g1 g2 final}) in $(( $(date +%s) - T0 )) s"
