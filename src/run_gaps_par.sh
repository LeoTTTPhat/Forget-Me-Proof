#!/bin/bash
# Stage 4: second worker for the G2 items of src/run_gaps.sh (uses the second guard slot while it is free). Processes the
# G2 list in REVERSE order so the two workers meet in the middle; collisions are safe (DONE markers, unlearn.py exits when
# done, guard() waits for an identical running command, probe.py skips finished JSONL keys). No final step (run_gaps.sh does it).
# Usage: nohup bash src/run_gaps_par.sh > logs/gaps_par.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CFG=src/configs/base.json
Q3=Qwen/Qwen2.5-3B-Instruct
DP=deleted,phantom
CUT=$((1792212120 + 5 * 3600))   # same Stage-4 cut-off as run_gaps.sh (launched 2026-10-17 11:42)
MD=models/full_key0/qwen3b
ITEMS=${*:-adaptive90-s2 adaptive50-s2 rmu-s2 graddiff-s2 adaptive90-s1 adaptive50-s1 rmu-s1 graddiff-s1}
for it in $ITEMS; do
  [ -n "${HONOUR_CUTOFFS:-}" ] && [ "$(date +%s)" -ge "$CUT" ] && { log "DESCOPED $it (Stage-4 cut-off passed)"; continue; }
  s=${it##*-s}; m=${it%-s*}
  set +e
  ( set -e
    if [[ $m == adaptive* ]]; then
      t=${m#adaptive}
      unlearn_variant $CFG $MD qwen3b $Q3 adaptive $it $s results/main/detector/adaptive_forget_tpr$t.jsonl
    else
      unlearn_variant $CFG $MD qwen3b $Q3 $m $it $s
    fi
    eval_variant $CFG $MD qwen3b $Q3 $it probes,utility 10 $DP )
  rc=$?; set -e
  [ $rc -ne 0 ] && log "item $it ended rc=$rc"
done
log "GAPS-PAR DONE"
