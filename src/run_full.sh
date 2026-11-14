#!/bin/bash
# Full experiment matrix (PLAN.md Sec. 6), in priority order; every step is idempotent and resumable
# (DONE markers for training, keyed JSONL for evaluation). Run detached:
#   nohup bash src/run_full.sh [STAGE...] > logs/full.log 2>&1 &
# STAGES (default after the Stage-2 trim: core e10 e11 final; e13 and e12 are implemented but descoped, see PLAN.md 6b)
#   core = E1-E9 + E14/E15 on Qwen2.5-3B-Instruct (key0); e11 = second key + FT seed; e10 = Llama-3.2-3B replication;
#   e13 = extra unlearning seeds; e12 = Qwen2.5-7B headline (dropped first when short of time); final = analyze+aggregate+plots.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CFG=src/configs/base.json
CFG1=src/configs/key1.json
Q3=Qwen/Qwen2.5-3B-Instruct
L3=unsloth/Llama-3.2-3B-Instruct
Q7=Qwen/Qwen2.5-7B-Instruct
STAGES=${*:-core e10 e11 final}
DP=deleted,phantom   # cheaper role set for secondary variants (retained-subject probes only on base/planted/retrain/npo)

core() {
  local MD=models/full_key0/qwen3b
  guard corpus $PY src/fmp/corpus.py --config $CFG
  # E1 never-fine-tuned model (exact null on every subject) + reference scores for calibration ablation
  eval_variant $CFG $MD qwen3b $Q3 base probes,trap,utility,mia 10
  # E2 planted (= lazy provider; pre-deletion memorisation check), E3 retain-only retrain (compliant null)
  train_variant $CFG $MD qwen3b $Q3 planted planted
  eval_variant $CFG $MD qwen3b $Q3 planted probes,m20,trap,utility,mia,samples 10
  train_variant $CFG $MD qwen3b $Q3 retain_only retrain
  eval_variant $CFG $MD qwen3b $Q3 retrain probes,trap,utility,mia,samples 10
  # E4 approximate unlearning (GA, NPO core; GradDiff, RMU next in descoping order)
  for m in ga npo graddiff rmu; do
    unlearn_variant $CFG $MD qwen3b $Q3 $m $m
    if [ "$m" = npo ]; then eval_variant $CFG $MD qwen3b $Q3 npo probes,trap,utility,mia,samples 10
    else eval_variant $CFG $MD qwen3b $Q3 $m probes,trap,utility,mia 10 $DP; fi
  done
  # E5 adaptive provider: deberta detector (3 seeds) + NPO suppression of flagged statements at TPR 0.5 / 0.9
  guard detector $PY src/fmp/detector.py --config $CFG --mode detector
  for s in 50 90; do
    unlearn_variant $CFG $MD qwen3b $Q3 adaptive adaptive$s "" results/main/detector/adaptive_forget_tpr$s.jsonl
    eval_variant $CFG $MD qwen3b $Q3 adaptive$s probes,trap,utility,mia 10 $DP
  done
  # E6 filters (derived), E8 perplexity filter for traps, E9 RAG
  guard filters $PY src/fmp/filters.py --config $CFG
  guard ppl $PY src/fmp/detector.py --config $CFG --mode ppl --lm $Q3
  guard rag $PY src/fmp/rag.py --config $CFG --reader $Q3
  guard analyze $PY src/fmp/analyze.py --config $CFG
}

# Stage-3 time budget (36 h ceiling from 2026-10-15 17:16, i.e. 2026-10-17 05:16; PLAN.md 6b descoping order).
# Cut-offs are epoch seconds; a stage/sub-item whose cut-off has passed is skipped and logged as DESCOPED.
E11_START_BY=${E11_START_BY:-1792155600}   # 2026-10-16 20:00: E11 needs ~8 h + final
E11_UNLEARN_BY=${E11_UNLEARN_BY:-1792173600} # 2026-10-17 01:00: E11 NPO/GA need ~2.5 h each incl. eval
E10_GA_BY=${E10_GA_BY:-1792177200}         # 2026-10-17 02:00: E10 GA (last item of E10)
past() { [ -n "${HONOUR_CUTOFFS:-}" ] && [ "$(date +%s)" -ge "$1" ]; }

e11() {  # second key / FT seed on Qwen-3B
  local MD=models/key1_key1/qwen3b
  if past $E11_START_BY && [ ! -f $MD/planted/DONE ]; then log "DESCOPED e11 (start cut-off passed)"; return 0; fi
  guard corpus-key1 $PY src/fmp/corpus.py --config $CFG1
  train_variant $CFG1 $MD qwen3b $Q3 planted planted 1
  eval_variant $CFG1 $MD qwen3b $Q3 planted probes,utility 10
  train_variant $CFG1 $MD qwen3b $Q3 retain_only retrain 1
  eval_variant $CFG1 $MD qwen3b $Q3 retrain probes,utility 10 $DP
  for m in npo ga; do
    if past $E11_UNLEARN_BY && [ ! -f $MD/$m/DONE ]; then log "DESCOPED e11 $m (cut-off passed)"; continue; fi
    unlearn_variant $CFG1 $MD qwen3b $Q3 $m $m; eval_variant $CFG1 $MD qwen3b $Q3 $m probes,utility 10 $DP
  done
}

e10() {  # Llama-3.2-3B replication (reduced form: planted, retrain, NPO, then GA)
  local MD=models/full_key0/llama3b
  train_variant $CFG $MD llama3b $L3 planted planted
  eval_variant $CFG $MD llama3b $L3 planted probes,utility,mia 10
  train_variant $CFG $MD llama3b $L3 retain_only retrain
  eval_variant $CFG $MD llama3b $L3 retrain probes,utility,mia 10 $DP
  for m in npo ga; do
    if [ $m = ga ] && past $E10_GA_BY && [ ! -f $MD/$m/DONE ]; then log "DESCOPED e10 ga (cut-off passed)"; continue; fi
    unlearn_variant $CFG $MD llama3b $L3 $m $m; eval_variant $CFG $MD llama3b $L3 $m probes,utility 10 $DP
  done
}

e13() {  # extra unlearning seeds (3 seeds total for GA and NPO); probe deleted + retained subjects only
  local MD=models/full_key0/qwen3b
  for s in 1 2; do for m in ga npo; do
    unlearn_variant $CFG $MD qwen3b $Q3 $m $m-s$s $s
    eval_variant $CFG $MD qwen3b $Q3 $m-s$s probes,utility 10 deleted,retained
  done; done
}

e12() {  # Qwen2.5-7B headline: planted + NPO (phantoms are the null; no retrain)
  local MD=models/full_key0/qwen7b
  train_variant $CFG $MD qwen7b $Q7 planted planted
  eval_variant $CFG $MD qwen7b $Q7 planted probes,utility 10 $DP
  unlearn_variant $CFG $MD qwen7b $Q7 npo npo
  eval_variant $CFG $MD qwen7b $Q7 npo probes,utility 10 $DP
}

final() {
  guard analyze $PY src/fmp/analyze.py --config $CFG
  [ -d results/key1/probes ] && guard analyze-key1 $PY src/fmp/analyze.py --config $CFG1 || true
  guard aggregate $PY src/aggregate.py
  guard plots $PY src/plots.py --run main
}

# single-instance lock: a second run_full.sh (e.g. relaunched by a supervisor or another session) exits immediately
LOCK=logs/run_full.pid
if [ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null && [ "$(cat $LOCK)" != "$$" ]; then
  log "another run_full.sh (pid $(cat $LOCK)) is running; exiting"; exit 0
fi
echo $$ > $LOCK
trap '[ "$(cat $LOCK 2>/dev/null)" = "$$" ] && rm -f $LOCK' EXIT
T0=$(date +%s)
for st in $STAGES; do log "=== stage $st ==="; $st; done
log "FULL DONE ($STAGES) in $(( $(date +%s) - T0 )) s"
