#!/bin/bash
# S0 smoke test: tiny subsets (3+3 canary subjects, 3 phantoms), smallest model (Qwen2.5-0.5B-Instruct), every stage
# of the pipeline end-to-end under memguard. Outputs: results/smoke/, models/smoke_key0/, figures/smoke/.
# Usage: bash src/run_smoke.sh      (idempotent; rerun resumes)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CFG=src/configs/smoke.json
M=Qwen/Qwen2.5-0.5B-Instruct
TAG=q05
MD=models/smoke_key0/$TAG
T0=$(date +%s)
guard smoke-corpus $PY src/fmp/corpus.py --config $CFG
log "unit tests"; (cd src && $PY -m pytest tests -q -p no:cacheprovider) | tail -3
guard smoke-bench $PY src/bench/bench_prefix.py $M 16
train_variant $CFG $MD $TAG $M planted planted
train_variant $CFG $MD $TAG $M retain_only retrain
for meth in ga graddiff npo rmu; do unlearn_variant $CFG $MD $TAG $M $meth $meth; done
guard smoke-detector $PY src/fmp/detector.py --config $CFG --mode detector
unlearn_variant $CFG $MD $TAG $M adaptive adaptive50 "" results/smoke/detector/adaptive_forget_tpr50.jsonl
unlearn_variant $CFG $MD $TAG $M adaptive adaptive90 "" results/smoke/detector/adaptive_forget_tpr90.jsonl
eval_variant $CFG $MD $TAG $M base probes,trap,utility,mia 10
eval_variant $CFG $MD $TAG $M planted probes,m20,trap,utility,mia,samples 10
eval_variant $CFG $MD $TAG $M retrain probes,trap,utility,mia,samples 10
for v in ga graddiff npo rmu adaptive50 adaptive90; do eval_variant $CFG $MD $TAG $M $v probes,trap,utility,mia 10; done
guard smoke-filters $PY src/fmp/filters.py --config $CFG
guard smoke-rag $PY src/fmp/rag.py --config $CFG --reader $M
guard smoke-ppl $PY src/fmp/detector.py --config $CFG --mode ppl --lm $M
guard smoke-analyze $PY src/fmp/analyze.py --config $CFG
guard smoke-aggregate $PY src/aggregate.py --run smoke
guard smoke-plots $PY src/plots.py --run smoke
log "SMOKE DONE in $(( $(date +%s) - T0 )) s"
