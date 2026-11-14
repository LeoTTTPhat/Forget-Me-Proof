#!/bin/bash
# Stage 8 (revision round 2) GPU runs: probing only (no training). Name-free genuine-attribute probes (kind 'attribute',
# src/make_attr_refs.py, config src/configs/rev2_attr.json) on existing key-0 adapters, deleted + retained + 20 phantom
# subjects with 9 canaries each. Idempotent (keyed JSONL + markers); every model-loading step runs under memguard.
#   nohup bash src/run_revise2.sh > logs/revise2.log 2>&1 &
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
Q3=Qwen/Qwen2.5-3B-Instruct
L3=unsloth/Llama-3.2-3B-Instruct
MD=models/full_key0/qwen3b
LD=models/full_key0/llama3b
CF=src/configs/rev2_attr.json
MK=results/rev2_attr/markers
mkdir -p $MK
step() { local n=$1; shift; [ -f $MK/$n ] && { log "skip $n (done)"; return 0; }; guard "rev2-$n" "$@" && touch $MK/$n; }
probe() { step "eval-$1" $PY src/fmp/probe.py --config $CF --variant "$2" --model "$3" --adapters "$4" --tasks probes --n_score 10 --roles deleted,retained,phantom; }
work() {
  [ -f data/derived/full_key0/attr_refs.json ] || $PY src/make_attr_refs.py --config src/configs/base.json
  step filters $PY src/attr_filters.py --config $CF
  probe planted qwen3b__planted $Q3 $MD/planted/adapter
  probe retrain qwen3b__retrain $Q3 $MD/retrain/adapter
  probe base qwen3b__base $Q3 ""
  probe npo qwen3b__npo $Q3 $MD/planted/adapter,$MD/npo/adapter
  probe llama-planted llama3b__planted $L3 $LD/planted/adapter
  probe llama-retrain llama3b__retrain $L3 $LD/retrain/adapter
  probe scrub-same qwen3b__scrub-same $Q3 $MD/scrub-same/adapter
}
T0=$(date +%s)
for pass in 1 2 3; do
  log "=== revise2 pass $pass ==="
  set +e; ( set -e; work ); rc=$?; set -e
  [ $rc -eq 0 ] && { log "REVISE2 DONE in $(( $(date +%s) - T0 )) s"; touch $MK/ALL_DONE; exit 0; }
  log "pass $pass failed rc=$rc; retrying in 120 s"; sleep 120
done
log "REVISE2 FAILED"; exit 1
