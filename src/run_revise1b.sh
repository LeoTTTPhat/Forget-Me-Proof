#!/bin/bash
# Stage 7 (revision round 1) GPU runs. Two workers (the machine allows 2 guarded jobs); every step is idempotent
# (training DONE markers, keyed probe JSONL, step markers in results/rev1/markers) and every model-loading step runs
# under memguard via guard() from lib.sh. Supervisor: each worker makes up to 3 passes over its steps.
#   nohup bash src/run_revise1.sh A > logs/revise1_A.log 2>&1 &
#   nohup bash src/run_revise1.sh B > logs/revise1_B.log 2>&1 &
# Outputs: results/rev1 (key-0 new variants), results/rev1_ph9{,_key1,_key2} (phantom canaries 5-8 = 9-canary phantom audits).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
Q3=Qwen/Qwen2.5-3B-Instruct
L3=unsloth/Llama-3.2-3B-Instruct
MD=models/full_key0/qwen3b
C=src/configs
MK=results/rev1/markers
mkdir -p $MK

# step NAME CMD... : run once (marker), via guard
step() { local n=$1; shift; [ -f $MK/$n ] && { log "skip $n (done)"; return 0; }; guard "rev1-$n" "$@" && touch $MK/$n; }

probe() {  # probe NAME CFG VARIANT MODEL ADAPTERS TASKS ROLES
  step "eval-$1" $PY src/fmp/probe.py --config "$2" --variant "$3" --model "$4" --adapters "$5" --tasks "$6" --n_score 10 --roles "$7"
}

worker_A() {
  # 1. canary-scrubbing provider (exact-template detector, label-free threshold 0), retrained like the compliant model
  [ -f $MD/scrub-same/DONE ] || guard rev1-train-scrub-same $PY src/fmp/train_lora.py --config $C/rev1.json --model $Q3 --corpus scrub_same --out $MD/scrub-same
  probe scrub-same $C/rev1.json qwen3b__scrub-same $Q3 $MD/scrub-same/adapter probes,utility,mia deleted,retained,phantom
  # 2. 1-epoch planted model (memorisation curve, no dilution)
  [ -f $MD/planted-1ep/DONE ] || guard rev1-train-planted-1ep $PY src/fmp/train_lora.py --config $C/rev1_1ep.json --model $Q3 --corpus planted --out $MD/planted-1ep
  probe planted-1ep $C/rev1.json qwen3b__planted-1ep $Q3 $MD/planted-1ep/adapter probes,utility deleted,phantom
  # 3. scrubbing with the held-out-phrasing detector
  [ -f $MD/scrub-heldout/DONE ] || guard rev1-train-scrub-heldout $PY src/fmp/train_lora.py --config $C/rev1.json --model $Q3 --corpus scrub_heldout --out $MD/scrub-heldout
  probe scrub-heldout $C/rev1.json qwen3b__scrub-heldout $Q3 $MD/scrub-heldout/adapter probes,utility deleted,phantom
}

worker_B() {
  # 1. 9-canary phantom audits: phantom canaries 5-8 on the compliant models, the never-trained model, planted and NPO
  probe ph9-base $C/rev1_ph9.json qwen3b__base $Q3 "" probes phantom
  probe ph9-retrain $C/rev1_ph9.json qwen3b__retrain $Q3 $MD/retrain/adapter probes phantom
  probe ph9-key1-retrain $C/rev1_ph9_key1.json qwen3b__retrain $Q3 models/key1_key1/qwen3b/retrain/adapter probes phantom
  probe ph9-key2-retrain $C/rev1_ph9_key2.json qwen3b__retrain $Q3 models/key2_key2/qwen3b/retrain/adapter probes phantom
  probe ph9-llama-retrain $C/rev1_ph9.json llama3b__retrain $L3 models/full_key0/llama3b/retrain/adapter probes phantom
  probe ph9-planted $C/rev1_ph9.json qwen3b__planted $Q3 $MD/planted/adapter probes phantom
  probe ph9-npo $C/rev1_ph9.json qwen3b__npo $Q3 $MD/planted/adapter,$MD/npo/adapter probes phantom
  # 2. gradient-ascent trajectory (adapter after each of the 5 epochs; epoch 5 = the GA configuration of the paper)
  [ -f $MD/ga-traj/DONE ] || guard rev1-unlearn-ga-traj $PY src/fmp/unlearn.py --config $C/rev1.json --model $Q3 --planted $MD/planted/adapter --method ga --out $MD/ga-traj --save_epochs
  for e in 1 2 3 4; do
    probe ga-ep$e $C/rev1.json qwen3b__ga-ep$e $Q3 $MD/planted/adapter,$MD/ga-traj/epoch$e/adapter probes,utility deleted,phantom
  done
  # 3. stronger RMU (10x learning rate)
  [ -f $MD/rmu-strong/DONE ] || guard rev1-unlearn-rmu-strong $PY src/fmp/unlearn.py --config $C/rev1_rmus.json --model $Q3 --planted $MD/planted/adapter --method rmu --out $MD/rmu-strong
  probe rmu-strong $C/rev1.json qwen3b__rmu-strong $Q3 $MD/planted/adapter,$MD/rmu-strong/adapter probes,utility deleted,phantom
  # 4. dilution: planted corpus + 4,000 extra background items (subject share 0.96% -> 0.25% of words), 1 epoch
  [ -f $MD/diluted-1ep/DONE ] || guard rev1-train-diluted-1ep $PY src/fmp/train_lora.py --config $C/rev1_1ep.json --model $Q3 --corpus diluted --out $MD/diluted-1ep
  probe diluted-1ep $C/rev1.json qwen3b__diluted-1ep $Q3 $MD/diluted-1ep/adapter probes,utility deleted,phantom
}

worker_C() {  # Stage 7: B's steps, run after A in one GPU worker so that the second guard slot stays free for CPU analysis
  worker_B
}

W=${1:-A}
T0=$(date +%s)
for pass in 1 2 3; do
  log "=== worker $W pass $pass ==="
  set +e; ( set -e; worker_$W ); rc=$?; set -e
  [ $rc -eq 0 ] && { log "WORKER $W DONE in $(( $(date +%s) - T0 )) s"; touch $MK/WORKER_$W; exit 0; }
  log "worker $W pass $pass failed rc=$rc; retrying in 120 s"; sleep 120
done
log "WORKER $W FAILED after 3 passes"; exit 1
