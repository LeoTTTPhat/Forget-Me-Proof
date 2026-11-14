#!/bin/bash
# Shared shell helpers for run_smoke.sh / run_full.sh. Every model-loading step goes through memguard.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python}
G=${MEMGUARD:-}   # optional memory guard; empty = run steps directly
PROJ=p2-forget-me-proof-unlearning-verify
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$ROOT"
mkdir -p logs

log() { echo "[$(date '+%F %T')] $*"; }

# guard <step-name> <cmd...>
# Retries a failed step up to 6 times (all steps resume from their checkpoints). The shared memguard can crash on a
# transient psutil error (seen 2026-10-15: SystemError in proc_cmdline); its child would then keep running unguarded,
# so before retrying we kill any process still running exactly this command.
guard() {
  local name=$1; shift
  local try=0 nerr=0 nsig=0 rc
  # A previous driver may have died (e.g. network outage) while its guarded child kept running as an orphan.
  # Never start a second copy of the same command (both would append to the same JSONL): wait for it first.
  if pgrep -f -x "$*" >/dev/null 2>&1; then
    log "STEP $name: identical command already running (orphan); waiting for it to exit"
    while pgrep -f -x "$*" >/dev/null 2>&1; do sleep 30; done
  fi
  while :; do
    try=$((try + 1))
    log "STEP $name (try $try): $*"
    set +e; if [ -n "$G" ]; then "$G" --name "$PROJ-$name" -- "$@"; else "$@"; fi; rc=$?; set -e
    [ $rc -eq 0 ] && return 0
    log "step $name failed rc=$rc"
    pkill -f -x "$*" 2>/dev/null && log "killed orphaned child of $name" || true
    # Stage 3: external SIGTERM/SIGKILL (rc 241/247 = child -15/-9 via memguard; 143/137 = guard itself signalled) are
    # frequent on this shared machine and are not our errors; they get up to 40 retries. Other failures: 6 tries.
    case $rc in 241|247|143|137) nsig=$((nsig + 1));; *) nerr=$((nerr + 1));; esac
    if [ $nerr -ge 6 ] || [ $nsig -ge 40 ]; then log "STEP $name FAILED after $try tries (errors $nerr, signals $nsig)"; return $rc; fi
    sleep $((20 * (nerr + 1)))
  done
}

# train_variant CFG MODELDIR TAG MODEL CORPUS VARIANT [SEED]
train_variant() {
  local cfg=$1 md=$2 tag=$3 model=$4 corpus=$5 var=$6 seed=${7:-}
  [ -f "$md/$var/DONE" ] && { log "skip train $tag/$var (done)"; return 0; }
  guard "train-$tag-$var" $PY src/fmp/train_lora.py --config "$cfg" --model "$model" --corpus "$corpus" --out "$md/$var" ${seed:+--seed $seed}
}

# unlearn_variant CFG MODELDIR TAG MODEL METHOD VARIANT [SEED] [FORGET_FILE]
unlearn_variant() {
  local cfg=$1 md=$2 tag=$3 model=$4 method=$5 var=$6 seed=${7:-} ff=${8:-}
  [ -f "$md/$var/DONE" ] && { log "skip unlearn $tag/$var (done)"; return 0; }
  guard "unlearn-$tag-$var" $PY src/fmp/unlearn.py --config "$cfg" --model "$model" --planted "$md/planted/adapter" \
        --method "$method" --out "$md/$var" ${seed:+--seed $seed} ${ff:+--forget_file $ff}
}

# eval_variant CFG MODELDIR TAG MODEL VARIANT TASKS NSCORE ROLES   (VARIANT=base -> no adapters)
eval_variant() {
  local cfg=$1 md=$2 tag=$3 model=$4 var=$5 tasks=$6 nscore=$7 roles=${8:-deleted,retained,phantom}
  local ad=""
  case "$var" in
    base) ad="";;
    planted|retrain) ad="$md/$var/adapter";;
    *) ad="$md/planted/adapter,$md/$var/adapter";;
  esac
  guard "eval-$tag-$var" $PY src/fmp/probe.py --config "$cfg" --variant "${tag}__${var}" --model "$model" \
        --adapters "$ad" --tasks "$tasks" --n_score "$nscore" --roles "$roles"
}
