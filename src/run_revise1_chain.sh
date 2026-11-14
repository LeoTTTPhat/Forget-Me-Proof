#!/bin/bash
# waits for worker A of run_revise1.sh, then runs worker C (= worker B's steps; finished steps are skipped by markers)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
until [ -f results/rev1/markers/WORKER_A ] || ! pgrep -f "run_revise1.sh A" >/dev/null; do sleep 60; done
bash src/run_revise1b.sh C
