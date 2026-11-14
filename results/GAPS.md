# Stage 4 completeness audit (2026-10-17)

Audit of `results/summary.json` and `results/*/analysis/*.json` against the experiment matrix in PLAN.md Sec. 6 / 6b
(E1-E15, metrics paragraph, ablation list, the ">= 3 seeds" rule in shared/README.md).
CLOSED = fixed in Stage 4 (new runs and/or re-analysis). OPEN = not run, with the exact reason.
Every number below is copied from `results/summary.json` or the CSVs that `src/aggregate.py` writes next to it.

Stage-4 runs: `src/run_gaps.sh` (G1 + G2 + final), a second worker `src/run_gaps_par.sh` (G2 items in reverse order on the
second guard slot), and the optional `src/run_gaps_llama.sh` (G3). Launched 11:42, finished 15:59 (4.3 h wall, of which
about 1.3 h was spent waiting for a guard slot; 4 external SIGTERMs, all retried from checkpoints). Logs: `logs/gaps*.log`.
Unit tests: 71/71 pass (`logs/pytest_stage4.log`, plus a re-run after the aggregate fix); one new test covers the new
independent-unit statistics.

## 1. Matrix coverage (what exists)

| Item | Planned | Present after Stage 4 | Status |
|---|---|---|---|
| E1 base model | Qwen-3B, key0 | yes | complete |
| E2/E3 planted / retrain | Qwen-3B key0 (+E11 key1) | key0, key1, **key2** (3 key/FT seeds) | CLOSED (G1) |
| E4 GA, NPO | 1 seed (+E13: 3 seeds) | GA/NPO: 3 unlearning seeds on key0 + key1 + key2 = 5 replicates each | complete |
| E4 GradDiff, RMU | 1 seed | **3 unlearning seeds** (s0, s1, s2) on key0 | CLOSED (G2) |
| E5 adaptive TPR 0.5 / 0.9 | 1 run each, 3 detector seeds | **3 unlearning seeds each**; detectors 3 seeds x {same, heldout} | CLOSED (G2) |
| E6 filters / routing | derived from planted logs | Qwen-3B, Llama-3B (kw_full, kw_tok, semantic; filter and route), Qwen-7B (filters only) | complete |
| E7 sampled answers | 20 canaries x 4 probes, T in {0, 0.7, 1.0}, 3 seeds | planted, retrain, NPO | complete, underpowered (O4) |
| E8 baselines | twins PPR + fixed-n, peeking binomial, Min-K%++ MIA, trap + ppl filter | all; trap test re-analysed | CLOSED for trap (C5); MIA see O7 |
| E9 RAG | compliant, lazy, partial, kw, semantic | all | complete; nulls limited (O6) |
| E10 Llama | planted, retrain, NPO, GA | one key/FT seed | OPEN (O1) |
| E11 key1 | planted, retrain, NPO, GA | yes | complete |
| E12 7B | planted + NPO, phantoms as null | yes (one seed) | OPEN for seeds (O2) |
| E13 extra unlearning seeds | GA, NPO s1-s2 | yes | complete |
| E14 ablations | m+1, probe mix, betting, canaries/subject, planting rate, temperature | all; planting-rate ablation extended | CLOSED for rate (C4) |
| E15 Type-I | MC 1e5, >= 250 compliant processes, CP CIs | MC per run; empirical views added | CLOSED (C1/C2) |
| Utility | TOFU retain accuracy + ROUGE-L, MMLU-500, TOFU forget | accuracy metrics only | OPEN for ROUGE-L (O3) |

## 2. CLOSED in Stage 4

**G1. Third key / fine-tune seed (key2/seed2; Qwen-3B planted, retrain, NPO, GA).** `src/configs/key2.json`, results in `results/key2/`.
Memorisation gate passed (deleted top-1 0.594; retained processes rejected 0.983). Deleted subjects rejected at alpha 0.05:
planted 59/60, retrain 2/60, NPO 42/60, GA 1/60; phantoms 0/60, 1/60, 1/60, 1/60. MMLU-500: planted 0.604, retrain 0.606,
NPO 0.610, GA 0.262 (GA utility collapse reproduced on a third key).
Across the 3 key/FT seeds (`across_seeds.csv`, t-interval clipped to [0,1]): planted rejection 0.922 +/- 0.079,
retrain 0.017 +/- 0.017, NPO 0.600 +/- 0.145, GA 0.006 +/- 0.010.
All replicates (`across_all_seeds.csv`): NPO 0.607 +/- 0.148 (n = 5), GA 0.010 +/- 0.009 (n = 5).

**G2. Unlearning seeds 1-2 for GradDiff, RMU, adaptive50, adaptive90 (key0).** Deleted rejections /60 for seeds 0, 1, 2:
GradDiff 16, 4, 2 (mean 0.122 +/- 0.126: strongly seed-dependent, so the Stage-3 single-seed value 16/60 is the high end);
RMU 46, 45, 44 (0.750 +/- 0.017); adaptive50 32, 34, 34 (0.556 +/- 0.019); adaptive90 28, 24, 29 (0.450 +/- 0.044).
MMLU is flat across seeds (GradDiff 0.604 +/- 0.004, RMU 0.625 +/- 0.002, adaptive 0.601-0.602).
Note: these seeds vary only the unlearning run (same planted model, same flagged set for the adaptive runs).

**C1. Independent-unit statistics (problem: n = 60 "processes" are 20 subjects x 3 orderings of the same canary ranks).**
`summarize_processes` (src/fmp/analyze.py) now adds, per role and alpha, `order0` (pre-registered ordering only: one process
per subject, Clopper-Pearson CI) and `subject_cluster_boot` (percentile bootstrap over subjects, B = 2000, seed 0, of the
pooled rate and of the median canaries-to-decision). `headline.csv` has the new columns `deleted_order0_k/n/ci95`,
`deleted_ci95_subject_cluster`, `deleted_canaries_to_decision_ci95`, `phantom_order0_k/n`, `phantom_ci95_subject_cluster`.
Example: Qwen-3B planted 50/60, ordering 0 18/20, subject-cluster CI [0.683, 0.950], canaries to decision 3 [3, 3].
Stage 5/6 must quote ordering-0 counts or cluster CIs, not the Clopper-Pearson CI over 60 processes.

**C2. Type-I without duplicated derived variants (Stage-3 issue 2).** `typeI_empirical.json` gains
`trained_variants` (only model-generated variants; derived filter_/route_ variants and ablation configs excluded) and
`trained_variants_order0`, each with a subject-cluster and a (variant, subject)-cluster bootstrap CI and per-variant counts.
main: 32/1560 = 0.021 (CP [0.014, 0.029]; subject-cluster [0.005, 0.040]); ordering 0: 13/520 = 0.025 (CP [0.013, 0.042];
subject-cluster [0.004, 0.055]). Pooled over the 3 keys (`typeI_pooled.csv`, new): trained variants 46/2160 = 0.021
(CP [0.016, 0.028]); ordering 0 15/720 = 0.021 (CP [0.012, 0.034]); alpha 0.01: 4/2160 and 1/720.
The old pooled default-config count is now main 60/2460 (more variants), all keys 74/3060 = 0.024.
Remaining dependence: the phantoms of one key are the same 20 subjects on every model of that run, hence the cluster CIs.

**C3. Queries-to-decision uncertainty.** Bootstrap CI over subjects of the median canaries-to-decision is now in every main
and RAG entry (`canaries_to_decision_median_ci95`).

**C4. Planting-rate ablation as evidence, not only top-1.** `ablations.json[*].by_rate_eprocess`: e-processes restricted to
the canaries planted with r mentions, truncated to 2 canaries (size of the smallest rate group) so rates are compared at equal
n, plus the rank law per rate; `by_rate_top1` now has CP CIs. Qwen-3B planted, r = 1/2/4: rejected 12/15/18 of 60, KL 0.52/0.63/0.90;
Llama planted 18/33/30, Qwen-7B planted 18/21/33. Monotone in r on Qwen, not strictly on Llama; group sizes are small
(40/40/100 canaries) and CIs overlap.

**C5. Trap baseline was structurally powerless.** One trap per subject with 10 candidates gives e <= 10 < 1/alpha = 20, so the
per-subject trap test could never reject (0/20 everywhere even with trap top-1 = 1.0). Added an auditor-level test pooled over
subjects (`baselines_mia_trap.json -> trap.*.*.pooled_over_subjects`: plug-in e-process over subjects + fixed-n binomial).
Result: deleted traps rejected after 2 subjects on planted, NPO, GradDiff, RMU, adaptive50/90 (log W = 43.0); not on GA, retrain or
base; phantoms never. Combined with the perplexity filter (AUC trap vs genuine 0.992) this is the "memorised but filterable" contrast.

**C6. Across-seed t-intervals left [0, 1].** `src/aggregate.py::t_ci` clips rate/accuracy intervals to [0, 1] and flags it
(`ci95_clipped_to_unit`); with n = 3 the intervals are wide and should be reported with the per-seed values.

## 3. OPEN (not run), with reasons

**O1. Llama-3.2-3B has one key/FT seed.** G3 (`src/run_gaps_llama.sh`: key1 planted, retrain, NPO) was descoped by its start
cut-off (11:42 + 3 h) because G1/G2 took until 15:59 under slot contention; Llama training ran at about 1 h per adapter under
contention in Stage 3, so G3 needs 2.5-4 h. Llama is a replication row: report it as single seed with ordering-0 and
subject-cluster CIs. Command to run later: `nohup bash src/run_gaps_llama.sh >> logs/gaps_llama.log 2>&1 &` (move the cut-off).

**O2. Qwen2.5-7B: one seed, no retrain, no retained-subject probes.** By design (phantoms are the 7B null; Stage-2 plan).
A second 7B planted + NPO seed needs about 2.5 h (44 min training at scale 0.25 + 47 min NPO + evals); not run within the 6 h
budget after G1/G2. The 7B NPO result (8/60, ordering 0 1/20) is single seed. The 3B seed spread (NPO 26-46/60) shows this
can move a lot.

**O3. ROUGE-L on greedy TOFU answers (planned utility metric) was never implemented.** Utility is TOFU candidate accuracy
(retain/forget, 200 items) and MMLU-500. Adding it needs a generation pass for about 25 variants (about 1.5-2 h GPU plus code).
Not run: the accuracy metrics already show the utility effects (GA collapse, flat MMLU elsewhere). The paper must not mention ROUGE-L.

**O4. Sampled-answer / temperature robustness (E7) is underpowered.** 20 deleted canaries per variant (Stage-2 trim); top-1
CIs are about +/- 0.2 (e.g. planted T = 0.7 top-1 0.60 vs phantom 0.10). Only a spot check is supported, not a temperature-robustness
claim. Tripling the canaries costs about 0.4 h x 2 per variant (3 variants: about 2.5 h); not run.

**O5. The m+1 = 20 ablation uses a dedicated subset** (Qwen-3B planted only, deleted subjects, direct + cloze probes). It is
internally comparable across M in {2, 5, 10, 20} but not with the default rows. There is no empirical null at M = 20; validity
there rests on the MC null (`mc_null.json`, M = 20 rows).

**O6. RAG nulls are thin.** Compliant and partial RAG logs probe only deleted subjects. RAG Type-I evidence: compliant deleted
2/60, phantoms of lazy/kw 2/60 and semantic 1/60 (the filter variants reuse the lazy log, so these are not independent);
single key. Running phantom probes on the compliant index needs a RAG pass (code supports roles); not run (time).

**O7. MIA baseline has no valid null (finding, not a gap).** Phantom false-accusation rates of the sequential Min-K%++ MIA at alpha
0.05 range from 1/20 to 9/20 (Qwen-3B planted 9/20, retrain 7/20, NPO 8/20, Llama planted 9/20), and deleted vs phantom rates
are not separated on compliant models (retrain 6/20 vs 7/20). Cause: document-level p-values of one subject are dependent and the
phantom pool is not exchangeable with subject documents. MIA ran on Qwen-3B variants and Llama planted/retrain only (20 subjects).

**O8. Adaptive provider uses one flagged set.** Both adaptive runs suppress the statements flagged by the `same`-template
detector (seed 0). Suppression driven by the `heldout` detector, which reaches only target TPR 0.13 / 0.34 at the val
operating points 0.5 / 0.9, was not run (needs a code path in detector.py for heldout forget files plus 2-6 unlearning runs).
The three adaptive seeds vary only the unlearning run.

**O9. Seed structure is heterogeneous.** planted/retrain: 3 key/FT seeds; NPO/GA: 3 unlearning seeds (key0) + 2 keys; GradDiff,
RMU, adaptive: 3 unlearning seeds on one planted model; Llama, 7B, RAG, detector-driven suppression: one key. Within a run the
"3 orderings" are not seeds.

**O10. Retained-subject probes only on base/planted/retrain/NPO (+NPO/GA seeds).** Collateral damage of GradDiff, RMU and adaptive
to retained subjects' canaries is unmeasured (TOFU-retain accuracy covers general retained knowledge). Stage-2 trim; not re-run.

**O11. Power per memorisation bucket is per canary.** `memorisation.json` gives top-1 per pre-deletion advantage tertile
(Qwen-3B planted 0.13/0.52/1.00; NPO 0.07/0.28/0.85), not per-subject rejection, because buckets are defined per canary.

**O12. Fixed-n binomial matches the e-process on raw power** (Stage-3 issue 3; key2 planted: binom10 fixed-n 60/60 vs
e-process 59/60; main 51 vs 50). The e-process advantage must be argued as anytime validity and early stopping
(binom10_peek is the invalid comparison: retrain 4/60, 5/60, 1/60 false rejections on main/key1/key2).

**O13. Rank-uniformity chi-square on null models.** key2 retrain deleted canaries give p = 0.042 (180 canaries); report all null
chi-square p-values together with the number of tests. Type-I is exact by construction; the empirical rates above are consistent.

**O14. qwen3b/retrain phantoms 6/60 (0.10)** come from 2 of 20 subjects (ordering 0: 2/20, CP [0.012, 0.317]);
P(>= 2 of 20 | alpha = 0.05) = 0.26. Not evidence of invalidity, but quote ordering-0 counts.
