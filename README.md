# Forget-Me-Proof: reproduction files

Forget-Me-Proof lets a data subject test whether a fine-tuned language model, or a retrieval-augmented (RAG) system, still
depends on facts the subject planted in their own documents. A secret key picks each planted value among N = 20
candidates. After the deletion request, the subject scores M = 10 candidates per planted fact from log-probabilities and
turns the rank of the planted value into an e-value. The product of e-values is a test martingale, so the evidence can be
monitored at any time without inflating false accusations.

This folder holds everything needed to repeat the experiments: code, configurations, derived corpora, the logged probe
scores of every run, the analysis outputs and the figures.

## Layout

| Path | Contents |
|---|---|
| `src/fmp/` | Core package: keyed canaries (`canary.py`), corpora (`corpus.py`), LoRA training (`train_lora.py`), approximate unlearning (`unlearn.py`), probing (`probe.py`), RAG (`rag.py`), filters (`filters.py`), provider-side canary detector (`detector.py`), e-processes and tests (`eprocess.py`), metrics (`analyze.py`) |
| `src/*.py` | Data acquisition, aggregation, statistics, follow-up analyses (`revise1_analysis.py` to `revise4_analysis.py`), figure scripts |
| `src/run_*.sh`, `src/lib.sh` | Experiment drivers; every step is idempotent and resumes from checkpoints |
| `src/configs/` | One JSON configuration per run (`base.json` = key 0, `key1.json`, `key2.json`, follow-up runs, `smoke.json`) |
| `src/tests/` | Unit tests (79): exact type-I error, martingale property by enumeration, Ville's inequality by simulation, query bound, exchangeability, tie-breaking, canary uniformity, an end-to-end synthetic pipeline |
| `data/canary_pools.json` | Public value pools for the canary candidates |
| `data/derived/` | Derived corpora of fictitious TOFU subjects for keys 0, 1 and 2 (planted and retain-only corpora, forget and retain sets, RAG passages, canary metadata) |
| `data/data_stats.json` | Statistics of the source datasets |
| `results/<run>/probes/*.jsonl` | Per-probe candidate scores of every evaluated model variant (the raw evidence) |
| `results/<run>/analysis/*.json`, `results/summary.json`, `results/stats_tests.json`, `results/revise*/` | All computed results |
| `figures/` | All figures (PDF); `figures/figures_manifest.json` maps each figure to its source files |
| `figkit/` | Script for the two icon figures (`fig_overview_art`, `fig_outcomes_art`) |
| `requirements.txt` | Exact Python package versions used (Python 3.11) |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run every command below from this folder. The shell drivers read two optional environment variables:
- `PY`: the Python interpreter (default `python`).
- `MEMGUARD`: a wrapper that runs each heavy step as `$MEMGUARD --name NAME -- command` (we used a memory guard on a
  shared machine). Leave it unset to run steps directly.

## 1. Check the results from the logged scores (CPU, about 5 minutes)

These commands rebuild every analysis file from the per-probe logs in `results/`. On the shipped folder they reproduce
the shipped files exactly (apart from the time stamp in `summary.json`).

```bash
(cd src && python -m pytest tests -q)
python src/fmp/analyze.py --config src/configs/base.json
python src/fmp/analyze.py --config src/configs/key1.json
python src/fmp/analyze.py --config src/configs/key2.json
python src/aggregate.py
python src/stats_tests.py
python src/export_processes.py --config src/configs/base.json
python src/revise1_analysis.py --force
python src/revise2_analysis.py --force
python src/revise3_analysis.py --force
python src/revise4_analysis.py --force
python src/extra_quantities.py
python src/plots.py --run main
python src/make_figures.py
python src/make_figures_rev1.py
python figkit/p2_figures.py
```

## 2. Re-run the experiments (GPU)

**Models.** Download these Hugging Face models into the local cache first, because the drivers run with
`HF_HUB_OFFLINE=1`:
- `Qwen/Qwen2.5-3B-Instruct`
- `unsloth/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct` (smoke test)
- `microsoft/deberta-v3-base` (provider-side canary detector)
- `BAAI/bge-small-en-v1.5` (RAG index and semantic filter)

The LLM-embedded canary pilot also uses `qwen2.5:7b` served by Ollama.

**Data.** `src/acquire_data.py` exports the source datasets into `data/raw/`:
- `locuslab/TOFU` (MIT)
- `muse-bench/MUSE-News`
- `databricks-dolly-15k` (CC BY-SA 3.0)
- a 1-in-200 sample of `teknium/OpenHermes-2.5`
- the `cais/mmlu` test split

Download the Hugging Face datasets into the local cache first. The derived corpora in `data/derived/` are already
included, so this step only matters for the utility and background data.

**Runs.** Run them in this order:

```bash
bash src/run_smoke.sh                      # end-to-end check on Qwen2.5-0.5B with tiny subsets
bash src/run_full.sh core e10 e11 final    # main matrix (Qwen2.5-3B key 0, Llama-3.2-3B, second key)
bash src/run_extra.sh                      # extra unlearning seeds and Qwen2.5-7B
bash src/run_gaps.sh                       # third key and further unlearning seeds (src/run_gaps_par.sh is an optional second worker)
bash src/run_gaps_llama.sh                 # Llama-3.2-3B with a second key
bash src/run_revise1.sh A; bash src/run_revise1.sh B; bash src/run_revise1b.sh A; bash src/run_revise1b.sh B
bash src/run_revise2.sh                    # name-free attribute probes
bash src/run_revise2_embed.sh; bash src/run_revise2_embed2.sh   # LLM-embedded canaries (Ollama)
```

Then run the commands of section 1.

Each driver can be stopped and restarted: finished training runs leave `DONE` markers, and evaluation appends to keyed
JSONL files. The original runs used wall-clock cut-offs to fit a time budget. A re-run ignores them unless
`HONOUR_CUTOFFS=1` is set. We ran everything on one Apple M3 Ultra (512 GB unified memory) with the PyTorch MPS backend.

## Notes

- The experimental subject keys are derived from public strings so that every run can be checked. A real deployment
  must use a secret 128-bit key per subject and publish the commitment before sharing any document.
- All subjects are fictitious TOFU authors; the corpora contain no data about real people.
- Not included: model weights and adapters (retrain them with the drivers), the raw third-party datasets (re-download
  them), and the smoke-test outputs.
- The icon set in `figkit/icons/` is Phosphor Icons v2.1.1 (MIT licence, `figkit/icons/LICENSE-phosphor-MIT.txt`).
