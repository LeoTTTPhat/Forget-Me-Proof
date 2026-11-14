"""Stage 5: export per-process e-process paths (default verifier configuration) for every model-generated variant,
so figures (detection curves, calibration) and paired significance tests can be computed from files.

Uses exactly the analysis code of src/fmp/analyze.py (same ranks, tie-break seeds and orderings), and checks that the
exported rejections reproduce results/<run>/analysis/main.json. CPU only, ~1 min per run.

python src/export_processes.py --config src/configs/base.json   -> results/main/analysis/processes.json
Also (once, from the main run) writes results/calibration_mc.json: exact-null Monte Carlo of sup_n W_n (plug-in, M=10,
n=9) and of the naive peeking binomial on a fine alpha grid.
"""
import argparse, json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fmp.common import run_dir, load_config, load_json, save_json, RESULTS
from fmp import analyze as A, eprocess as E

ALPHA_GRID = np.round(np.concatenate([np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.2001, 0.005)]), 4).tolist()


def export_run(cfg):
    rd = run_dir(cfg)
    alphas = cfg["stats"]["alpha"]
    n_order = cfg["stats"]["orderings"]
    CAN = A.Canaries(cfg)
    out = {}
    main = load_json(os.path.join(rd, "analysis", "main.json"))
    mismatches = []
    for f in sorted(glob.glob(os.path.join(rd, "probes", "*.jsonl"))):
        tag, var = os.path.basename(f)[:-6].split("__", 1)
        vid = f"{tag}/{var}"
        T = A.load_table(f)
        ranks = A.compute_ranks(T, CAN, A.DEFAULT, tag=vid)
        tb = {(r["subject"], r["order"]): r for r in A.twin_and_binomial(T, CAN, alphas, n_order, vid)}
        procs = []
        for a in sorted({a for (a, _) in T}):
            slots = [sl for sl in A.subject_canaries(T, a) if (a, sl) in ranks]
            role = T[(a, slots[0])]["role"]
            for o in range(n_order):
                so = A.ordering(slots, a, o)
                rk = [int(ranks[(a, sl)][0]) for sl in so]
                W = E.eprocess_path(rk, 10, "plugin")
                logW = np.log(np.maximum(W, 1e-300))
                b = tb[(a, o)]
                procs.append(dict(subject=int(a), role=role, order=o, ranks=rk, logW=[round(float(x), 5) for x in logW],
                                  rates=[T[(a, sl)]["rate"] for sl in so],
                                  base={k: bool(v) for k, v in b.items() if "@" in k}))
        out[vid] = procs
        # consistency check against main.json
        for role in ["deleted", "retained", "phantom"]:
            ent = main.get(vid, {}).get(role, {}).get("alpha=0.05")
            if ent is None:
                continue
            k = sum(1 for p in procs if p["role"] == role and max(p["logW"]) >= np.log(20) - 1e-9)
            if k != ent["k"]:
                mismatches.append((vid, role, k, ent["k"]))
    save_json(dict(config=cfg["name"], strategy="plugin", M=10, processes=out, mismatches_vs_main_json=mismatches),
              os.path.join(rd, "analysis", "processes.json"), indent=None)
    print(f"[export] {cfg['name']}: {len(out)} variants, mismatches vs main.json: {mismatches}")
    return mismatches


def peek_min_p(H, p0):
    """Min over n of the one-sided binomial p-value P(Bin(n,p0) >= k_n) along each row of hits H (N, n)."""
    from scipy.stats import binom
    N, n = H.shape
    K = np.cumsum(H, 1)
    mp = np.ones(N)
    for j in range(n):
        p = binom.sf(K[:, j] - 1, j + 1, p0)
        mp = np.minimum(mp, p)
    return mp


def calibration_mc(n_runs=100000, seed=0):
    rng = np.random.default_rng(seed)
    out = dict(alpha_grid=ALPHA_GRID, n_runs=n_runs, seed=seed)
    for M, n in [(10, 9), (10, 5), (2, 9)]:
        R = rng.integers(1, M + 1, size=(n_runs, n))
        sup = np.log(E.eprocess_paths_vec(R, M, "plugin").max(1))
        out[f"plugin,M={M},n={n}"] = [float(np.mean(sup >= np.log(1 / a) - 1e-12)) for a in ALPHA_GRID]
        mp = peek_min_p((R == 1).astype(int), 1 / M)
        out[f"binom_peek,M={M},n={n}"] = [float(np.mean(mp <= a)) for a in ALPHA_GRID]
        last = None
        from scipy.stats import binom
        k = (R == 1).sum(1)
        pfix = binom.sf(k - 1, n, 1 / M)
        out[f"binom_fixedn,M={M},n={n}"] = [float(np.mean(pfix <= a)) for a in ALPHA_GRID]
    X = rng.integers(0, 2, size=(20000, 100))
    mp = peek_min_p(X, 0.5)
    out["binom_peek,twins,n=100"] = [float(np.mean(mp <= a)) for a in ALPHA_GRID]
    save_json(out, os.path.join(RESULTS, "calibration_mc.json"))
    print("[export] calibration_mc.json written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mc", action="store_true")
    a = ap.parse_args()
    mm = export_run(load_config(a.config))
    if a.mc:
        calibration_mc()
    sys.exit(1 if mm else 0)
