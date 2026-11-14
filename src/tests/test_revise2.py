"""Tests for the revision-2/3 analysis code (src/revise2_analysis.py, src/revise3_analysis.py): exact DP null sizes and
thresholds, closed-form plug-in wealth, KM-prior bets, the rebuttal cap, and the exact McNemar test."""
import math
import numpy as np
from fmp import eprocess as E
import revise2_analysis as R2
import revise3_analysis as R3

M = 10
ALPHA = 0.05


def test_logw_plugin_matches_sequential_wealth():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = list(rng.integers(1, M + 1, size=9))
        W = E.eprocess_path(r, M, "plugin")
        cnt = tuple(np.bincount(r, minlength=M + 1)[1:])
        assert abs(R2.logW_plugin(cnt) - math.log(W[-1])) < 1e-9


def test_dp_plugin_null_size_matches_monte_carlo():
    U = [np.full(M, 1 / M)] * 9
    dp = R2.dp_cross(U, R2.logW_plugin, math.log(1 / ALPHA))[-1]
    assert abs(dp - 0.0273) < 5e-4  # value reported in results/revise2/exact.json
    rng = np.random.default_rng(1)
    n = 40000
    hits = sum(E.first_crossing(E.eprocess_path(list(rng.integers(1, M + 1, size=9)), M, "plugin"), ALPHA) is not None for _ in range(n))
    assert abs(hits / n - dp) < 4 * math.sqrt(dp * (1 - dp) / n)


def test_exact_threshold_size_at_most_alpha():
    c, size = R2.null_threshold(R2.logW_plugin)
    assert size <= ALPHA + 1e-12
    assert abs(math.exp(c) - 8.35) < 0.01
    # a slightly lower threshold exceeds alpha (the threshold is the smallest valid one up to bisection precision)
    U = [np.full(M, 1 / M)] * 9
    assert R2.dp_cross(U, R2.logW_plugin, c - 0.05)[-1] > ALPHA


def test_dp_matches_monte_carlo_under_alternative():
    P = np.array([0.4, 0.15, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02])
    dp = R2.dp_cross([P] * 9, R2.logW_plugin, math.log(1 / ALPHA))[-1]
    rng = np.random.default_rng(2)
    n = 20000
    hits = sum(E.first_crossing(E.eprocess_path(list(rng.choice(np.arange(1, M + 1), size=9, p=P)), M, "plugin"), ALPHA) is not None
               for _ in range(n))
    assert abs(hits / n - dp) < 4 * math.sqrt(dp * (1 - dp) / n)


def test_km_prior_bet_sums_to_M_and_is_valid():
    for lam in R2.LAMS:
        f = (1 - lam) + lam * M * R2.Q
        assert abs(f.sum() - M) < 1e-9 and f.min() > 0
    # the KM-prior mixture is a test martingale under independent uniform ranks: Ville bound holds in simulation
    rng = np.random.default_rng(3)
    n, acc = 20000, 0
    for _ in range(n):
        ks = [{k: int(rng.integers(1, M + 1)) for k in R2.KINDS5} for _ in range(9)]
        acc += E.first_crossing(R2.km_prior_path(ks, 0.75, R2.KINDS5), ALPHA) is not None
    assert acc / n <= ALPHA


def test_rebuttal_cap_lowers_wealth_pathwise():
    rng = np.random.default_rng(4)
    for _ in range(200):
        r = list(rng.integers(1, M + 1, size=9))
        bet = E.PluginBet(M)
        W, Wc = [1.0], [1.0]
        rebut = set(rng.choice(9, size=3, replace=False).tolist())
        for j, x in enumerate(r):
            e = bet.e(x); bet.update(x)
            W.append(W[-1] * e); Wc.append(Wc[-1] * (min(e, 1.0) if j in rebut else e))
        assert all(c <= w + 1e-12 for c, w in zip(Wc, W))


def test_rebuttal_cap_valid_with_nonuniform_rebutted_ranks():
    """Supplement S1, 'Soundness when (A3) fails': the provider learned the rebutted canaries' values from the rebuttal
    source, so their planted value always ranks first (deliberately non-uniform); the rebutted set is chosen independently
    of the key material and the other ranks are uniform. The judge's capped plug-in e-process (rebutted factors
    min{f_j(R_j), 1}, rank counts kept for later bets) stays valid at alpha, whereas the uncapped process is not. A
    key-dependent rebutted set (the subject republishes the canaries whose planted value the compliant model disfavours)
    is the excluded case, and the same check then fails."""
    rng = np.random.default_rng(5)
    n_paths, n_can = 20000, 9

    def max_wealth(ranks, rebut):
        bet = E.PluginBet(M)
        W = Wc = mW = mWc = 1.0
        for j, x in enumerate(ranks):
            e = bet.e(x); bet.update(x)
            W *= e; Wc *= min(e, 1.0) if j in rebut else e
            mW, mWc = max(mW, W), max(mWc, Wc)
        return mW, mWc

    hit = hit_c = 0
    for _ in range(n_paths):
        rebut = set(rng.choice(n_can, size=3, replace=False).tolist())
        ranks = [1 if j in rebut else int(rng.integers(1, M + 1)) for j in range(n_can)]
        w, wc = max_wealth(ranks, rebut)
        hit += w >= 1 / ALPHA; hit_c += wc >= 1 / ALPHA
    se = math.sqrt(ALPHA * (1 - ALPHA) / n_paths)
    assert hit_c / n_paths <= ALPHA + 3 * se
    assert hit / n_paths > 3 * ALPHA
    hit_sel = 0
    for _ in range(n_paths):
        base = rng.integers(1, M + 1, size=n_can)          # rank of the planted value under the compliant model
        rebut = {j for j in range(n_can) if base[j] != 1}   # republished and learned: rank 1 after training on the source
        hit_sel += max_wealth([1] * n_can, rebut)[1] >= 1 / ALPHA
    assert hit_sel / n_paths > 2 * ALPHA


def test_mcnemar_exact():
    a = [True] * 5 + [False] * 15
    b = [False] * 20
    assert abs(R3.mcnemar_exact(a, b)["p_two_sided"] - 0.0625) < 1e-12
    assert R3.mcnemar_exact(b, b)["p_two_sided"] == 1.0
