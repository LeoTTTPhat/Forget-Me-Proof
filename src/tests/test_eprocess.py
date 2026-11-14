"""Statistical-core tests: exact Type-I under a synthetic null, martingale property, power/queries bounds,
baseline validity, invariance lemmas."""
import math
import numpy as np
import pytest
from scipy import stats as st
from fmp import eprocess as E

ALPHA = 0.05


def null_ranks(rng, n_proc, n, M):
    return rng.integers(1, M + 1, size=(n_proc, n))


def typeI(strategy, M=10, n=9, n_proc=20000, seed=0, **kw):
    rng = np.random.default_rng(seed)
    R = null_ranks(rng, n_proc, n, M)
    rej = 0
    for r in R:
        W = E.eprocess_path(r, M, strategy, **kw)
        rej += E.first_crossing(W, ALPHA) is not None
    return rej / n_proc


@pytest.mark.parametrize("strategy", E.STRATEGIES)
@pytest.mark.parametrize("M", [2, 5, 10, 20])
def test_typeI_anytime(strategy, M):
    rate = typeI(strategy, M=M, n=30 if M == 2 else 9, n_proc=6000, seed=M)
    se = math.sqrt(ALPHA * (1 - ALPHA) / 6000)
    assert rate <= ALPHA + 3 * se, (strategy, M, rate)


@pytest.mark.parametrize("strategy", E.STRATEGIES)
@pytest.mark.parametrize("M,n", [(5, 4), (10, 3), (2, 8)])
def test_martingale_mean_one_exact(strategy, M, n):
    """Exact enumeration of all M^n rank sequences: E_H0[W_k] = 1 for k = 1..n (martingale)."""
    import itertools
    tot = np.zeros(n + 1)
    for seq in itertools.product(range(1, M + 1), repeat=n):
        tot += E.eprocess_path(list(seq), M, strategy)
    assert np.allclose(tot / M ** n, 1.0, atol=1e-9), (strategy, tot / M ** n)


@pytest.mark.parametrize("strategy", E.STRATEGIES)
def test_ville_maximal_inequality_by_simulation(strategy):
    """P(sup_k W_k >= c) <= 1/c for several c (Ville), checked by simulation with optional stopping."""
    rng = np.random.default_rng(11)
    M, n, N = 10, 12, 20000
    R = null_ranks(rng, N, n, M)
    sup = np.array([E.eprocess_path(r, M, strategy).max() for r in R])
    for c in [2, 5, 10, 20, 100]:
        p = (sup >= c).mean()
        assert p <= 1 / c + 3 * math.sqrt((1 / c) * (1 - 1 / c) / N), (strategy, c, p)


def test_evalue_exact_expectation():
    """E_H0[f_j(R)] = 1 exactly (sum_r f_j(r) = M) for every strategy and arbitrary history."""
    M = 10
    for hist in [[], [1], [1, 1, 3], [10, 2, 2, 7]]:
        b = E.PluginBet(M)
        for r in hist:
            b.update(r)
        assert abs(b.f().mean() - 1) < 1e-12
    for lam in [0.1, 0.5, 1.0]:
        assert abs(np.mean([E.e_top1(r, M, lam) for r in range(1, M + 1)]) - 1) < 1e-12
        assert abs(np.mean([E.e_linear(r, M, lam) for r in range(1, M + 1)]) - 1) < 1e-12


def test_perfect_memorisation_queries():
    """Thm 2(iii): perfectly memorised canaries -> ceil(log(1/alpha)/log M) canaries with the full top-1 bet."""
    for M, expect in [(10, 2), (2, 5), (20, 1), (5, 2)]:
        W = E.eprocess_path([1] * 10, M, "top1_full")
        assert E.first_crossing(W, ALPHA) == expect == math.ceil(math.log(1 / ALPHA) / math.log(M))
    # plug-in bet with geometric prior also decides fast on perfect memorisation
    assert E.first_crossing(E.eprocess_path([1] * 10, 10, "plugin"), ALPHA) <= 3


def test_thm2_expected_stopping_bound():
    rng = np.random.default_rng(2)
    M, p = 10, 0.4
    lam = E.lam_star(p, M)
    taus = []
    for _ in range(3000):
        W, k = 1.0, 0
        while W < 1 / ALPHA and k < 500:
            r = 1 if rng.random() < p else int(rng.integers(2, M + 1))
            W *= E.e_top1(r, M, lam); k += 1
        taus.append(k)
    assert np.mean(taus) <= E.expected_canaries_bound(p, M, ALPHA)


def test_twin_ppr_valid_and_peeking_invalid():
    rng = np.random.default_rng(3)
    N, n = 4000, 60
    X = rng.integers(0, 2, size=(N, n))
    ppr = np.mean([E.first_crossing(E.twin_ppr_path(x), ALPHA) is not None for x in X])
    fixed = np.mean([E.fixed_n_binom_reject(x, n, 0.5, ALPHA) for x in X])
    peek = np.mean([E.peeking_binom_first(x, 0.5, ALPHA) is not None for x in X])
    se = math.sqrt(ALPHA * (1 - ALPHA) / N)
    assert ppr <= ALPHA + 3 * se
    assert fixed <= ALPHA + 3 * se
    assert peek > ALPHA + 3 * se  # optional stopping inflates a fixed-n test


def test_ranks_uniform_under_null_with_heterogeneous_priors():
    """Lemma (exchangeability by construction): candidate scores depend on candidate identity (priors,
    lengths) but not on which one is planted -> planted rank is exactly uniform for every aggregation rule."""
    rng = np.random.default_rng(4)
    M, P, N = 10, 16, 6000
    for method in ["sum", "z", "rank"]:
        ranks = []
        for _ in range(N):
            prior = rng.normal(0, 3, M)  # strongly non-uniform plausibility
            Pm = prior[None, :] + rng.normal(0, 1, (P, M))
            ref = prior[None, :] + rng.normal(0, 1, (P, M))
            planted = int(rng.integers(0, M))  # uniform planting, independent of the model
            s = E.aggregate(Pm, method, ref=ref if method == "sum" else None)
            ranks.append(E.planted_rank(s, planted, rng))
        cnt = np.bincount(ranks, minlength=M + 1)[1:]
        assert st.chisquare(cnt).pvalue > 1e-3, (method, cnt)


def test_refused_probes_give_uniform_rank():
    rng = np.random.default_rng(5)
    rows = [{"refused": True, "scores": [[0.0, 1]] * 10}] * 4
    P = E.probe_matrix(rows, 10)
    s = E.aggregate(P, "z")
    ranks = [E.planted_rank(s, 3, rng) for _ in range(20000)]
    cnt = np.bincount(ranks, minlength=11)[1:]
    assert st.chisquare(cnt).pvalue > 1e-3


def test_rank_aggregation_invariant_to_monotone_transform():
    rng = np.random.default_rng(6)
    P = rng.normal(size=(8, 10))
    a = E.aggregate(P, "rank")
    b = E.aggregate(np.exp(3 * P) + 7, "rank")
    assert np.allclose(a, b)
    # z aggregation invariant to per-probe affine maps (e.g. per-probe temperature / offsets)
    c = E.aggregate(P * rng.uniform(0.5, 2, (8, 1)) + rng.normal(size=(8, 1)), "z")
    assert np.allclose(E.aggregate(P, "z"), c)


def test_planted_rank_tiebreak():
    rng = np.random.default_rng(7)
    s = np.array([1.0, 1.0, 0.5])
    rs = [E.planted_rank(s, 0, rng) for _ in range(4000)]
    assert set(rs) == {1, 2} and abs(np.mean(rs) - 1.5) < 0.05
    assert E.planted_rank(np.array([0.0, 2.0, 1.0]), 0, rng) == 3


def test_clopper_pearson_coverage():
    rng = np.random.default_rng(8)
    p, n = 0.05, 250
    cover = 0
    for _ in range(2000):
        k = rng.binomial(n, p)
        lo, hi = E.clopper_pearson(k, n)
        cover += lo <= p <= hi
    assert cover / 2000 >= 0.94


def test_prop4_bound_on_synthetic_roc():
    rng = np.random.default_rng(9)
    for sep in [0.0, 0.5, 1.0, 2.0]:
        pos, neg = rng.normal(sep, 1, 400), rng.normal(0, 1, 400)
        fpr, tpr, _ = E.roc_points(pos, neg)
        auc = E.hull_auc(fpr, tpr)
        for s in [0.3, 0.5, 0.9]:
            assert E.min_fpr_at_tpr(fpr, tpr, s) >= E.prop4_bound(s, auc) - 1e-9


def test_hull_auc_ge_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(10)
    pos, neg = rng.normal(0.7, 1, 300), rng.normal(0, 1, 300)
    fpr, tpr, _ = E.roc_points(pos, neg)
    auc = roc_auc_score(np.r_[np.ones(300), np.zeros(300)], np.r_[pos, neg])
    assert E.hull_auc(fpr, tpr) >= auc - 1e-9


@pytest.mark.parametrize("strategy", E.STRATEGIES)
def test_vectorised_matches_scalar(strategy):
    rng = np.random.default_rng(12)
    R = rng.integers(1, 11, size=(200, 9))
    V = E.eprocess_paths_vec(R, 10, strategy)
    S = np.array([E.eprocess_path(r, 10, strategy) for r in R])
    assert np.allclose(V, S)
