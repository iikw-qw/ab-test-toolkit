import numpy as np
import pytest

from src import core, data


def test_sample_size_matches_textbook_value():
    n = core.sample_size_per_group(0.10, 0.02)
    assert 3800 <= n <= 3900


def test_ztest_detects_obvious_difference():
    res = core.two_proportion_ztest(100, 1000, 160, 1000)
    assert res.significant
    assert res.diff == pytest.approx(0.06)
    assert res.ci_low < res.diff < res.ci_high


def test_ztest_equal_groups_not_significant():
    assert not core.two_proportion_ztest(100, 1000, 100, 1000).significant


def test_welch_matches_scipy_direction():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0.5, 1, 500)
    res = core.welch_ttest(a, b)
    assert res.diff > 0 and res.significant


def test_bootstrap_interval_contains_diff():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 2, 300)
    b = rng.normal(10.4, 2, 300)
    res = core.bootstrap_diff(a, b, n_boot=500)
    assert res.ci_low < res.diff < res.ci_high


def test_srm_flags_broken_split():
    assert core.srm_check(5000, 5000) > 0.5
    assert core.srm_check(5300, 4700) < 0.01


def test_cuped_reduces_variance_and_keeps_mean():
    x, y, _ = data.simulate_with_covariate(5000, rng=np.random.default_rng(2))
    adj = core.cuped_adjust(y, x)
    assert adj.var() < y.var()
    assert adj.mean() == pytest.approx(y.mean(), abs=1e-6)
