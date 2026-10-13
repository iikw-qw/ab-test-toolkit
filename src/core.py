from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    diff: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant(self):
        return self.p_value < 0.05


def two_proportion_ztest(x_a, n_a, x_b, n_b, alpha=0.05):
    """Pooled z-test for the difference of two proportions (B minus A)."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return TestResult(diff, diff - crit * se_unpooled, diff + crit * se_unpooled, p_value)


def welch_ttest(a, b, alpha=0.05):
    t = stats.ttest_ind(b, a, equal_var=False)
    diff = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    return TestResult(diff, diff - crit * se, diff + crit * se, t.pvalue)


def bootstrap_diff(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the difference of means (B minus A)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return TestResult(b.mean() - a.mean(), low, high, min(1.0, p))


def sample_size_per_group(p_base, mde_abs, alpha=0.05, power=0.8):
    """Sample size per group to detect an absolute lift in a conversion rate."""
    p2 = p_base + mde_abs
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    var = p_base * (1 - p_base) + p2 * (1 - p2)
    return int(np.ceil((z_a + z_b) ** 2 * var / mde_abs**2))


def srm_check(n_a, n_b, expected_ratio=0.5):
    """Sample ratio mismatch: chi-square test of the observed split."""
    total = n_a + n_b
    exp = [total * expected_ratio, total * (1 - expected_ratio)]
    return stats.chisquare([n_a, n_b], exp).pvalue


def cuped_adjust(y, x):
    """Variance reduction with a pre-experiment covariate."""
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())
