import numpy as np


def simulate_conversions(n_a, n_b, p_a, lift=0.0, rng=None):
    """Binary outcomes for control (A) and variant (B); lift is relative."""
    rng = rng or np.random.default_rng()
    a = rng.random(n_a) < p_a
    b = rng.random(n_b) < p_a * (1 + lift)
    return a.astype(int), b.astype(int)


def simulate_revenue(n_a, n_b, mean=50.0, lift=0.0, rng=None):
    """Skewed per-user revenue with many zeros, as in a typical shop experiment."""
    rng = rng or np.random.default_rng()

    def draw(n, scale):
        buyers = rng.random(n) < 0.12
        return np.where(buyers, rng.lognormal(np.log(scale), 0.8, n), 0.0)

    return draw(n_a, mean), draw(n_b, mean * (1 + lift))


def simulate_with_covariate(n, effect=0.0, rho=0.7, rng=None):
    """Pre-experiment metric x and outcome y correlated with it (for CUPED)."""
    rng = rng or np.random.default_rng()
    x = rng.normal(100, 20, n)
    noise = rng.normal(0, 20 * np.sqrt(1 - rho**2), n)
    group = rng.integers(0, 2, n)
    y = 100 + rho * (x - 100) + noise + effect * group
    return x, y, group
