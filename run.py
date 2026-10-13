import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import core, data

OUT = Path("reports")


def aa_false_positive_rate(n=4000, runs=1000, p=0.1, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(runs):
        a, b = data.simulate_conversions(n, n, p, 0.0, rng)
        hits += core.two_proportion_ztest(a.sum(), n, b.sum(), n).significant
    return hits / runs


def empirical_power(n, p=0.1, lift=0.2, runs=500, seed=1):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(runs):
        a, b = data.simulate_conversions(n, n, p, lift, rng)
        hits += core.two_proportion_ztest(a.sum(), n, b.sum(), n).significant
    return hits / runs


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    n_needed = core.sample_size_per_group(0.10, 0.02)
    power_at_needed = empirical_power(n_needed)
    fpr = aa_false_positive_rate()

    a, b = data.simulate_conversions(n_needed, n_needed, 0.10, 0.2, rng)
    ztest = core.two_proportion_ztest(a.sum(), n_needed, b.sum(), n_needed)

    ra, rb = data.simulate_revenue(6000, 6000, lift=0.15, rng=rng)
    welch = core.welch_ttest(ra, rb)
    boot = core.bootstrap_diff(ra, rb)

    x, y, g = data.simulate_with_covariate(4000, effect=2.0, rng=rng)
    raw = core.welch_ttest(y[g == 0], y[g == 1])
    y_adj = core.cuped_adjust(y, x)
    adj = core.welch_ttest(y_adj[g == 0], y_adj[g == 1])
    width_raw = raw.ci_high - raw.ci_low
    width_adj = adj.ci_high - adj.ci_low

    sizes = np.arange(500, 8001, 500)
    power_curve = [empirical_power(int(n), runs=200) for n in sizes]
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, power_curve, marker="o", color="black")
    plt.axhline(0.8, linestyle="--", color="grey")
    plt.axvline(n_needed, linestyle=":", color="grey")
    plt.xlabel("users per group")
    plt.ylabel("empirical power")
    plt.title("Power for a 20% relative lift on a 10% base rate")
    plt.tight_layout()
    plt.savefig(OUT / "power_curve.png", dpi=130)

    summary = {
        "sample_size_per_group": n_needed,
        "empirical_power_at_that_size": round(power_at_needed, 3),
        "aa_false_positive_rate": round(fpr, 3),
        "ztest_diff": round(ztest.diff, 4),
        "ztest_p_value": float(f"{ztest.p_value:.3g}"),
        "revenue_welch_p_value": float(f"{welch.p_value:.3g}"),
        "revenue_bootstrap_ci": [round(boot.ci_low, 2), round(boot.ci_high, 2)],
        "cuped_ci_width_reduction": round(1 - width_adj / width_raw, 3),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
