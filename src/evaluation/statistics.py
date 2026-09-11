import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional

def compute_confidence_interval_95(values: List[float]) -> Dict[str, float]:
    """
    Computes empirical mean, standard deviation, standard error, and 95% Student-t confidence interval.
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}

    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(n)
    
    # 95% critical value for Student-t distribution with (n-1) degrees of freedom
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * sem

    return {
        "mean": mean,
        "std": std,
        "sem": float(sem),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "ci95_margin": margin,
        "n": n
    }

def aggregate_seed_metrics(runs_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregates per-seed metrics across runs into mean, standard deviation, and 95% CI.
    """
    if not runs_metrics:
        return {}

    all_keys = runs_metrics[0].keys()
    aggregated = {}
    for key in all_keys:
        vals = [run[key] for run in runs_metrics if key in run and not np.isnan(run[key])]
        aggregated[key] = compute_confidence_interval_95(vals)
    return aggregated

def paired_comparison_test(
    baseline_values: List[float],
    proposed_values: List[float]
) -> Dict[str, Any]:
    """
    Performs statistically defensible paired comparison tests (Paired Student's t-test and Wilcoxon signed-rank).
    """
    b = np.array(baseline_values)
    p = np.array(proposed_values)
    assert len(b) == len(p), "Paired tests require identical sample sizes across matching seeds."

    diff = p - b
    mean_diff = float(np.mean(diff))
    
    if len(b) > 1 and np.std(diff) > 1e-12:
        t_stat, p_value_t = stats.ttest_rel(p, b)
        try:
            wilcoxon_stat, p_value_w = stats.wilcoxon(p, b)
        except Exception:
            wilcoxon_stat, p_value_w = None, None
    else:
        t_stat, p_value_t = 0.0, 1.0
        wilcoxon_stat, p_value_w = None, None

    return {
        "mean_difference": mean_diff,
        "relative_improvement_pct": float(-mean_diff / (np.mean(b) + 1e-12) * 100.0),
        "paired_t_statistic": float(t_stat) if t_stat is not None else None,
        "p_value_t_test": float(p_value_t) if p_value_t is not None else None,
        "is_statistically_significant_005": bool(p_value_t is not None and p_value_t < 0.05),
        "n_pairs": len(b)
    }
