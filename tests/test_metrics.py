import numpy as np
import pandas as pd

from pedwri.metrics import (
    binary_metrics, bootstrap_ci, froc, match_lesions, sensitivity_at_fppi, threshold_at_specificity, wilson,
)


def test_threshold_reaches_target_specificity():
    neg = np.linspace(0.05, 0.95, 10)
    t = threshold_at_specificity(neg, 0.9)
    assert (neg <= t).mean() >= 0.9
    assert (neg <= t - 1e-9).mean() < 0.9


def test_threshold_with_ties_at_zero():
    neg = np.r_[np.zeros(95), np.full(5, 0.7)]
    t = threshold_at_specificity(neg, 0.9)
    assert t == 0.0
    m = binary_metrics(np.r_[np.zeros(100), np.ones(3)], np.r_[neg, [0.0, 0.2, 0.9]], t)
    assert m["specificity"] == 0.95
    assert m["tp"] == 2 and m["fn"] == 1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 2)
    y = rng.integers(0, 2, 400)
    s = y * 0.3 + rng.random(400)
    ci = bootstrap_ci(groups, lambda i: binary_metrics(y[i], s[i], 0.6), n_boot=300)
    for k in ("sensitivity", "specificity"):
        assert ci[k]["ci_low"] <= ci[k]["value"] <= ci[k]["ci_high"]


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert 0.23 < lo < 0.24 and 0.76 < hi < 0.77


def test_lesion_matching_and_froc():
    gt = pd.DataFrame([("a", 3, 0.5, 0.5, 0.2, 0.2), ("b", 3, 0.3, 0.3, 0.1, 0.1)],
                      columns=["filestem", "cls", "cx", "cy", "w", "h"])
    preds = pd.DataFrame([
        ("a", 3, 0.9, 0.40, 0.40, 0.60, 0.60),
        ("a", 3, 0.8, 0.41, 0.41, 0.61, 0.61),
        ("b", 3, 0.4, 0.70, 0.70, 0.80, 0.80),
    ], columns=["filestem", "cls", "conf", "x1", "y1", "x2", "y2"])
    lesions, fps = match_lesions(["a", "b", "c"], gt, preds, 3, iou_thr=0.3)
    assert lesions.loc[lesions["filestem"] == "a", "hit_score"].item() == 0.9
    assert lesions.loc[lesions["filestem"] == "b", "hit_score"].item() == -1.0
    assert sorted(fps) == [0.4, 0.8]
    curve = froc(lesions["hit_score"].to_numpy(), fps, n_images=3)
    assert sensitivity_at_fppi(curve, 0.0) == 0.5
    assert curve["sensitivity"].max() == 0.5
