"""Tests for the reputation-threshold calibration harness (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass

from axor_sentinel.bench.eval.calibration import calibrate_threshold


@dataclass
class _Scn:
    scenario_id: str
    label: str
    attack_class: str | None = None


def _dataset(attack_scores, benign_scores):
    scenarios, scores = [], {}
    for i, s in enumerate(attack_scores):
        sid = f"a{i}"
        scenarios.append(_Scn(sid, "ATTACK", "slow_and_low"))
        scores[sid] = s
    for i, s in enumerate(benign_scores):
        sid = f"b{i}"
        scenarios.append(_Scn(sid, "BENIGN"))
        scores[sid] = s
    return scenarios, scores


def test_perfectly_separable_gives_auc_one():
    scenarios, scores = _dataset(
        attack_scores=[0.9, 0.95, 1.0, 0.85],
        benign_scores=[0.0, 0.1, 0.2, 0.05],
    )
    report = calibrate_threshold(scenarios, scores, target_fpr=0.0)
    assert report.roc_auc == 1.0
    assert report.achieved_tpr == 1.0      # all attacks caught at 0 FPR
    assert report.achieved_fpr == 0.0
    assert 0.2 < report.selected_threshold <= 0.85


def test_threshold_respects_fpr_budget():
    # Overlap forces a trade-off; with target_fpr=0 no benign may be flagged.
    scenarios, scores = _dataset(
        attack_scores=[0.6, 0.7, 0.8],
        benign_scores=[0.0, 0.5, 0.55],
    )
    report = calibrate_threshold(scenarios, scores, target_fpr=0.0)
    assert report.achieved_fpr == 0.0
    assert report.selected_threshold > 0.55


def test_reports_current_threshold_performance():
    scenarios, scores = _dataset(
        attack_scores=[0.85, 0.9],
        benign_scores=[0.0, 0.81],   # one benign above the evaluated flag threshold
    )
    report = calibrate_threshold(scenarios, scores, current_threshold=0.8)
    # A flag threshold of 0.8 would false-positive on the 0.81 benign sample.
    assert report.current_threshold_fpr == 0.5
    assert report.current_threshold_tpr == 1.0


def test_degenerate_dataset_no_crash():
    scenarios, scores = _dataset(attack_scores=[0.9], benign_scores=[])
    report = calibrate_threshold(scenarios, scores)
    assert report.roc_auc == 0.0
    assert report.n_benign == 0
