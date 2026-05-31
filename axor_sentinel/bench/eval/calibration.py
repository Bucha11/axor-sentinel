"""Threshold calibration for the reputation gate.

The Phase-1 deterministic deny in axor-core fires at a *fixed*
REPUTATION_RULE_THRESHOLD (0.8). That constant is only defensible if it has a
measured false-positive rate on representative data. This module turns a labeled
scenario set + per-scenario scores into a full ROC / PR curve, AUCs, and a
threshold selected for a target FPR — i.e. it replaces a guessed constant with a
data-derived one.

The math here is exact and unit-tested. The *quality* of the calibration is only
as good as the input dataset: feeding the synthetic bench set gives a synthetic
calibration; production calibration requires real labeled traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from axor_sentinel.bench.dataset.schema import Scenario


@dataclass(frozen=True)
class ROCPoint:
    threshold: float
    fpr: float
    tpr: float


@dataclass
class CalibrationReport:
    """Result of calibrating the reputation threshold against labeled data."""
    selected_threshold: float
    achieved_fpr: float
    achieved_tpr: float
    target_fpr: float
    roc_auc: float
    pr_auc: float
    n_attack: int
    n_benign: int
    roc_curve: list[ROCPoint] = field(default_factory=list)
    # Comparison against the constant currently shipped in axor-core.
    current_threshold: float = 0.8
    current_threshold_fpr: float = 0.0
    current_threshold_tpr: float = 0.0

    def summary(self) -> str:
        return (
            f"calibrated threshold={self.selected_threshold:.3f} "
            f"(target_fpr<={self.target_fpr:.3f}) → "
            f"fpr={self.achieved_fpr:.3f} tpr={self.achieved_tpr:.3f} | "
            f"roc_auc={self.roc_auc:.3f} pr_auc={self.pr_auc:.3f} | "
            f"current={self.current_threshold:.2f} "
            f"(fpr={self.current_threshold_fpr:.3f} tpr={self.current_threshold_tpr:.3f})"
        )


def _confusion(labeled: Sequence[tuple[float, bool]], thresh: float) -> tuple[int, int, int, int]:
    tp = sum(1 for sc, atk in labeled if sc >= thresh and atk)
    fp = sum(1 for sc, atk in labeled if sc >= thresh and not atk)
    fn = sum(1 for sc, atk in labeled if sc < thresh and atk)
    tn = sum(1 for sc, atk in labeled if sc < thresh and not atk)
    return tp, fp, fn, tn


def _roc_curve(labeled: Sequence[tuple[float, bool]]) -> list[ROCPoint]:
    n_attack = sum(1 for _, atk in labeled if atk)
    n_benign = sum(1 for _, atk in labeled if not atk)
    if n_attack == 0 or n_benign == 0:
        return []
    thresholds = sorted({sc for sc, _ in labeled}, reverse=True)
    # Prepend a threshold above the max (flag nothing) and append below min (flag all).
    thresholds = [thresholds[0] + 1e-9] + thresholds + [0.0]
    points: list[ROCPoint] = []
    for t in thresholds:
        tp, fp, _, _ = _confusion(labeled, t)
        points.append(ROCPoint(threshold=t, fpr=fp / n_benign, tpr=tp / n_attack))
    return points


def _auc(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal AUC; xs need not be pre-sorted."""
    pairs = sorted(zip(xs, ys))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def calibrate_threshold(
    scenarios: list["Scenario"],
    scores: dict[str, float],
    target_fpr: float = 0.02,
    current_threshold: float = 0.8,
) -> CalibrationReport:
    """Derive a reputation threshold meeting ``target_fpr`` from labeled scenarios.

    scores maps scenario_id → reputation score in [0, 1]; missing → 0.0.
    Picks the lowest threshold whose FPR ≤ target_fpr (maximising TPR), and
    reports ROC/PR AUC plus how the currently-shipped constant performs.
    """
    labeled = [(scores.get(s.scenario_id, 0.0), s.label == "ATTACK") for s in scenarios]
    n_attack = sum(1 for _, atk in labeled if atk)
    n_benign = sum(1 for _, atk in labeled if not atk)

    roc = _roc_curve(labeled)
    roc_auc = _auc([p.fpr for p in roc], [p.tpr for p in roc]) if roc else 0.0

    # PR curve points (recall=tpr, precision) for AUC.
    pr_xs, pr_ys = [], []
    for p in roc:
        tp, fp, _, _ = _confusion(labeled, p.threshold)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        pr_xs.append(p.tpr)
        pr_ys.append(precision)
    pr_auc = _auc(pr_xs, pr_ys) if pr_xs else 0.0

    # Threshold selection: lowest threshold with fpr <= target_fpr, max tpr.
    selected = 1.0
    sel_fpr = 0.0
    sel_tpr = 0.0
    for p in roc:
        if p.fpr <= target_fpr and (
            p.tpr > sel_tpr or (p.tpr == sel_tpr and p.fpr < sel_fpr)
        ):
            selected, sel_fpr, sel_tpr = p.threshold, p.fpr, p.tpr

    cur_tp, cur_fp, _, _ = _confusion(labeled, current_threshold)
    cur_fpr = cur_fp / n_benign if n_benign else 0.0
    cur_tpr = cur_tp / n_attack if n_attack else 0.0

    return CalibrationReport(
        selected_threshold=selected,
        achieved_fpr=sel_fpr,
        achieved_tpr=sel_tpr,
        target_fpr=target_fpr,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        n_attack=n_attack,
        n_benign=n_benign,
        roc_curve=roc,
        current_threshold=current_threshold,
        current_threshold_fpr=cur_fpr,
        current_threshold_tpr=cur_tpr,
    )
