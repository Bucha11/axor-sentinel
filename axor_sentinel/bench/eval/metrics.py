from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_sentinel.bench.dataset.schema import Scenario

# Maximum false-positive rate for TPR@FPR budget (spec §Evaluation)
FPR_BUDGET: float = 0.02


@dataclass
class EvaluationResult:
    """
    Evaluation metrics for a sentinel run over a benchmark dataset.

    Metrics follow spec §Evaluation:
      - TPR@FPR≤0.02 is the primary metric
      - fpr is the achieved false-positive rate at the decision threshold
      - threshold is the reputation score used for classification
    """
    tpr_at_fpr_budget: float          # true-positive rate at FPR ≤ FPR_BUDGET
    fpr: float                        # achieved false-positive rate at threshold
    threshold: float                  # score threshold used for classification
    tp: int                           # true positives
    fp: int                           # false positives
    fn: int                           # false negatives
    tn: int                           # true negatives
    n_attack: int                     # total attack scenarios in dataset
    n_benign: int                     # total benign scenarios in dataset
    per_class_tpr: dict[str, float] = field(default_factory=dict)
    """TPR broken down by attack_class (e.g. 'slow_and_low', 'fanout')."""


def evaluate(
    scenarios: list["Scenario"],
    scores: dict[str, float],
    fpr_budget: float = FPR_BUDGET,
) -> EvaluationResult:
    """
    Compute TPR@FPR≤fpr_budget given per-scenario reputation scores.

    Parameters
    ----------
    scenarios:
        List of Scenario objects (from DatasetComposer or bench dataset).
    scores:
        Mapping scenario_id → reputation score (float in [0, 1]).
        Scenarios missing from scores are treated as score=0.0.
    fpr_budget:
        Maximum acceptable false-positive rate (default 0.02 per spec).

    Returns
    -------
    EvaluationResult with primary metric ``tpr_at_fpr_budget``.

    Algorithm
    ---------
    1. Gather (score, is_attack) pairs for all scenarios.
    2. Sort candidate thresholds by descending score.
    3. Find the lowest threshold where FPR ≤ fpr_budget.
    4. Compute TPR at that threshold.
    """
    attack_scenarios = [s for s in scenarios if s.label == "ATTACK"]
    benign_scenarios = [s for s in scenarios if s.label == "BENIGN"]

    n_attack = len(attack_scenarios)
    n_benign = len(benign_scenarios)

    if n_attack == 0 or n_benign == 0:
        return EvaluationResult(
            tpr_at_fpr_budget=0.0,
            fpr=0.0,
            threshold=1.0,
            tp=0, fp=0, fn=n_attack, tn=n_benign,
            n_attack=n_attack, n_benign=n_benign,
        )

    # Collect (score, is_attack, attack_class)
    labeled: list[tuple[float, bool, str | None]] = []
    for s in scenarios:
        sc = scores.get(s.scenario_id, 0.0)
        labeled.append((sc, s.label == "ATTACK", s.attack_class))

    # Unique thresholds — add 0.0 to cover the "flag everything" end
    unique_thresholds = sorted({sc for sc, _, _ in labeled}, reverse=True) + [0.0]

    best_threshold = 1.0
    best_tpr = 0.0
    best_fpr = 0.0

    for thresh in unique_thresholds:
        fp = sum(1 for sc, is_atk, _ in labeled if sc >= thresh and not is_atk)
        tp = sum(1 for sc, is_atk, _ in labeled if sc >= thresh and is_atk)
        fpr = fp / n_benign
        tpr = tp / n_attack
        if fpr <= fpr_budget:
            if tpr > best_tpr or (tpr == best_tpr and fpr < best_fpr):
                best_tpr = tpr
                best_fpr = fpr
                best_threshold = thresh

    # Counts at chosen threshold
    tp = sum(1 for sc, is_atk, _ in labeled if sc >= best_threshold and is_atk)
    fp = sum(1 for sc, is_atk, _ in labeled if sc >= best_threshold and not is_atk)
    fn = n_attack - tp
    tn = n_benign - fp

    # Per-class TPR
    classes: dict[str, list[tuple[float, bool]]] = {}
    for sc, is_atk, cls in labeled:
        if is_atk and cls:
            classes.setdefault(cls, []).append((sc, is_atk))

    per_class_tpr: dict[str, float] = {}
    for cls, items in classes.items():
        cls_tp = sum(1 for sc, _ in items if sc >= best_threshold)
        per_class_tpr[cls] = cls_tp / len(items) if items else 0.0

    return EvaluationResult(
        tpr_at_fpr_budget=best_tpr,
        fpr=best_fpr,
        threshold=best_threshold,
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_attack=n_attack, n_benign=n_benign,
        per_class_tpr=per_class_tpr,
    )
