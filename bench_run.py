"""
In-memory bench run — no Neo4j required.

Replays each scenario's sessions through the weight model and computes
a per-scenario score (max resource suspicion_score at end of scenario).
Then runs evaluate() and prints results.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure package is on path when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from axor_sentinel.bench.dataset.composer import DatasetComposer
from axor_sentinel.bench.dataset.schema import Scenario
from axor_sentinel.bench.eval.metrics import evaluate, FPR_BUDGET
from axor_sentinel.graph.model import SignalType, HOT_WEIGHTS
from axor_sentinel.sentinel.weight import (
    accumulate,
    apply_decay_to_score,
)
from axor_sentinel.sentinel.evidence import Evidence, resolution_from_confidence
from axor_sentinel.sentinel.predicates import (
    ReputationLevel,
    SentinelPolicy,
    evaluate_resource,
)


# ── Deterministic predicate scorer (unfitted — declared policy constants) ──────

def level_scenario(scenario: Scenario, policy: SentinelPolicy) -> bool:
    """Replay a scenario through the deterministic verdict layer.

    Returns True iff any resource ends the scenario FLAGGED. Unlike the score
    path there is nothing to threshold: the predicates are declared, so this
    is a fixed classifier — the bench MEASURES its FPR instead of tuning it.
    """
    conf = {r.resource_id: r.canonical_confidence for r in scenario.resources}
    store: dict[str, list[Evidence]] = defaultdict(list)
    last_t = 0.0

    for session in sorted(scenario.sessions, key=lambda s: s.started_at):
        last_t = max(last_t, session.started_at)
        if not session.had_taint:
            continue
        for access in session.accessed_resources:
            try:
                rank = SignalType(access.signal_type)
            except ValueError:
                continue
            rid = access.resource_id
            store[rid].append(Evidence(
                origin=session.agent_id,
                session_id=session.session_id,
                rank=rank,
                tainted=True,
                observed_at=session.started_at,
                resolution=resolution_from_confidence(conf.get(rid, 0.4)),
            ))

    now = last_t + 1.0
    return any(
        evaluate_resource(evs, policy, now).level == ReputationLevel.FLAGGED
        for evs in store.values()
    )


# ── In-memory scorer ───────────────────────────────────────────────────────────

def score_scenario(scenario: Scenario) -> float:
    """
    Replay a scenario's sessions and return max resource suspicion_score.

    - Sessions processed in chronological order.
    - Time decay applied between sessions (30-day half-life).
    - For tainted sessions: effective_weight applied per accessed resource.
    - No diversity/dampening: each scenario is independent (clean slate).
    - canonical_confidence looked up from scenario.resources.
    """
    # Build resource_id → canonical_confidence map
    conf: dict[str, float] = {
        r.resource_id: r.canonical_confidence for r in scenario.resources
    }

    scores: dict[str, float] = defaultdict(float)
    last_decay_at: dict[str, float] = {}
    last_seen_at: float | None = None

    for session in sorted(scenario.sessions, key=lambda s: s.started_at):
        t = session.started_at

        # Apply time decay since last audit to all resources with non-zero score
        if last_seen_at is not None:
            days = (t - last_seen_at) / 86400.0
            if days > 0:
                for rid in list(scores.keys()):
                    days_since = (t - last_decay_at.get(rid, t)) / 86400.0
                    scores[rid] = apply_decay_to_score(scores[rid], days_since)
                    last_decay_at[rid] = t

        last_seen_at = t

        if not session.had_taint:
            continue

        for access in session.accessed_resources:
            rid = access.resource_id
            try:
                signal = SignalType(access.signal_type)
            except ValueError:
                continue
            raw = HOT_WEIGHTS[signal]
            canonical = conf.get(rid, 0.4)
            # No diversity/dampening for isolated scenario replay
            eff = raw * canonical
            scores[rid] = accumulate(scores.get(rid, 0.0), eff)

    return max(scores.values(), default=0.0)


def run_bench() -> tuple[dict[str, float], list[Scenario]]:
    t0 = time.time()
    scenarios = DatasetComposer().compose()
    t1 = time.time()

    scores: dict[str, float] = {}
    for s in scenarios:
        scores[s.scenario_id] = score_scenario(s)
    t2 = time.time()

    print(f"composed {len(scenarios)} scenarios in {t1-t0:.2f}s", flush=True)
    print(f"scored   {len(scenarios)} scenarios in {t2-t1:.2f}s", flush=True)
    return scores, scenarios



# ── Predicate vs calibrated-score comparison ───────────────────────────────────

def compare_predicates(scenarios: list[Scenario], scores: dict[str, float]) -> None:
    policy = SentinelPolicy()
    flagged = {s.scenario_id: level_scenario(s, policy) for s in scenarios}

    def _rates(pred: dict[str, bool]) -> tuple[float, float, int, int]:
        atk = [s for s in scenarios if s.label == "ATTACK"]
        ben = [s for s in scenarios if s.label == "BENIGN"]
        tp = sum(1 for s in atk if pred[s.scenario_id])
        fp = sum(1 for s in ben if pred[s.scenario_id])
        return tp / len(atk), fp / len(ben), tp, fp

    tpr_p, fpr_p, tp_p, fp_p = _rates(flagged)
    score_flag = {sid: sc >= 0.7 for sid, sc in scores.items()}
    tpr_s, fpr_s, tp_s, fp_s = _rates(score_flag)

    print("\n── predicates (declared, unfitted) vs score (calibrated 0.7) ──")
    print(f"{'classifier':28} {'TPR':>7} {'FPR':>7} {'TP':>5} {'FP':>5}")
    print(f"{'score >= 0.7 (calibrated)':28} {tpr_s:7.3f} {fpr_s:7.3f} {tp_s:5d} {fp_s:5d}")
    print(f"{'predicates P1-P4 (declared)':28} {tpr_p:7.3f} {fpr_p:7.3f} {tp_p:5d} {fp_p:5d}")

    # per attack class
    classes = sorted({s.attack_class for s in scenarios if s.attack_class})
    print(f"\n{'attack_class':24} {'n':>4} {'pred TPR':>9} {'score TPR':>10}")
    for cls in classes:
        subset = [s for s in scenarios if s.attack_class == cls]
        tpr_pc = sum(1 for s in subset if flagged[s.scenario_id]) / len(subset)
        tpr_sc = sum(1 for s in subset if score_flag[s.scenario_id]) / len(subset)
        print(f"{cls:24} {len(subset):4d} {tpr_pc:9.3f} {tpr_sc:10.3f}")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scores, scenarios = run_bench()
    result = evaluate(scenarios, scores)
    compare_predicates(scenarios, scores)

    # Per-threshold sweep for table
    labeled = [
        (scores[s.scenario_id], s.label == "ATTACK", s.attack_class)
        for s in scenarios
    ]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_attack = sum(1 for s in scenarios if s.label == "ATTACK")
    n_benign = sum(1 for s in scenarios if s.label == "BENIGN")

    rows = []
    for th in thresholds:
        tp = sum(1 for sc, is_atk, _ in labeled if sc >= th and is_atk)
        fp = sum(1 for sc, is_atk, _ in labeled if sc >= th and not is_atk)
        tpr = tp / n_attack if n_attack else 0.0
        fpr = fp / n_benign if n_benign else 0.0
        rows.append((th, tpr, fpr, tp, fp))

    # Score distribution stats
    attack_scores = sorted([scores[s.scenario_id] for s in scenarios if s.label == "ATTACK"])
    benign_scores = sorted([scores[s.scenario_id] for s in scenarios if s.label == "BENIGN"])

    def pct(lst: list, p: float) -> float:
        if not lst:
            return 0.0
        i = int(len(lst) * p)
        return lst[min(i, len(lst) - 1)]

    # Per-class score stats
    class_scores: dict[str, list[float]] = defaultdict(list)
    for s in scenarios:
        if s.label == "ATTACK":
            class_scores[s.attack_class or "unknown"].append(scores[s.scenario_id])

    # Print to stdout for reference
    print(f"\nTPR @ FPR≤{FPR_BUDGET}: {result.tpr_at_fpr_budget:.3f}")
    print(f"threshold:             {result.threshold:.2f}")
    print(f"achieved FPR:          {result.fpr:.4f}")
    print(f"TP/FP/FN/TN:          {result.tp}/{result.fp}/{result.fn}/{result.tn}")
    print(f"per-class TPR:         {result.per_class_tpr}")

    # ── Write results.md ──────────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# axor-sentinel bench results")
    md_lines.append("")
    md_lines.append("Dataset: **820 scenarios** (420 attack, 400 benign) — paper baseline, seed=42  ")
    md_lines.append("Scorer: in-memory weight replay (no Neo4j)  ")
    md_lines.append(f"Date: {time.strftime('%Y-%m-%d')}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # Primary metric
    md_lines.append("## Primary metric")
    md_lines.append("")
    md_lines.append("| Metric | Value |")
    md_lines.append("|---|---|")
    md_lines.append(f"| **TPR @ FPR ≤ 0.02** | **{result.tpr_at_fpr_budget:.3f}** |")
    md_lines.append(f"| Decision threshold | {result.threshold:.2f} |")
    md_lines.append(f"| Achieved FPR | {result.fpr:.4f} ({result.fp} / {n_benign} benign) |")
    md_lines.append(f"| TP | {result.tp} |")
    md_lines.append(f"| FP | {result.fp} |")
    md_lines.append(f"| FN | {result.fn} |")
    md_lines.append(f"| TN | {result.tn} |")
    md_lines.append("")

    # Per-class TPR
    md_lines.append("## Per-class TPR")
    md_lines.append("")
    md_lines.append("| Attack class | Scenarios | TPR | Median score | p90 score |")
    md_lines.append("|---|---|---|---|---|")
    for cls, tpr in sorted(result.per_class_tpr.items()):
        cs = sorted(class_scores[cls])
        n = len(cs)
        med = pct(cs, 0.5)
        p90 = pct(cs, 0.9)
        md_lines.append(f"| `{cls}` | {n} | {tpr:.3f} | {med:.3f} | {p90:.3f} |")
    md_lines.append("")

    # Threshold sweep
    md_lines.append("## Threshold sweep")
    md_lines.append("")
    md_lines.append("| Threshold | TPR | FPR | TP | FP |")
    md_lines.append("|---|---|---|---|---|")
    for th, tpr, fpr, tp, fp in rows:
        flag = " ← **selected**" if abs(th - result.threshold) < 0.001 else ""
        md_lines.append(f"| {th:.1f} | {tpr:.3f} | {fpr:.4f} | {tp} | {fp} |{flag}")
    md_lines.append("")

    # Score distribution
    md_lines.append("## Score distribution")
    md_lines.append("")
    md_lines.append("| | min | p25 | p50 | p75 | p90 | max |")
    md_lines.append("|---|---|---|---|---|---|---|")
    a = attack_scores
    b = benign_scores
    md_lines.append(
        f"| Attack  ({len(a)}) | {a[0]:.3f} | {pct(a,0.25):.3f} | {pct(a,0.5):.3f} | {pct(a,0.75):.3f} | {pct(a,0.9):.3f} | {a[-1]:.3f} |"
    )
    md_lines.append(
        f"| Benign  ({len(b)}) | {b[0]:.3f} | {pct(b,0.25):.3f} | {pct(b,0.5):.3f} | {pct(b,0.75):.3f} | {pct(b,0.9):.3f} | {b[-1]:.3f} |"
    )
    md_lines.append("")

    # Notes
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- **Scorer**: in-memory weight replay. Each scenario scored independently")
    md_lines.append("  (no cross-scenario accumulation, no diversity/dampening between scenarios).")
    md_lines.append("  This is a lower bound on real deployment performance where reputation")
    md_lines.append("  accumulates across all sessions sharing resources.")
    md_lines.append("- **Time decay** applied between sessions within each scenario")
    md_lines.append("  using actual timestamps from the scenario generator.")
    md_lines.append("- **`canonical_confidence`** applied per resource from scenario metadata.")
    md_lines.append("- **Fanout detection** not included in this scorer (requires baseline history).")
    md_lines.append("  Fanout scenarios are scored via hot weights only; real deployment would")
    md_lines.append("  additionally trigger the z-score path.")
    md_lines.append("- **`benign_false_taint`**: these scenarios have `had_taint=True` but no")
    md_lines.append("  export. A single READ signal (weight=0.4) × typical confidence (0.4–1.0)")
    md_lines.append("  gives max score ≤ 0.40, well below FLAG_THRESHOLD=0.70.")

    out = Path(__file__).parent / "axor_sentinel" / "bench" / "results.md"
    out.write_text("\n".join(md_lines) + "\n")
    print(f"\nwrote {out}")
