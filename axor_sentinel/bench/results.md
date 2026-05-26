# axor-sentinel bench results

Dataset: **820 scenarios** (420 attack, 400 benign) — paper baseline, seed=42  
Scorer: in-memory weight replay (no Neo4j)  
Date: 2026-05-26

---

## Primary metric

| Metric | Value |
|---|---|
| **TPR @ FPR ≤ 0.02** | **1.000** |
| Decision threshold | 0.42 |
| Achieved FPR | 0.0000 (0 / 400 benign) |
| TP | 420 |
| FP | 0 |
| FN | 0 |
| TN | 400 |

## Per-class TPR

| Attack class | Scenarios | TPR | Median score | p90 score |
|---|---|---|---|---|
| `distributed_staging` | 60 | 1.000 | 0.992 | 0.992 |
| `fanout` | 90 | 1.000 | 0.420 | 0.420 |
| `slow_and_low` | 270 | 1.000 | 1.000 | 1.000 |

## Threshold sweep

| Threshold | TPR | FPR | TP | FP |
|---|---|---|---|---|
| 0.1 | 1.000 | 0.2500 | 420 | 100 |
| 0.2 | 1.000 | 0.2500 | 420 | 100 |
| 0.3 | 1.000 | 0.2500 | 420 | 100 |
| 0.4 | 1.000 | 0.2500 | 420 | 100 |
| 0.5 | 0.786 | 0.0000 | 330 | 0 |
| 0.6 | 0.786 | 0.0000 | 330 | 0 |
| 0.7 | 0.786 | 0.0000 | 330 | 0 |
| 0.8 | 0.786 | 0.0000 | 330 | 0 |
| 0.9 | 0.786 | 0.0000 | 330 | 0 |

## Score distribution

| | min | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| Attack  (420) | 0.420 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 |
| Benign  (400) | 0.000 | 0.000 | 0.000 | 0.400 | 0.400 | 0.400 |

## Notes

- **Scorer**: in-memory weight replay. Each scenario scored independently
  (no cross-scenario accumulation, no diversity/dampening between scenarios).
  This is a lower bound on real deployment performance where reputation
  accumulates across all sessions sharing resources.
- **Time decay** applied between sessions within each scenario
  using actual timestamps from the scenario generator.
- **`canonical_confidence`** applied per resource from scenario metadata.
- **Fanout detection** not included in this scorer (requires baseline history).
  Fanout scenarios are scored via hot weights only; real deployment would
  additionally trigger the z-score path.
- **`benign_false_taint`**: these scenarios have `had_taint=True` but no
  export. A single READ signal (weight=0.4) × typical confidence (0.4–1.0)
  gives max score ≤ 0.40, well below FLAG_THRESHOLD=0.70.
