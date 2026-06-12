# axor-sentinel

Cross-session behavioral analysis layer for axor-core. Detects **slow-and-low staging attacks** — coordinated data exfiltration spread across many sessions over days or weeks — that are invisible to per-session anomaly detection.

## The problem it solves

Per-session anomaly detectors see each session in isolation. An attacker staging an exfiltration across dozens of short, individually normal sessions — touching sensitive resources a little each time, then exfiltrating later — never exceeds any single-session threshold.

axor-sentinel maintains a **resource reputation graph** across all sessions. Resources accumulate suspicion over time, and that suspicion is fed back to axor-core as **observe-only telemetry**: it never denies a request on its own. The only enforcement coupling is opt-in — when a resource's suspicion is high enough, core's `DegradationEngine` (configured with a `detection_floor`) can *tighten* the session's degradation level, narrowing the surface. Reputation is a signal that raises caution, not a gate.

## How it works

```
sessions (axor-core)
        │
        ▼
  SentinelCycle          ← background audit, runs every ~1h
  ├── time decay          ← scores halve every 30 days
  ├── hot weights         ← READ=0.4 / READ_SUMMARIZE=0.6 / EXPORT_ADJACENT=0.8 / EXPORT_FAILED=1.0
  ├── caution weights     ← adjacent resources get BASE_CAUTION=0.3 × topology_factor
  ├── fanout detection    ← z-score vs agent's historical baseline (fires at z > 2.5)
  └── snapshot swap       ← atomic write to disk (symlink rename / os.replace)
        │
        ▼
  ReputationSnapshot      ← flat dict resource_id → score, loaded at startup
        │
        ▼
  SnapshotIntentEnricher  ← converts suspicion → trust, populates
        │                    NormalizedIntent.target_resource_reputation (telemetry)
        ▼
  IntentLoop (axor-core)  ← observe-only: records the reputation signal; with an
                            opt-in detection_floor it can only TIGHTEN degradation,
                            never deny. (floor = 1 - flag_threshold)
```

> **Polarity note.** Sentinel scores *suspicion* (high = bad). Core's reputation
> field is *trust* (a positive reading `<=` `detection_floor` crosses and tightens;
> `0.0` = unknown). `SnapshotIntentEnricher` converts at the boundary
> (`reputation = 1 - suspicion`), so core tightens on suspicious resources.

Score accumulation uses **logarithmic diminishing returns** — each signal contributes proportionally to remaining headroom, so scores are bounded to `[0, 1]` and saturate naturally:

```
new_score = current + new_weight × (1 − current)
```

## Quick start

```bash
pip install -e ".[neo4j]"
```

```python
from pathlib import Path
from axor_sentinel.sentinel.cycle import SentinelCycle
from axor_sentinel.integration.intent_enricher import SnapshotIntentEnricher
from axor_sentinel.sentinel.weight import FLAG_THRESHOLD
from axor_core.degradation.engine import DegradationEngine

snapshot_dir = Path("~/.axor/sentinel").expanduser()

# Background audit cycle (runs hourly, not on hot path)
cycle = SentinelCycle(neo4j_session, snapshot_dir=snapshot_dir)
snapshot = cycle.run_once(sessions)

# Hot-path enrichment — pure dict lookup, no Neo4j
enricher = SnapshotIntentEnricher.from_dir(snapshot_dir)

# Wire into axor-core IntentLoop. Enrichment alone only populates telemetry; to act
# on it, give the loop a DegradationEngine with a detection_floor so a suspicious
# resource TIGHTENS degradation (never denies). The floor pairs with the suspicion
# flag threshold: floor = 1 - FLAG_THRESHOLD.
degradation = DegradationEngine(detection_floor=1.0 - FLAG_THRESHOLD)
loop = IntentLoop(..., reputation_enricher=enricher, degradation_engine=degradation)
```

## Bench suite

820-scenario synthetic dataset (paper baseline):

```bash
pip install -e ".[bench]"
```

```python
from axor_sentinel.bench.dataset.composer import DatasetComposer
from axor_sentinel.bench.eval.metrics import evaluate

scenarios = DatasetComposer().compose()        # 820 scenarios, seed=42
result = evaluate(scenarios, scores=my_scores)
print(result.tpr_at_fpr_budget)               # TPR @ FPR ≤ 0.02
```

---

See [docs/architecture.md](docs/architecture.md) for the full design: graph schema, weight model, invariants, attack patterns, and bench composition.
