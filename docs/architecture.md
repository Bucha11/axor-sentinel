# axor-sentinel — Architecture

## Table of contents

1. [Overview](#1-overview)
2. [System context](#2-system-context)
3. [Package layout](#3-package-layout)
4. [Graph schema](#4-graph-schema)
5. [Weight model](#5-weight-model)
6. [Audit cycle](#6-audit-cycle)
7. [Snapshot subsystem](#7-snapshot-subsystem)
8. [Hot-path integration](#8-hot-path-integration)
9. [Attack patterns detected](#9-attack-patterns-detected)
10. [Invariants](#10-invariants)
11. [Bench suite](#11-bench-suite)

---

## 1. Overview

axor-sentinel is a background analysis layer for axor-core. It answers a question per-session enforcement cannot: *has this resource been systematically probed across many sessions over time?* The answer is fed to core as **observe-only telemetry** — it can tighten degradation at most, never deny.

It does this by maintaining a **resource reputation graph** in Neo4j and publishing a lightweight **reputation snapshot** that the hot path reads without any database I/O.

**Primary threat model:** slow-and-low staging — an agent (or coordinated group of agents) reads from a sensitive resource across many individually-normal sessions, then issues an export. No single session exceeds anomaly thresholds; the cross-session accumulation does.

**Secondary threat model:** fanout — an agent accesses an unusually large number of containers in a single session compared to its historical baseline, which may indicate exploratory pre-staging.

---

## 2. System context

```
┌─────────────────────────────────────────────────────┐
│  axor-core                                          │
│                                                     │
│  IntentLoop                                         │
│  ├─ IntentNormalizer   → NormalizedIntent           │
│  ├─ SnapshotIntentEnricher ← axor-sentinel         │
│  │   └─ populates target_resource_reputation        │
│  │      (telemetry — never gates)                   │
│  └─ DegradationEngine.record_detection (opt-in)     │
│      └─ reputation <= detection_floor → TIGHTEN     │
│         degradation (never deny)                    │
└─────────────────────────────────────────────────────┘
         ↑ reads snapshot (no Neo4j on hot path)
┌─────────────────────────────────────────────────────┐
│  axor-sentinel                                      │
│                                                     │
│  SentinelCycle (background, ~1h interval)           │
│  ├─ reads session traces from axor-core             │
│  ├─ applies weights to Neo4j graph                  │
│  └─ writes ReputationSnapshot atomically            │
│                                                     │
│  Neo4j  ←─────────────────────────────────────────  │
│  (resource graph with suspicion_score per resource) │
└─────────────────────────────────────────────────────┘
```

**Dependency direction:** axor-core never imports axor-sentinel. axor-core defines the `ReputationEnricher` protocol in `axor_core.contracts.reputation`; axor-sentinel implements it. The only enforcement consumer of the enriched fields is core's `DegradationEngine.record_detection` (opt-in, tightening-only).

---

## 3. Package layout

```
axor_sentinel/
├── graph/
│   ├── model.py          # Node/edge dataclasses, SignalType enum
│   ├── normalizer.py     # Resource ID normalization (3-tier)
│   └── queries.py        # Cypher query strings + runner functions
├── sentinel/
│   ├── events.py         # ReputationEvent, FanoutSignal, AgentContainerBaseline
│   ├── weight.py         # All weight math (pure Python, no I/O)
│   ├── snapshot.py       # ReputationSnapshot, atomic_swap, load_snapshot
│   └── cycle.py          # SentinelCycle — the audit loop
├── integration/
│   ├── intent_enricher.py  # SnapshotIntentEnricher (ReputationEnricher impl)
│   ├── core_sink.py        # CoreSessionSink (forward: ingest closed sessions)
│   └── probe_bridge.py     # ProbeTaintBridge (probe-flagged sessions)
├── reports/
│   └── slow_and_low.py   # SlowAndLowReport — wraps slow-and-low Cypher query
└── bench/
    ├── dataset/
    │   ├── schema.py     # Scenario, SessionRecord, GroundTruth dataclasses
    │   └── composer.py   # DatasetComposer — assembles 820-scenario paper baseline
    ├── eval/
    │   └── metrics.py    # evaluate() → EvaluationResult (TPR@FPR≤0.02)
    ├── scenarios/
    │   ├── attack.py     # slow_and_low, fanout, distributed_staging builders
    │   └── benign.py     # benign_narrow, benign_false_taint, benign_fanout_like
    ├── topology/
    │   ├── generator.py  # TopologyGenerator — synthetic resource graphs
    │   └── pool.py       # TopologyPool — 10 fixed-seed topologies
    ├── agents/
    │   └── profiles.py   # AgentProfile — 5 profiles (narrow/broad/noisy/etl/research)
    └── configs/
        └── paper_baseline.yaml  # 820-scenario composition table
```

---

## 4. Graph schema

### Nodes

| Label | Key fields | Notes |
|---|---|---|
| `:Resource` | `id`, `suspicion_score`, `flagged`, `last_signal_at`, `last_decay_at`, `normalization_method`, `canonical_confidence` | Core tracked entity. `last_signal_at` and `last_decay_at` are separate timestamps (invariant A-3). |
| `:Container` | `id`, `type`, `suspicion_score`, `flagged` | Directory / workspace / MCP namespace. Score = mean of member resources above threshold. |
| `:Agent` | `agent_id` | Stable identity across sessions. |
| `:Session` | `session_id`, `agent_id`, `started_at`, `had_taint`, `had_export_attempt`, `had_failed_export`, `had_escalation` | One governed session. |
| `:Destination` | `id`, `is_external` | Export target. |

### Edges

| Edge | Direction | Key properties | Notes |
|---|---|---|---|
| `:ACCESSED` | Session → Resource | `weight`, `signal_type`, `at` | Written per resource access. |
| `:ADJACENT_TO` | Resource → Resource | `topology_factor` | Static topology; computed once on resource discovery. |
| `:MEMBER_OF` | Resource → Container | — | Membership for container score aggregation. |
| `:IN_SESSION` | Agent → Session | — | Links agent to its sessions for cross-session queries. |
| `:EXPORTED_TO` | Session → Destination | — | Recorded on export attempt. |

### topology_factor values

| Relationship | Factor |
|---|---|
| Same directory / workspace | 1.0 |
| Same service / datasource | 0.7 |
| Same MCP namespace / policy | 0.6 |
| Same project / shared user | 0.5 |
| Different service | 0.4 |

---

## 5. Weight model

All weight math lives in `sentinel/weight.py` — pure Python, no I/O, fully testable.

### 5.1 Score accumulation

Logarithmic diminishing returns; score bounded to `[0, 1]`:

```
delta    = new_weight × (1 − current_score)
new_score = current_score + delta
```

Two weak signals never equal one strong signal. Score saturates as it approaches 1.

### 5.2 Hot signal weights

Weights encode involvement depth in a tainted session:

| `SignalType` | Raw weight | Trigger condition |
|---|---|---|
| `READ` | 0.4 | Read-only access |
| `READ_SUMMARIZE` | 0.6 | Read + LLM summarization |
| `READ_EXPORT_ADJACENT` | 0.8 | Read while another export was attempted |
| `READ_EXPORT_FAILED` | 1.0 | Read during failed export / escalation |

### 5.3 Effective weight (invariant A-8)

```
effective_weight = raw_weight
                 × canonical_confidence
                 × source_diversity_factor
                 × origin_dampening
```

**`canonical_confidence`** — reliability of the resource ID normalization:

| Method | Confidence |
|---|---|
| Provider object ID (SharePoint, OneDrive, …) | 1.0 |
| Normalized path | 0.7 |
| Heuristic fingerprint (filename + size + mtime) | 0.4 |

**`source_diversity_factor`** — reduces weight when signals are concentrated from one taint source. Full concentration (100% from one source) → 70% weight reduction:

```
concentration = signals_from_this_source / total_signals
factor        = 1.0 − (concentration × 0.7)
```

**`origin_dampening`** — exponential dampening for repeated signals from the same source:

```
factor = 0.5^prior_count    # 1st=1.0, 2nd=0.5, 3rd=0.25, …
```

### 5.4 Caution weight (adjacent resources)

Resources not directly accessed but topologically adjacent to a hot resource receive a caution weight:

```
caution = BASE_CAUTION × topology_factor × time_decay(days_since_last_decay) × canonical_confidence
BASE_CAUTION = 0.3
```

### 5.5 Time decay

Score halves every 30 days. Always computed against `last_decay_at`, never `last_signal_at` (invariant A-3):

```
decay_factor = 0.5^(days_since_last_decay / 30)
```

Applied in Neo4j at the start of every audit cycle (before hot weights) via `DECAY_QUERY`.

### 5.6 Flagging threshold

```
flagged = (suspicion_score >= FLAG_THRESHOLD)    # FLAG_THRESHOLD = 0.7
```

`flagged` is updated on every score change — never deferred (invariant A-2).

### 5.7 Container score

```
container_score = mean(member_scores where score > CONTAINER_MEMBER_THRESHOLD)
CONTAINER_MEMBER_THRESHOLD = 0.2
```

Recomputed after any member resource score change (invariant A-9).

### 5.8 Fanout weight (A-10)

When a fanout signal fires, all touched resources receive an additional flat accumulation:

```
score_after_fanout = accumulate(score_after_hot, FANOUT_WEIGHT)
FANOUT_WEIGHT = 0.5
```

Applied as a **separate** `accumulate()` call, not folded into `effective_weight`.

---

## 6. Audit cycle

`SentinelCycle.run_once()` in `sentinel/cycle.py`. Runs in the background (~1h interval), not in the hot path.

```
1. DECAY_QUERY           → Neo4j: decay all resources with non-zero score
                                   update last_decay_at; leave last_signal_at unchanged

2. For each tainted session:
   a. Fanout detection  → check z-score vs AgentContainerBaseline
                          guard: session.had_taint = True (A-15)
                          guard: session_count ≥ 10 (cold-start, A-14)
                          guard: max signal_type ≥ READ_SUMMARIZE (A-15)
                          z > 2.5 → emit FanoutSignal

   b. Hot weights       → per accessed resource:
                          compute effective_weight
                          accumulate in-memory score
                          if fanout: accumulate(score, 0.5) separately (A-10)
                          HOT_WEIGHT_QUERY → Neo4j (passes raw×diversity×dampening;
                          Cypher multiplies by canonical_confidence — A-8)
                          record ReputationEvent

   c. Caution weights   → CAUTION_ADJACENT_QUERY → Neo4j
                          (adjacent resources not directly accessed)

3. Container scores     → recompute for all affected containers

4. Snapshot swap        → ReputationSnapshot.with_checksum()
                          atomic_swap(snapshot_dir, snapshot)   ← A-5, A-16
```

### Fanout detection detail

```python
z_score = (unique_containers − baseline.mean) / baseline.std

# Special case: std ≈ 0 (very regular agent)
if std < 0.01:
    if unique_containers ≤ mean + FANOUT_MIN_DELTA(3):
        return None      # not a fanout
    z_score = inf        # clearly anomalous

# Trigger condition (all three required — A-15):
# 1. had_taint
# 2. z_score > 2.5
# 3. max signal_type ≥ READ_SUMMARIZE
```

Baseline is updated with exponential smoothing (α=0.3) after each session window of 50 sessions.

---

## 7. Snapshot subsystem

`sentinel/snapshot.py`

### ReputationSnapshot

```python
@dataclass(frozen=True)
class ReputationSnapshot:
    version: int
    generated_at: float
    resource_reputation: dict[str, float]   # resource_id → score
    container_reputation: dict[str, float]  # container_id → score
    checksum: str                           # SHA-256 of JSON body
```

### Atomic write (A-5, A-16)

Integrity guarantee: no reader ever sees a partial write.

```
POSIX:
  1. serialize → bytes
  2. write to snapshot_v{N}.json (temp)
  3. verify checksum of in-memory bytes (never re-reads from disk — A-5)
  4. os.symlink(new_file, snapshot_new)
  5. os.rename(snapshot_new, snapshot_current)    ← atomic on POSIX

Windows:
  1–3. same
  4. os.replace(new_file, snapshot_current)       ← atomic on Windows
```

### Load + checksum verification

```python
snapshot = load_snapshot(snapshot_dir)
# Returns None (with AuditIntegrityWarning) on checksum mismatch
# Never raises — fail-safe for hot path
```

### Network mount warning (A-17)

`validate_snapshot_dir(path)` emits `AuditIntegrityWarning` if the snapshot directory is on a network filesystem (`nfs`, `cifs`, `smb`, `nfs4`) detected via `/proc/mounts`.

---

## 8. Hot-path integration

### SnapshotIntentEnricher

Implements `axor_core.contracts.reputation.ReputationEnricher`. Called by `IntentLoop` after `IntentNormalizer.normalize()`; the enriched fields are telemetry consumed only by the opt-in degradation floor (never a gate).

```python
# The snapshot stores SUSPICION (high = bad); core's reputation field is TRUST
# (a positive reading <= detection_floor crosses and tightens; 0.0 = unknown).
# Convert at the boundary so core tightens on suspicious resources, not benign ones.
enriched = dataclasses.replace(
    normalized,
    target_resource_reputation=_suspicion_to_reputation(
        snapshot.resource_reputation.get(resource_id, 0.0)),
    target_container_reputation=_suspicion_to_reputation(
        snapshot.container_reputation.get(container_id, 0.0)),
)
```

**Key constraints:**
- Never queries Neo4j (A-6) — reads pre-loaded snapshot only
- Never raises — failures are logged, original intent returned
- Resource ID derived from intent args via `graph/normalizer.py` (same 3-tier normalization)
- Converts suspicion → trust polarity at the boundary (`reputation = 1 - suspicion`, clamped above 0)

### Observe-only coupling (reputation never denies)

In core's `IntentLoop`, after enrichment the reputation signal is recorded into the
degradation engine and nowhere else:

```python
if reputation_enricher:
    normalized = enricher.enrich(normalized, intent)
    if degradation_engine:                       # opt-in
        degradation_engine.record_detection(normalized)   # tightening-only
```

`record_detection` is a **no-op unless a `detection_floor` is configured**, and when
configured it can only *tighten* the session to RESTRICTED (monotone) — it never
returns an allow/deny and never loosens. The operator pairs the floor with the
suspicion flag threshold: `detection_floor = 1 - FLAG_THRESHOLD` (default `0.3`), so
a sentinel-flagged resource (suspicion `>= FLAG_THRESHOLD` → reputation `<= floor`)
crosses and tightens. There is no Phase-1 deterministic deny and no Layer-2 ML
handoff in core today; those were removed.

---

## 9. Attack patterns detected

### 9.1 Slow-and-low

**Pattern:** Single agent reads from sensitive resources across N sessions spread over `gap_days`, then issues an export.

**Detection mechanism:** Hot weights accumulate on the staging resources across sessions. By the time the export session arrives, the target resources have `suspicion_score >= FLAG_THRESHOLD`. The enricher then feeds core a reputation `<= detection_floor`, which (opt-in) tightens the export session's degradation level — narrowing its surface. It does not deny; core's per-session gates remain the only deny path.

**Config variants in bench:** 2, 4, 8 staging sessions × 7-day gap × 2 noise sessions interspersed.

### 9.2 Fanout

**Pattern:** Single agent accesses an unusually large number of containers in one session (broad reconnaissance), with `signal_type >= READ_SUMMARIZE`.

**Detection mechanism:** z-score of unique containers touched vs agent's historical baseline. Requires cold-start guard (≥10 sessions of history) to prevent false positives on new agents.

**Config variants in bench:** 3, 5, 10 containers.

### 9.3 Distributed staging

**Pattern:** Multiple agents with distinct `agent_id` values each stage small amounts of the same target resources. Reputation accumulates on the resource node regardless of which agent contributed the signal.

**Detection mechanism:** Resource-centric graph — `suspicion_score` lives on the `:Resource` node, not the `:Agent` node. Multiple agents accessing the same resource each increment its score.

---

## 10. Invariants

| ID | Statement | Where enforced |
|---|---|---|
| A-1 | `suspicion_score` ∈ [0, 1] always | `accumulate()` in `weight.py` |
| A-2 | `flagged` updated on every score change, never deferred | `update_resource_score()`, Cypher queries |
| A-3 | Decay uses `last_decay_at`; signal events use `last_signal_at`; they are never swapped | `DECAY_QUERY` updates `last_decay_at`; `HOT_WEIGHT_QUERY` updates `last_signal_at` |
| A-4 | Decay runs first in every audit cycle, before any hot weight | `cycle.py`: `apply_decay()` before session loop |
| A-5 | Checksum verified from in-memory bytes before snapshot is made visible | `atomic_swap()`: `_verify_checksum_bytes(serialized.encode(), checksum)` |
| A-6 | No Neo4j call on the hot path | `SnapshotIntentEnricher.enrich()` reads dict only |
| A-7 | `flagged` is never exposed as a feature on the intent | `NormalizedIntent` has no `flagged` field |
| A-8 | `effective_weight = raw × confidence × diversity × dampening` | `compute_effective_weight()` in `weight.py` |
| A-9 | Container score recomputed after every member score change | `cycle.run_once()` step 2f |
| A-10 | Fanout flat 0.5 applied as separate `accumulate()` after hot weight | `cycle.py` lines 172–174 |
| A-11 | Reputation is observe-only — it never denies | core `record_detection` is tightening-only, no deny path |
| A-12 | Reputation acts only via the opt-in degradation floor | `detection_floor` unset → `record_detection` is a no-op |
| A-13 | `origin_dampening` never zero, never > 1 | `0.5^n ∈ (0, 1]` for all n ≥ 0 |
| A-14 | Fanout detection disabled for agents with < 10 sessions | `_check_fanout()`: `baseline.session_count < FANOUT_MIN_SESSIONS → None` |
| A-15 | Fanout requires: `had_taint AND z > 2.5 AND signal ≥ READ_SUMMARIZE` | `_check_fanout()` checks all three |
| A-16 | Snapshot write is atomic: symlink rename (POSIX) / `os.replace` (Windows) | `atomic_swap()` in `snapshot.py` |
| A-17 | `validate_snapshot_dir()` warns if path is a network mount | `_warn_if_network_mount()` reads `/proc/mounts` |

---

## 11. Bench suite

### Dataset composition (paper baseline)

Total: **820 scenarios** — 420 attack, 400 benign. Seed: 42.

| Scenario type | Count | Notes |
|---|---|---|
| `slow_and_low_2` | 90 | 2 staging sessions |
| `slow_and_low_4` | 90 | 4 staging sessions |
| `slow_and_low_8` | 90 | 8 staging sessions |
| `fanout_3` | 30 | 3 containers |
| `fanout_5` | 30 | 5 containers |
| `fanout_10` | 30 | 10 containers |
| `distributed_staging` | 60 | 3 agents × 2 sessions each |
| `benign_narrow` | 150 | Normal narrow-access agent |
| `benign_broad_etl` | 150 | Wide-access (etl/broad), split equally |
| `benign_false_taint` | 100 | Reads suspicious file, no export |

### Evaluation metric

**TPR @ FPR ≤ 0.02** — true-positive rate at a reputation score threshold where false-positive rate does not exceed 2%.

```python
result = evaluate(scenarios, scores)
# result.tpr_at_fpr_budget  ← primary metric
# result.per_class_tpr      ← breakdown by attack class
# result.threshold          ← score threshold used
# result.fpr                ← achieved FPR at threshold
```

### Topology pool

10 pre-generated synthetic topologies (fixed seeds `i × 1337`). Each topology has:
- 3 services × 2 containers × 4 resources = 24 resources
- Same-container `ADJACENT_TO` edges (factor 1.0)
- Cross-service edges (factor 0.4)
- Cycled normalization methods to guarantee at least one of each tier per topology

### Agent profiles

| Profile | Mean containers/session | Std | Use case |
|---|---|---|---|
| `narrow` | 1.5 | 0.5 | Focused single-task agent |
| `broad` | 6.0 | 2.0 | Cross-service research agent |
| `noisy` | 4.0 | 3.5 | Unpredictable; tests false-positive rate |
| `etl` | 8.0 | 1.0 | Regular wide-access pipeline |
| `research` | 5.0 | 2.5 | Multi-source research |
