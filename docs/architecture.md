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

**Secondary threat model:** fanout — a tainted session touches more distinct containers at rank ≥ `READ_SUMMARIZE` than a declared per-actor-class quota allows (`SentinelPolicy.fanout_containers`, default 7), which may indicate exploratory pre-staging.

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
│   ├── normalizer.py     # Resource ID normalization (3-tier, lexical — §4b)
│   ├── derive.py         # tool call → identity (shared by enricher + core_sink)
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
| `:Session` | `session_id`, `agent_id`, `started_at`, `had_taint`, `had_export_attempt`, `had_failed_export`, `had_escalation` | One governed session. Upserted as a union across cycles: the `had_*` flags OR together and `started_at` only moves earlier, so a later, thinner record for the same session (e.g. the probe bridge's drift summary) never clears a flag. |
| `:Destination` | `id`, `is_external` | Export target. |

### Edges

| Edge | Direction | Key properties | Notes |
|---|---|---|---|
| `:ACCESSED` | Session → Resource | `weight`, `signal_type`, `at` | Written per resource access. |
| `:ADJACENT_TO` | Resource → Resource | `topology_factor` | Static topology; computed once on resource discovery. |
| `:MEMBER_OF` | Resource → Container | — | Membership for container score aggregation. |
| `:IN_SESSION` | Agent → Session | — | Links agent to its sessions for cross-session queries. |
| `:EXPORTED_TO` | Session → Destination | — | Recorded on export attempt. |

As implemented, the sentinel writes `:Agent`, `:Session`, `:Resource` and
`:Attestation` nodes and `:IN_SESSION` / `:ACCESSED` / `:ADJACENT_TO` /
`:ATTESTS` edges; containers exist only as `ADJACENT_TO` edges derived from
co-membership (no `:Container` node). `SentinelCycle.__init__` runs
`construct.ensure_schema`, which idempotently creates uniqueness constraints
(`IF NOT EXISTS`) on `Resource.id`, `Session.session_id`, `Agent.agent_id` and
`Attestation.attestation_id` — MERGE becomes an index lookup and concurrent
writers cannot duplicate a node. On a server that rejects the statements
(pre-4.4 syntax, read replica, duplicate data) the failure is logged and the
cycle runs without them.

### topology_factor values

| Relationship | Factor |
|---|---|
| Same directory / workspace | 1.0 |
| Same service / datasource | 0.7 |
| Same MCP namespace / policy | 0.6 |
| Same project / shared user | 0.5 |
| Different service | 0.4 |

---

## 4a. Deterministic verdict layer (authoritative)

Since the deterministic rework, the VERDICT path is `sentinel/evidence.py` +
`sentinel/predicates.py`: windowed typed evidence sets per resource, and
declared decidable predicates over them —

| Predicate | Fires when | Constant |
|---|---|---|
| P1 export-denied | an `EXPORT_FAILED` fact exists | — |
| P2 export adjacency | `EXPORT_ADJACENT` facts from distinct origins | `export_adjacent_origins = 2` |
| P3 staging count | distinct tainted sessions at rank ≥ READ_SUMMARIZE | `staging_sessions = 3` |
| P4 staging sequence | read-under-taint, then a later export-adjacent touch | — |
| fanout quota | > quota distinct containers in one tainted session | `fanout_containers = 7`, per-class overrides |

Levels form a monotone lattice `CLEAN < WATCH < FLAGGED`; heuristic-resolved
resource ids cap at WATCH; container verdicts count flagged members
(`container_flagged_members = 2`); adjacency to a FLAGGED resource is a WATCH
label. The snapshot's `resource_reputation` carries `LEVEL_SUSPICION[level]`
(finite codomain 0.0 / 0.4 / 1.0) so core's `detection_floor` comparison is
decidable end-to-end. Every constant is DECLARED policy (`SentinelPolicy`) —
semantic and auditable, never fitted to a dataset; the bench VALIDATES the
unfitted predicates (TPR 1.000 / FPR 0.000 on the 820-scenario baseline)
instead of tuning a threshold. Anti-poisoning is set semantics: predicates
count distinct `mitigation_origin`s, so same-actor repeats collapse.

The scalar machinery below (§5) is retained as demoted TELEMETRY — the
`resource_score_telemetry` snapshot fields and the graph's `suspicion_score`
property; `FLAG_THRESHOLD` labels that telemetry and gates nothing.

## 4b. Resource identity

Cross-session detection needs **one resource ⇒ one id**. An id that varies with
something the caller controls but that does not change *what* is touched (the tool
verb, the encoding, `./`) is **evasion** — accesses spread over several nodes and
none accumulates. An id that drops something distinguishing resources (the URL host,
the query) is **poisoning** — traffic on one resource raises or launders another.

`graph.derive.derive_identity(tool, args)` is the single function both the hot-path
enricher and the audit-path `CoreSessionSink` call, so they agree by construction.

| Source | Resource id | Container id | Tier |
|---|---|---|---|
| local path (`path` / `file_path` / `file`) | `file:/data/secret.txt` | `file:/data` | path 0.7 |
| relative local path | `file:rel/x` | `file:rel` | path 0.7 |
| non-URL path, recognised provider tool | `sharepoint:/sites/hr/x.xlsx` | `sharepoint:/sites/hr` | path 0.7 |
| http(s) URL (`url` / `uri`) | `url:corp.example.com/a/b?id=1&v=2` | `url:corp.example.com/a` | path 0.7 |
| other-scheme URL | `url:s3://bucket/key` | `url:s3://bucket/` | path 0.7 |
| `file://host/...` (remote) | `file://host/share/x` | `file://host/share` | path 0.7 |
| provider object id, **no path**, recognised provider | `sharepoint:item:42` | itself | provider_id 1.0 |
| `filename` (+ `size`, `last_modified`) | `heuristic:name\|size\|mtime` (`<provider>:heuristic:…` under a recognised provider) | itself | heuristic 0.4 |
| nothing of the above (bash, send_email …) | **no access recorded** | — | — |

Rules:

- **The tool verb never reaches the id.** The only thing a tool name contributes is a
  provider from the `KNOWN_PROVIDERS` allowlist (sharepoint, onedrive, gdrive, gmail,
  outlook, slack, teams, dropbox, box, confluence, jira, notion, github, gitlab,
  salesforce), matched on whole name tokens (split on `_`, `__`, `.`, `-`, `/`, `:`,
  camelCase). `read_file`, `write_file`, `fs_read`, `mcp__fs__read_file` and `Read`
  on `/data/secret.txt` all yield `file:/data/secret.txt`; `wasp_tool` is not
  SharePoint. An unrecognised service never invents a namespace.
- **Path before provider id.** A path/URL the tool opens outranks an `object_id` /
  `item_id` in the args (which an attacker can add alongside a real path). A provider
  id is used only without a path, and only namespaced by a recognised provider — a
  bare `42` would be one node for every tool that numbers its objects.
- **Lexical only — no disk, network or clock.** Percent-decode once; resolve `.`,
  `..` (never above `/`) and repeated `/`; strip trailing `/`. Symlinks are *not*
  resolved (see §10a).
- **Case.** Only the URL scheme and host are lowercased; paths keep their case
  (case-sensitive filesystems and URL paths).
- **URLs** keep the host, a non-default port and the query (sorted, re-encoded; the
  query is routinely the identity, `?id=1` ≠ `?id=2`). Dropped: fragment, userinfo,
  default port, and credential params (`token`, `access_token`, `id_token`,
  `refresh_token`, `api_key`, `apikey`, `sig`, `signature`, `x-amz-*`, `x-goog-*`) —
  they rotate per request and must not be persisted into the graph or snapshot.
  `http` and `https` share an id. Local paths keep `#` and `?` (legal filename chars).
- **Containers derive from the normalised locator**, never the raw arg. A resource
  with no hierarchy (provider item, fingerprint) is its own container rather than one
  provider-wide container that would make every item adjacent to every other.
- **`""` is never an id.** A call that names no resource records no access (enricher
  and sink); `construct.upsert_graph` and `evidence_from_session` also drop `""`
  defensively.

**Migration (ids changed after 0.4.2).** Resource and container ids changed format
(previously e.g. `read:/data/secret.txt`, `fs:/data/secret.txt`, lowercased paths,
host-less URLs, bare provider ids, `""` for path-less calls). Reputation, evidence
and `Resource`/`Container` nodes accumulated under the old ids **do not carry over**
— nothing maps old ids to new ones (the old mapping was many-to-many, which is the
bug). After upgrading, old snapshot entries and graph nodes simply stop being hit
and decay/expire on their normal schedule (evidence TTL 30 days); purge them
explicitly if a clean start is preferred.

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
| Provider object ID (recognised provider, no path — §4b) | 1.0 |
| Normalized path / URL | 0.7 |
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

Both counters (`signal_history`, `prior_counts`) are **windowed** with the
evidence window (`SentinelPolicy.window_days`): each entry records when the
cycle applied it and is pruned once older than the window, at the start of
every cycle. Unwindowed, a year of old signals kept `0.5^prior_count ≈ 0` for
an actor forever although every fact behind the count had expired. State files
written before the counters were windowed load intact, with their entries dated
at load time (they expire one window later).

### 5.4 Caution weight (adjacent resources)

Resources not directly accessed but topologically adjacent to a hot resource receive a caution weight:

```
caution = BASE_CAUTION × topology_factor × canonical_confidence
BASE_CAUTION = 0.3
```

The spec's `time_decay(days_since_last_decay)` term is 1 at application time
and is not computed: caution is written in the same cycle as the hot signal
that causes it, right after `DECAY_QUERY`, so for any scored node the elapsed
time is ≈ 0. The only nodes for which the term differed were score-0
neighbours, whose `last_decay_at` is stale by design (§5.5) — there it shrank
caution by the neighbour's *age*, not by anything about the signal. Ageing of a
caution contribution is left to `DECAY_QUERY` on later cycles, as for hot
weights.

### 5.5 Time decay

Score halves every 30 days. Always computed against `last_decay_at`, never `last_signal_at` (invariant A-3):

```
decay_factor = 0.5^(days_since_last_decay / 30)
```

Applied in Neo4j at the start of every audit cycle (before hot weights) via `DECAY_QUERY`.

`DECAY_QUERY` skips score-0 nodes, so their `last_decay_at` is not advanced.
Every write that lifts a node off 0 (hot, caution, fanout) therefore restarts
its decay clock (`last_decay_at = now` when the score before the write was 0):
`last_decay_at` means "decaying since", and a node that sat at 0 from day 0 to
day 100 has nothing to decay for those days. Without the restart the first
decay after a day-100 hit multiplied the fresh score by `0.5^(100/30) ≈ 0.1`.

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
0. Graph construction   → construct.upsert_graph: MERGE Agent/Session/Resource +
                          ACCESSED/IN_SESSION for every session; derive ADJACENT_TO
                          from container co-membership. The scoring Cypher below only
                          reads/updates nodes, so this producer must run first.

1. DECAY_QUERY           → Neo4j: decay all resources with non-zero score
                                   update last_decay_at; leave last_signal_at unchanged

2. For each tainted session:
   a. Fanout detection  → declared quota (predicates.fanout_exceeded)
                          guard: session.had_taint = True (A-15)
                          guard: max signal_type ≥ READ_SUMMARIZE (A-15)
                          count: DISTINCT containers touched at rank ≥ READ_SUMMARIZE
                                 (predicates.fanout_containers; "" never counts)
                          > policy.fanout_quota_for(source_class) → emit FanoutSignal
                          (z-score vs AgentContainerBaseline = telemetry only)

   b. Hot weights       → once per (session, resource, signal) — the ACCESSED
                          edge's key; "" resource ids skipped:
                          compute raw×diversity×dampening pre-scale
                          HOT_WEIGHT_QUERY → Neo4j (Cypher accumulates and multiplies
                          by canonical_confidence — A-8); returns before/after
                          record ReputationEvent from the returned scores
                          if fanout: FANOUT_WEIGHT_QUERY adds 0.5 separately (A-10)

   c. Caution weights   → CAUTION_ADJACENT_QUERY → Neo4j
                          (adjacent resources not directly accessed)

3. Read back            → RESOURCE_SCORES_QUERY: Neo4j is authoritative — the snapshot
                          scores (incl. decay, fanout AND caution) are read from the
                          graph, not re-accumulated in Python
   Container scores     → recompute from the read-back scores (A-9)

4. Snapshot swap        → ReputationSnapshot.with_checksum()
                          atomic_swap(snapshot_dir, snapshot)   ← A-5, A-16
```

### Fanout detection detail

```python
qualifying = fanout_containers((a.container_id, a.signal_type) for a in accesses)
#   = {cid for cid, rank in ... if cid and rank >= READ_SUMMARIZE}

# Trigger condition (all required — A-15):
# 1. had_taint
# 2. max signal_type ≥ READ_SUMMARIZE
# 3. len(qualifying) > policy.fanout_quota_for(source_class)   # default 7
```

Containers touched only at `READ` do not count: read-only breadth is not
staging breadth (8 READ containers + 1 READ_SUMMARIZE container is 1, not 9).
The flat fanout weight (A-10) still lands on every resource the session touched,
each once. `FanoutSignal.unique_containers` is the qualifying count; its
`z_score` against the smoothed per-agent `AgentContainerBaseline` (α=0.3, last
50 sessions, all containers) is telemetry only — no warm-up guard exists or is
needed, since a declared quota needs no history.

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
    checksum: str                           # SHA-256 of the two reputation maps
                                            # (unkeyed: corruption detection only)
    signature: str                          # HMAC-SHA256 (AXOR_SNAPSHOT_KEY) over the
                                            # whole snapshot minus checksum/signature
    # + resource_level / container_level / verdict_facts / *_score_telemetry
```

### Atomic write (A-5, A-16)

Integrity guarantee: no reader ever sees a partial write.

```
POSIX:
  0. refuse N <= live version (SnapshotVersionRegression)
  1. serialize → bytes
  2. write temp + fsync + os.replace → snapshot_v{N}.json (never in place)
  3. verify checksum of in-memory bytes (never re-reads from disk — A-5)
  4. os.symlink(new_file, snapshot_link_v{N})
  5. os.rename(snapshot_link_v{N}, snapshot_current)  ← atomic on POSIX; fsync dir
  6. prune old versions, stale snapshot_link_v* and temp files

Windows:
  0–3. same
  4. os.replace(new_file, snapshot_current)       ← atomic on Windows
```

Versions never go backwards: on start the cycle resumes at
max(sentinel_state.json version, highest snapshot_v*.json, live link version), so
a lost or truncated state file cannot make it reuse a retained version number.
`sentinel_state.json` is itself written atomically (temp + fsync + os.replace).

### Load + checksum verification

```python
snapshot = load_snapshot(snapshot_dir)
# Returns None (with AuditIntegrityWarning) on checksum / signature mismatch,
# a malformed file, or levels that contradict the suspicion maps
# Never raises — fail-safe for hot path
```

`load_snapshot` has no memory, so it cannot detect a rollback (an older, validly
signed `snapshot_v{N}.json` re-linked as current). `SnapshotIntentEnricher.reload`
is the stateful reader: it keeps its held snapshot when the load fails and refuses
a version lower than the one it holds.

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
- Resource/container ids derived via `graph.derive.derive_identity` — the same function `CoreSessionSink` uses (§4b); a call naming no resource is left unenriched
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

**Detection mechanism:** declared quota — more than `fanout_quota_for(source_class)` distinct containers touched at rank ≥ `READ_SUMMARIZE` in one tainted session (§6). No baseline and no cold-start guard: the quota needs no history, and there is nothing an attacker can walk upward.

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
| A-3 | Decay uses `last_decay_at`; signal events use `last_signal_at`; they are never swapped | `DECAY_QUERY` updates `last_decay_at`; `HOT_WEIGHT_QUERY` updates `last_signal_at` (and restarts `last_decay_at` only when lifting a score-0 node, §5.5) |
| A-4 | Decay runs first in every audit cycle, before any hot weight | `cycle.py`: `apply_decay()` before session loop |
| A-5 | Checksum verified from in-memory bytes before snapshot is made visible | `atomic_swap()`: `_verify_checksum_bytes(serialized.encode(), checksum)` |
| A-6 | No Neo4j call on the hot path | `SnapshotIntentEnricher.enrich()` reads dict only |
| A-7 | `flagged` is never exposed as a feature on the intent | `NormalizedIntent` has no `flagged` field |
| A-8 | `effective_weight = raw × confidence × diversity × dampening` | `compute_effective_weight()` in `weight.py` |
| A-9 | Container score recomputed after every member score change | `cycle.run_once()` step 2f |
| A-10 | Fanout flat 0.5 applied as separate `accumulate()` after hot weight | `SentinelCycle._run_once_locked` step "2d(fanout)": `q.apply_fanout_weight(... fanout.affected_resources ...)` after the per-access hot-weight loop |
| A-11 | Reputation is observe-only — it never denies | core `record_detection` is tightening-only, no deny path |
| A-12 | Reputation acts only via the opt-in degradation floor | `detection_floor` unset → `record_detection` is a no-op |
| A-13 | `origin_dampening` never zero, never > 1 | `0.5^n ∈ (0, 1]` for all n ≥ 0 |
| A-14 | *(retired)* There is no fanout warm-up guard: the trigger is a declared quota, which needs no history (no cold-start gap to guard) | `predicates.fanout_exceeded` takes no baseline |
| A-15 | Fanout requires: `had_taint AND max signal ≥ READ_SUMMARIZE AND count(containers at rank ≥ READ_SUMMARIZE) > quota` | `_check_fanout()` → `predicates.fanout_containers` + `fanout_exceeded` |
| A-16 | Snapshot write is atomic: symlink rename (POSIX) / `os.replace` (Windows) | `atomic_swap()` in `snapshot.py` |
| A-17 | `validate_snapshot_dir()` warns if path is a network mount | `_warn_if_network_mount()` reads `/proc/mounts` |

---

## 10a. Security model & known limitations

Sentinel is a **resource-centric** detector: it scores *patterns of access*, it does
not authenticate *intent legitimacy*. Reputation feeds core as observe-only telemetry
(it tightens degradation at most, never denies), which bounds the blast radius of the
limitations below — a poisoned score cannot cause a wrong deny.

- **Contributing sessions are not authenticated** (by design). Any session that
  reaches the cycle contributes hot weight to whatever `resource_id` / `signal_type`
  it claims, so an attacker who controls sessions can directly raise a resource's
  score. Accumulation is bounded to `[0, 1]` and dampened, but the score is evidence
  of *access concentration*, not attacker-proof intent. Do not treat a high score as
  unforgeable.
- **Anti-poisoning factors key on the actor identity, not the claimed source label.**
  `origin_dampening` and `source_diversity_factor` key on `SessionSummary.mitigation_origin`
  — the authenticated `source_class` core attests when available, else the `agent_id`
  — never the attacker-controllable `taint_source` label. So rotating the source label
  no longer resets dampening (the F1 fix); an attacker would have to rotate the actor
  identity, which requires distinct authenticated principals. Residual: if `agent_id`
  itself is not authenticated upstream (no `source_class` attested), an attacker who
  can spin up distinct agent identities can still spread the count — bounded by
  observe-only core, and fully closed once core attests `source_class`.
- **Window wait-out (was: time-decay wait-out).** Verdicts are computed over a
  declared sliding window (`SentinelPolicy.window_days`, default 30): evidence
  either counts or has expired — the margin an attacker can wait out is now an
  EXPLICIT, auditable policy constant instead of an exponential-decay side
  effect. Pacing staging beyond the window still evades P3-style counting; the
  P4 sequence predicate (staged-then-export on the same resource) fires
  whenever both facts land inside one window regardless of pacing density.
- **Fanout baseline: CLOSED.** The trigger is a declared per-actor-class quota
  (`SentinelPolicy.fanout_quota_for(source_class)`), not the agent's smoothed
  baseline — there is nothing to walk upward and no cold-start gap. The z-score
  is computed as telemetry on emitted signals only.
- **Snapshot/state authentication is opt-in.** HMAC engages only with
  `AXOR_SNAPSHOT_KEY` (fail-closed only under `AXOR_ENV=production` /
  `AXOR_SNAPSHOT_REQUIRE_SIGNATURE`). The default checksum-only mode is forgeable by
  any writer to the snapshot dir; `load_snapshot` now warns loudly when unauthenticated.
  **Set the key in any real deployment** — the `sentinel_state.json` it also signs
  gates the dampening/baseline counters across restarts.

What is sound by construction: no raw session content crosses the boundary (the
snapshot is `id → float` only); all Cypher is parameterized; the hot-path enricher is
Neo4j-free and fail-safe; the snapshot swap is atomic and TOCTOU-safe; and the
resource-id normalizer is purely lexical and tool-independent (§4b), so the hot path
and the audit path compute identical ids.

- **Symlink aliases split identity (residual).** The normalizer no longer
  `realpath`-resolves: that touched the disk on the hot path, answered differently
  on the enricher's and the sink's hosts (breaking id parity) and was racy (a link
  can be repointed between resolution and use). A path reached through a symlink
  therefore gets its own id. Likewise a relative path is not joined to a cwd, so
  `file:x` and `file:/work/x` are distinct.

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
