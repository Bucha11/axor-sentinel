# Deferred follow-ups — sentinel review

Findings from the review that were intentionally **not** changed in the cleanup
pass, because each is a larger refactor or a design change that carries more risk
than the finding (sentinel feeds core observe-only, so none of these is a wrong-deny
bug). Ordered by value/effort. The stale Phase-1 model, polarity inversion, dead
code, doc drift, and snapshot hardening (F7/F8) were already fixed.

---

## Status (this pass)

- **#2 concurrency/crash-consistency — DONE.** `run_once` is serialised by an
  instance lock; `save_state` now runs before `atomic_swap` (state ahead of snapshot
  on crash); version is in the signed state blob.
- **#1 dual-write — DONE.** Neo4j is now the single source of truth. The
  graph-construction layer (`graph/construct.py`, Step 0 of `_run_once_locked`)
  upserts Agent/Session/Resource + ACCESSED/IN_SESSION each cycle and derives
  ADJACENT_TO from container co-membership; decay/hot/fanout/caution all run in
  Cypher; and the snapshot is **read back** from the graph
  (`RESOURCE_SCORES_QUERY`), deleting the parallel Python `accumulate`. Caution and
  decay — previously graph-only and invisible to the served snapshot — now land in
  it the same cycle. The per-resource hot-weight query was also tightened to match a
  specific resource id (the old signal_type-wide match double-counted siblings, a
  latent bug that read-back would have made live). Validated end-to-end against a live
  Neo4j; the mock-based unit tests that asserted snapshot magnitudes moved to
  `tests/test_neo4j_integration.py::TestFullCycleSnapshot`.
- **#3 F1 anti-poisoning keying — DONE.** Dampening/diversity now key on
  `mitigation_origin` (authenticated `source_class` if attested, else `agent_id`),
  not the attacker-controllable `taint_source`. Item 3 below records the residual.

---

## 1. Dual-write scoring (in-memory dict ↔ Neo4j) — DONE

Neo4j is now authoritative for resource scores and the snapshot is read back from it.
History of why this was a feature, not a refactor, is kept below for context.

**Original problem:** `run_once` maintained scores in two stores reconciled by
convention — an in-memory `scores` dict (the snapshot source of truth) and Neo4j
(the next-cycle store). Hot-weight accumulation ran twice (Python `weight.accumulate`
+ Cypher) with hand-tuned pre-scaling to keep them agreeing, while **caution and decay
ran only in Neo4j**, so the snapshot written *this* cycle omitted them.

**Discovered when a live Neo4j was finally run against the queries (5.26):** the write
path was not merely lagging — it was **non-functional**:
1. Three write queries used `min(1.0, …)` — an *aggregating* function — in a `SET`,
   a **syntax error on Neo4j 5.x**. *(Fixed: scalar `CASE WHEN new > 1.0 …`.)*
2. Sentinel **never created any graph nodes/edges**, so even the parseable queries
   matched nothing. *(Fixed: `graph/construct.py`.)*

**What was built (in order):**
- **(a) graph construction** — `graph/construct.py` upserts `Agent`/`Session`/
  `Resource` + `ACCESSED`/`IN_SESSION` every cycle (Step 0 of `_run_once_locked`).
- **(b) ADJACENT_TO source** — derived from container co-membership (topology_factor
  defaults to the same-container 1.0; finer per-spec factors — same service 0.7, same
  MCP namespace 0.6, … — need container-type metadata the cycle isn't handed yet).
- **(c) read-back** — after decay/hot/fanout/caution, the snapshot scores come from
  `RESOURCE_SCORES_QUERY`; the Python `accumulate` path is deleted. Caution and decay
  now appear in the served snapshot the same cycle. The hot-weight query was tightened
  to match a specific `resource_id` (the signal_type-wide match double-counted
  siblings — harmless while the snapshot was Python-built, a live bug once read back),
  and it returns before/after so `ReputationEvent` evidence is sourced from Neo4j too.

**Residual (surfaced by the step-3 review, all minor / non-wrong-deny):**
- **Unbounded read-back.** `RESOURCE_SCORES_QUERY` returns every resource with
  `score > 0`. Decay halves every 30 days and never reaches 0 in float, so resources
  linger for months and the snapshot grows across cycles. Needs a periodic prune
  (delete/forget resources below an epsilon) to bound snapshot size. The snapshot is a
  full reputation map by design, so scoping the read-back to *this cycle* would be
  wrong — pruning is the right lever.
- **Coarse topology.** Same-container `ADJACENT_TO` is pinned at `topology_factor`
  1.0; the per-spec 5-tier grade (same service 0.7, MCP namespace 0.6, …) and any
  *cross*-container adjacency need container-type metadata the cycle isn't handed.
  Adjacency is also re-derived and re-MERGEd every cycle though it is static topology
  (`AdjacentToEdge` doc: "computed once on resource discovery") — make it incremental.
- **Per-access round-trips.** Hot weight is one `apply_hot_weight` Bolt round-trip per
  accessed resource. `FANOUT_WEIGHT_QUERY` already shows the batched `UNWIND` form;
  a session touching N resources could collapse N round-trips into one.
- **Event vs snapshot for fanout.** `ReputationEvent.score_after` is the post-hot,
  *pre*-fanout score (sourced from the hot-weight `RETURN`); for a fanout-affected
  resource the served snapshot score is higher. Intentional (the FanoutSignal carries
  the fanout evidence), but a consumer correlating the two sees different numbers.
- **Seed-only `resource_scores`.** The `run_once` arg now only seeds brand-new
  resources (existing nodes persist in Neo4j); the forward-integration docstrings in
  `integration/` still show the old "thread scores back in" call shape. Harmless
  (those sinks have no producer yet) but should be refreshed when they go live.

## 2. Cycle concurrency & crash-consistency — DONE (see Status)

**Where:** `cycle.py` `run_once` mutates `_signal_history` / `_prior_counts` /
`_baselines` / `_current_version` with no lock, and there is no guard that version
monotonicity survives a crash between `atomic_swap` (snapshot at version N) and
`save_state` (state at N-1). Overlapping cycles, or a cycle racing `save_state`,
corrupt the counters; a crash in the window re-emits version N with different
contents next run.

**Plan:** (a) take a `threading.Lock` around `run_once` (cheap insurance even though
the cycle is meant to be a single background job); (b) persist `save_state` *before*
the snapshot swap so a crash leaves state ahead of the snapshot (next run re-derives a
fresh snapshot) rather than behind; (c) include the snapshot version in the signed
state so a mismatch is detectable on load. Add a crash-ordering test.

## 3. Anti-poisoning keyed on attacker-controlled `taint_source` (F1) — DONE, with a residual

**Fixed:** dampening/diversity now key on `SessionSummary.mitigation_origin` —
the authenticated `source_class` (forward contract, threaded through `CoreSessionRecord`)
when core attests one, else `agent_id` — never the rotatable `taint_source` label.
Rotating the label no longer resets the mitigation (regression-tested in
`test_cycle.py::TestPoisoningMitigationKeying`).

**Residual (needs core):** when no `source_class` is attested, the fallback `agent_id`
is only as trustworthy as the upstream agent-identity authentication. An attacker who
can mint distinct agent identities can still spread the count across them. Fully
closing this needs core to attest `source_class` on the (forward) session record —
the `CoreSessionRecord.source_class` field is already in place to receive it. Bounded
meanwhile by observe-only core.

## 4. `update_resource_score` vs. inline cycle accumulation — MINOR (duplication)

`weight.update_resource_score` is the tested canonical scoring helper, but `run_once`
inlines the equivalent `accumulate` + flag logic rather than calling it. Fold the
cycle onto `update_resource_score` (or vice-versa) once item 1 collapses the
dual-write, so there is one scoring code path.

## 5. `FanoutSignal.window_minutes` vestigial — NIT

Always written `0.0` (`cycle.py`), and the `events.py` docstring still says "within a
window". Either compute a real window or drop the field (a public event field, so
removing it is a minor breaking change — bundle with the next event-schema revision).

## Not planned (accepted, with rationale)
- **No authentication of contributing sessions (F2)** — by design (resource-centric);
  documented in §10a. Bounded by observe-only core.
- **Time-decay wait-out (F3) / self-trained fanout baseline (F5)** — inherent to
  threshold detection; documented in §10a. A decay floor for once-flagged resources
  and frozen-baseline-when-tainted would help but change detection behavior (false-
  positive risk), so left as tuning, not a code change.
- **Model dataclasses are partly aspirational** (`DestinationNode` never
  instantiated) — schema-as-doc; harmless. (`ADJACENT_TO` is no longer inert:
  `graph/construct.py` now writes it from container co-membership — see item 1.)
