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
- **#1 dual-write — PARTIALLY DONE.** The hand-tuned pre-scale drift is gone:
  `compute_weight_factors` is the single source for both the in-memory `effective`
  weight and the Cypher `without_confidence`. Read-back FEATURE phase 1 has since
  landed: the broken Cypher is fixed (two-sided `[0,1]` clamp), `upsert_session` +
  `read_resource_scores` exist and are validated end-to-end against a live Neo4j (now
  run in CI via a `neo4j` service). Still **not** done: rewiring `run_once` to the
  read-back path (and an `ADJACENT_TO` topology producer for caution). Item 1 below.
- **#3 F1 anti-poisoning keying — DONE.** Dampening/diversity key on
  `mitigation_origin` (authenticated `source_class` if attested, else `agent_id`,
  else the constant `"unattributed"` — never `taint_source`). Item 3 records the
  residual (agent-identity authentication).
- **Review round-2 follow-ups — DONE.** Two-sided score clamp; constant origin
  fallback (no `taint_source`); empty-`resource_id` accesses dropped (path-less
  egress no longer collides onto one node); CI `neo4j` service so the live
  parse-guard + read-back tests run; `probe_bridge` test coverage (was 0%).

---

## 1. Dual-write scoring (in-memory dict ↔ Neo4j) — MAJOR (engineering risk) — REMAINING: read-back

The pre-scale drift is fixed (see Status). What remains needs a live Neo4j:

**Where:** `axor_sentinel/sentinel/cycle.py` `run_once`. Scores are maintained in two
stores reconciled by convention: an in-memory `scores` dict (the snapshot source of
truth) and Neo4j (the next-cycle source of truth). Hot-weight accumulation is done
twice — once in Python (`weight.accumulate`) and once in Cypher (`HOT_WEIGHT_QUERY`)
— with a hand-tuned pre-scaling so the two agree. **Caution weights and time-decay
are applied only in Neo4j**, never to the in-memory `scores`, so the snapshot written
*this* cycle omits caution/decay until the *next* cycle reads Neo4j back.

**Discovered when a live Neo4j was finally run against the queries (5.26):** the
Neo4j write path was not merely lagging — it was **non-functional**:
1. Three of the five write queries (`HOT_WEIGHT`, `CAUTION_ADJACENT`,
   `FANOUT_WEIGHT`) used `min(1.0, …)` — an *aggregating* function — in a `SET`,
   which is a **syntax error on Neo4j 5.x**; they would crash on any real server.
   (Now fixed: scalar `CASE WHEN new > 1.0 THEN 1.0 ELSE new END`, validated end-to-
   end against Neo4j 5.26 — see `tests/test_neo4j_integration.py`.)
2. Sentinel **created no graph nodes/edges at the time** (no `MERGE`/`CREATE`
   anywhere), so even the parseable queries matched nothing on the graph nobody
   populated. The in-memory Python path was the only working, tested scorer.
   (Construction layer landed since: `upsert_session` in `graph/queries.py` now
   MERGEs the nodes/edges — but `run_once` is not yet rewired to it; see Status.)

**So full "Neo4j-authoritative read-back" is a FEATURE, not a refactor.** It needs:
(a) a graph-construction layer — upsert `Session`/`Resource`/`Agent` nodes +
`ACCESSED`/`IN_SESSION` edges per cycle (**done**: `upsert_session`); (b) a source
for the `ADJACENT_TO` topology the caution query walks (no producer in sentinel, so caution
is inert regardless); (c) then build the snapshot by reading scores back from Neo4j
after all writes, deleting the Python re-accumulation. Validate with the live-Neo4j
test harness (`AXOR_TEST_NEO4J_BOLT`).

**Done in this pass:** the hand-tuned pre-scale drift is gone (`compute_weight_factors`),
and the broken Cypher is fixed + validated against a real server. The snapshot/Neo4j
eventual-consistency for caution/decay remains (Python is authoritative).

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

`FanoutSignal.window_minutes` (`events.py`) is always written `0.0` by the cycle —
a vestigial field. Either compute a real window or drop the field (a public event
field, so removing it is a minor breaking change — bundle with the next event-schema
revision).

## Not planned (accepted, with rationale)
- **No authentication of contributing sessions (F2)** — by design (resource-centric);
  documented in §10a. Bounded by observe-only core.
- **Time-decay wait-out (F3) / self-trained fanout baseline (F5)** — inherent to
  threshold detection; documented in §10a. A decay floor for once-flagged resources
  and frozen-baseline-when-tainted would help but change detection behavior (false-
  positive risk), so left as tuning, not a code change.
- **Model dataclasses are partly aspirational** (`DestinationNode`, `AdjacentToEdge`
  never instantiated; `ADJACENT_TO` edges never written by sentinel, so the caution
  query is inert without an external topology populator) — schema-as-doc; harmless.
