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
- **#1 dual-write — DONE (Neo4j-authoritative read-back, incl. caution topology).**
  `run_once` now upserts the graph, writes decay/hot/fanout/caution, and reads scores
  back from Neo4j to build the snapshot — the in-memory accumulation (and the drift it
  risked) is gone. The broken Cypher is fixed (two-sided `[0,1]` clamp) and the whole
  path is validated end-to-end against a live Neo4j, run in CI via a `neo4j` service.
  Item 1 (b) — the `ADJACENT_TO` topology the caution query walks — is now produced
  from container membership (`_build_adjacency_pairs`), so caution is live, not inert.
- **#3 F1 anti-poisoning keying — DONE.** Dampening/diversity key on
  `mitigation_origin` (authenticated `source_class` if attested, else `agent_id`,
  else the constant `"unattributed"` — never `taint_source`). Item 3 records the
  residual (agent-identity authentication).
- **Review round-2 follow-ups — DONE.** Two-sided score clamp; constant origin
  fallback (no `taint_source`); empty-`resource_id` accesses dropped (path-less
  egress no longer collides onto one node); CI `neo4j` service so the live
  parse-guard + read-back tests run; `probe_bridge` test coverage (was 0%).

---

## 1. Dual-write scoring (in-memory dict ↔ Neo4j) — DONE (Neo4j-authoritative read-back)

**Resolved.** `run_once` is now Neo4j-authoritative: it upserts the graph, applies
decay + hot/fanout/caution writes, and **reads the scores back** from Neo4j to build
the snapshot — the in-memory `scores` accumulation is gone, so there is no longer a
second scorer to drift. Decay and fanout are reflected in the same-cycle snapshot.
Validated end-to-end against live Neo4j (`tests/test_neo4j_integration.py::TestRunOnceLive`):
scores match the `accumulate` reference, accumulate across cycles, and flow into the
enricher. The mock-based cycle tests now assert orchestration + Python state only
(the mock can't execute Cypher); score VALUES are asserted live.

**Caution topology (item (b)) — DONE.** `apply_caution_adjacent` walks `ADJACENT_TO`
edges, which `_build_adjacency_pairs` now produces each cycle from container
membership (co-members are "same directory / workspace", topology_factor 1.0). Edges
are MERGEd both directions on already-observed Resource nodes (MATCH-only), so caution
propagates to a known-but-not-accessed-this-cycle neighbor — the intended
cross-session signal. Validated live
(`TestRunOnceLive::test_caution_boosts_unaccessed_container_neighbor`). Finer
topology_factor tiers (same service / namespace / project) need a service map sentinel
does not yet receive; container membership is the source available today and matches
the bench topology pool.

Historical context (the bugs this surfaced):

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

The feature, as built: (a) a graph-construction layer — upsert
`Session`/`Resource`/`Agent` nodes + `ACCESSED`/`IN_SESSION` edges per cycle
(**done**: `upsert_session`); (b) a source for the `ADJACENT_TO` topology the caution
query walks (**done**: `_build_adjacency_pairs` from container membership, so caution
is live); (c) build the snapshot by reading scores back from Neo4j after all writes,
deleting the Python re-accumulation (**done**). Validated with the live-Neo4j harness
(`AXOR_TEST_NEO4J_BOLT`), run in CI via a `neo4j` service.

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
- **Model dataclasses are partly aspirational** (`DestinationNode` never instantiated;
  `EXPORTED_TO`/`MEMBER_OF` edges not yet written) — schema-as-doc; harmless.
  (`ADJACENT_TO` is now written each cycle from container membership — caution is no
  longer inert.)
