# Deferred follow-ups — sentinel review

Findings from the review that were intentionally **not** changed in the cleanup
pass, because each is a larger refactor or a design change that carries more risk
than the finding (sentinel feeds core observe-only, so none of these is a wrong-deny
bug). Ordered by value/effort. The stale Phase-1 model, polarity inversion, dead
code, doc drift, and snapshot hardening (F7/F8) were already fixed.

---

## 1. Dual-write scoring (in-memory dict ↔ Neo4j) — MAJOR (engineering risk)

**Where:** `axor_sentinel/sentinel/cycle.py` `run_once`. Scores are maintained in two
stores reconciled by convention: an in-memory `scores` dict (the snapshot source of
truth) and Neo4j (the next-cycle source of truth). Hot-weight accumulation is done
twice — once in Python (`weight.accumulate`) and once in Cypher (`HOT_WEIGHT_QUERY`)
— with a hand-tuned pre-scaling so the two agree. **Caution weights and time-decay
are applied only in Neo4j**, never to the in-memory `scores`, so the snapshot written
*this* cycle omits caution/decay until the *next* cycle reads Neo4j back.

**Why deferred:** this is the riskiest part of the system to change; the eventual-
consistency (caution/decay appear next cycle) is a real latency property, not a
crash. A correct fix is a single-source-of-truth refactor: make Neo4j authoritative
and build the snapshot by reading scores *back* from Neo4j after all writes (decay,
hot, caution, fanout), removing the Python re-accumulation entirely. That deletes the
dual-write and the pre-scaling trick. Guard with the full suite; verify snapshot
values match a Neo4j read-back on a fixture graph.

**Interim:** the eventual-consistency property is now documented; do not rely on a
single cycle reflecting caution/decay.

## 2. Cycle concurrency & crash-consistency — MAJOR

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

## 3. Anti-poisoning keyed on attacker-controlled `taint_source` (F1) — MAJOR (graph integrity)

**Where:** `weight.source_diversity_factor` / `weight.origin_dampening`, keyed on
`(resource_id, taint_source)` in `cycle.py`. `taint_source` comes from the session
record and is attacker-influenceable, so rotating the label resets both factors.

**Why deferred:** there is no truly authenticated identity in sentinel's inputs, so
this is partly inherent. Keying on fewer dimensions (per-resource, source-agnostic)
makes dampening rotation-proof but penalizes legitimately multi-source access; keying
on more dimensions is easier to evade. A real fix needs core to hand sentinel an
*authenticated* source class (a trusted label assigned kernel-side), which is a
cross-repo contract change. Documented as a known limitation
(`docs/architecture.md` §10a) until then.

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
- **Model dataclasses are partly aspirational** (`DestinationNode`, `AdjacentToEdge`
  never instantiated; `ADJACENT_TO` edges never written by sentinel, so the caution
  query is inert without an external topology populator) — schema-as-doc; harmless.
