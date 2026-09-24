from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axor_sentinel.graph import construct
from axor_sentinel.graph import queries as q
from axor_sentinel.graph.model import SignalType
from axor_sentinel.sentinel.attestation import (
    AttestationRecord,
    active_attestation,
    effective_score,
    is_superseded,
    validate,
)
from axor_sentinel.sentinel.events import (
    AgentContainerBaseline,
    FanoutSignal,
    ReputationEvent,
)
from axor_sentinel.sentinel.evidence import EvidenceStore, evidence_from_session
from axor_sentinel.sentinel.predicates import (
    LEVEL_SUSPICION,
    ReputationLevel,
    SentinelPolicy,
    Verdict,
    evaluate_container,
    evaluate_resource,
    fanout_containers,
    fanout_exceeded,
)
from axor_sentinel.sentinel.snapshot import (
    ReputationSnapshot,
    atomic_swap,
    latest_snapshot_version,
    sign_blob,
    snapshot_payload,
    validate_snapshot_dir,
    verify_blob,
    write_file_atomic,
)
from axor_sentinel.sentinel.weight import (
    FLAG_THRESHOLD,
    compute_container_score,
    compute_hot_weight,
    compute_weight_factors,
)

log = logging.getLogger("axor.sentinel.cycle")

# Fanout parameters. The TRIGGER is the declared quota
# (SentinelPolicy.fanout_containers, evaluated by predicates.fanout_exceeded);
# the smoothed per-agent baseline below feeds the z-score TELEMETRY on emitted
# signals only.
BASELINE_WINDOW_SESSIONS: int = 50  # sessions used to recompute baseline (telemetry)
FANOUT_WEIGHT: float = 0.5          # flat weight added to all touched resources (A-10)


@dataclass
class ResourceAccess:
    """
    A single resource access within a session, as fed to SentinelCycle.

    Strongly typed runtime record: ``canonical_confidence`` from the normalizer,
    ``signal_type`` as the ``SignalType`` enum. Compare with
    ``bench.dataset.schema.AccessEvent`` — the bench's thin serialisation DTO
    (plain string signal_type, no canonical_confidence).
    """
    resource_id: str
    container_id: str
    canonical_confidence: float
    signal_type: SignalType   # graded involvement depth


@dataclass
class SessionSummary:
    """Lightweight session record pulled from axor-core DecisionTrace."""
    session_id: str
    agent_id: str
    started_at: float
    had_taint: bool
    had_export_attempt: bool
    had_failed_export: bool
    had_escalation: bool
    accessed_resources: list[ResourceAccess] = field(default_factory=list)
    taint_source: str = "unknown_external"   # TaintSource.value — descriptive evidence
    # Authenticated source class, when core can attest one (forward contract). Empty
    # = not attested. NEVER set from the attacker-influenceable taint_source label.
    source_class: str = ""

    @property
    def mitigation_origin(self) -> str:
        """The origin key the poisoning-mitigation factors (dampening / diversity)
        key on. Uses the authenticated source_class when present, else the agent_id —
        both are the *actor* identity, far harder to rotate than the taint_source
        label (which an attacker controls and could rotate to reset dampening). See
        the F1 limitation in docs/architecture.md §10a."""
        return self.source_class or self.agent_id or self.taint_source


class SentinelCycle:
    """
    Background audit cycle — pulls session traces, updates graph, writes snapshot.

    Not in the hot path. Runs every AUDIT_INTERVAL (default: 1 hour).

    Audit cycle order (per spec):
      1. Apply time decay to all resources (against last_decay_at)
      2. Per tainted session:
         a. Determine signal_type per accessed resource
         b. Check for fanout → emit FanoutSignal if triggered
         c. Apply hot weights + ReputationEvent per resource
         d. Apply caution weights to adjacent resources
         e. flagged updated on every score change (A-2)
         f. Recompute container scores for affected containers
      3. Atomically swap reputation snapshot
      4. Notify axor-core of new snapshot version
    """

    def __init__(
        self,
        neo4j_session: Any,
        snapshot_dir: Path,
        agent_baselines: dict[str, AgentContainerBaseline] | None = None,
        signal_history: dict[str, list[str]] | None = None,
        prior_counts: dict[tuple[str, str], int] | None = None,
        policy: SentinelPolicy | None = None,
        publish: Callable[[dict], None] | None = None,
    ) -> None:
        """
        Args:
            neo4j_session:    live neo4j.Session for graph operations
            snapshot_dir:     directory for atomic snapshot writes
            publish:          optional; called with ``snapshot_payload(snapshot)``
                              after each snapshot is made visible on disk — the
                              hook a node uses to report its reputation to a
                              control plane (e.g. axor-wrap's
                              ``PlaneConnector.reputation_publisher()``)
            agent_baselines:  agent_id → AgentContainerBaseline (updated in-place)
            signal_history:   resource_id → list[origin] (for diversity factor)
            prior_counts:     (resource_id, origin) → count (for dampening)

        Both counters are WINDOWED like the evidence (policy.window_days): each
        entry carries the time the cycle applied it, and entries older than the
        window are pruned every cycle. Seeded or legacy entries without a
        timestamp are treated as observed at construction/load time, so they
        expire one window later (see _reconcile_counter_times).

        The Neo4j uniqueness constraints are ensured once here
        (construct.ensure_schema; idempotent, failures logged, never raised).
        """
        self._neo4j = neo4j_session
        self._snapshot_dir = Path(snapshot_dir)
        self._reputation_events: list[ReputationEvent] = []
        self._fanout_signals: list[FanoutSignal] = []
        # Operator attestations, keyed by the resource whose branch they cover
        # (UI spec 8.1.1). Append-only: an attestation lowers the score the
        # snapshot exports via a downward recompute, it never mutates Neo4j —
        # the graph stays the untouched evidence, laundering-proof. Persisted
        # in the signed state file (save_state) and restored below: held only
        # in memory, a restart silently dropped every attestation and the
        # exported level jumped back up.
        self._attestations: dict[str, list[AttestationRecord]] = {}
        # Serialise cycles: run_once mutates _signal_history / _prior_counts /
        # _baselines / _current_version, none of which is safe under overlap.
        # Every other reader/writer of that state (save_state, update_baseline)
        # takes it too — save_state iterates the dicts while serialising, and a
        # concurrent insert raised "dictionary changed size during iteration",
        # which the old blanket except turned into a silently missing save.
        # A plain Lock, not an RLock: run_once already holds it when it saves,
        # so it calls _save_state_locked directly instead of re-entering.
        self._lock = threading.Lock()

        # If explicit state is provided (tests / controlled init), use it directly.
        # Otherwise try to restore persisted state from disk so poisoning-mitigation
        # counters and agent baselines survive process restarts.
        # Declared predicate constants for the deterministic verdict layer.
        self._policy = policy or SentinelPolicy()
        self._publish = publish
        if agent_baselines is not None or signal_history is not None or prior_counts is not None:
            self._baselines: dict[str, AgentContainerBaseline] = agent_baselines or {}
            self._signal_history: dict[str, list[str]] = signal_history or {}
            self._prior_counts: dict[tuple[str, str], int] = prior_counts or {}
            self._current_version: int = 0
            self._evidence = EvidenceStore()
            self._signal_history_at: dict[str, list[float]] = {}
            self._prior_counts_at: dict[tuple[str, str], list[float]] = {}
        else:
            # One authenticated read feeds both parsers, so the counters and
            # the attestations always come from the same (verified) file.
            raw = SentinelCycle._read_state_payload(
                self._snapshot_dir / "sentinel_state.json"
            )
            sh, pc, bl, ver, ev = SentinelCycle._parse_state(raw)
            self._attestations = SentinelCycle._parse_attestations(raw)
            self._baselines = bl
            self._signal_history = sh
            self._prior_counts = pc
            self._current_version = ver
            self._evidence = ev
            self._signal_history_at, self._prior_counts_at = (
                SentinelCycle._parse_counter_times(raw)
            )
            if ver > 0:
                log.info("sentinel: restored persisted state version=%d", ver)
        # Entries without a timestamp (explicit seeds, or state written before
        # the counters were windowed) are dated "now": the counts they carry
        # stay in force for one more window and then expire, instead of either
        # vanishing on upgrade or — the old behaviour — never expiring at all.
        self._reconcile_counter_times(time.time())

        # Uniqueness constraints make MERGE an index lookup (not a label scan)
        # and stop two concurrent writers from creating duplicate nodes for one
        # id. Idempotent; a failure (older Neo4j, a mock) is logged, not raised.
        construct.ensure_schema(self._neo4j)

        # Versions never go backwards, even when the state file does. The state
        # can be missing, truncated by a crash (pre-atomic writes), rejected by
        # its signature check, or simply older than the snapshots (a failed
        # save) — every one of those used to restart the sequence at 0, and the
        # next cycle then rewrote the retained snapshot_v1..vN with DIFFERENT
        # content under the same numbers. The snapshot directory itself is the
        # other witness to how far the sequence got, so resume above both. This
        # applies to explicitly-seeded state too: a fresh counter pointed at a
        # used directory would otherwise be refused by atomic_swap's regression
        # guard on its first write.
        on_disk = latest_snapshot_version(self._snapshot_dir)
        if on_disk > self._current_version:
            log.warning(
                "sentinel: state version %d is behind the snapshot directory "
                "(version %d on disk) — resuming above it so no version is reused",
                self._current_version, on_disk,
            )
            self._current_version = on_disk

        # Warn early if snapshot_dir is a network mount (invariant A-17).
        # Done in __init__ so operators learn about the misconfiguration at startup,
        # not at the first write hours later.
        validate_snapshot_dir(self._snapshot_dir)

    def _publish_snapshot(self, snapshot: ReputationSnapshot) -> None:
        """Hand the snapshot to the node's reporter, if one is wired.

        After the swap, never before: the local enricher is the consumer that
        enforces, and what a control plane renders must be what it already
        reads. A reporter that fails is logged and swallowed — reporting is
        observation, and a plane being down must not fail the audit cycle that
        feeds the node's own governance."""
        if self._publish is None:
            return
        try:
            self._publish(snapshot_payload(snapshot))
        except Exception:  # noqa: BLE001 — see docstring
            log.warning(
                "sentinel: publishing snapshot version=%d failed",
                snapshot.version, exc_info=True,
            )

    # ── Operator attestations (UI spec 8.1.1) ──────────────────────────────────

    def attest(self, record: AttestationRecord) -> None:
        """Append an operator attestation over a resource branch. Reason,
        operator and resource_id are required (decision 8); nothing is deleted
        — the score the snapshot exports descends via effective_score, the
        graph is untouched. Revocation is a new record whose ``revokes`` names
        the prior one.

        ``created_at`` is stamped from this cycle's clock — it is what newer
        evidence is compared against (attestation.is_superseded). A caller
        value is kept only when it is set and not in the future: back-dating
        can only make the attestation lapse sooner, but a future timestamp
        would make every later fact look "older" and pin the discount on for
        good — the "trust this forever" the design refuses.

        The operator/org identity is taken as given; authenticating it is the
        caller's job (see attestation module docstring). The record is
        persisted with the rest of the state on the next save (run_once saves
        every cycle; call save_state() to persist immediately).
        """
        validate(record)
        now = time.time()
        stamp = record.created_at
        if not (math.isfinite(stamp) and 0.0 < stamp <= now):
            record = dataclasses.replace(record, created_at=now)
        with self._lock:
            self._attestations.setdefault(record.resource_id, []).insert(0, record)

    def attestations_for(self, resource_id: str) -> list[AttestationRecord]:
        return list(self._attestations.get(resource_id, []))

    def run_once(
        self,
        sessions: list[SessionSummary],
        resource_scores: dict[str, float] | None = None,
        container_members: dict[str, list[str]] | None = None,
    ) -> ReputationSnapshot:
        """
        Execute one audit cycle.

        Args:
            sessions:          completed sessions since last audit
            resource_scores:   optional seeds for brand-new resources only; Neo4j is
                               authoritative for resources it already knows, and the
                               snapshot is read back from it (not from this dict)
            container_members: container_id → [resource_ids] for aggregation

        Returns:
            The newly written ReputationSnapshot.

        Serialised by an instance lock — overlapping cycles would corrupt the
        in-memory counters and the version sequence.
        """
        with self._lock:
            return self._run_once_locked(sessions, resource_scores, container_members)

    @staticmethod
    def _dedupe_sessions(sessions: list[SessionSummary]) -> list[SessionSummary]:
        """Collapse records that share a session_id into a single merged record.

        The caller assembles one cycle's sessions from several sources — e.g. the
        core-derived list plus axor-probe's behavioral-drift buffer
        (ProbeTaintBridge.drain_pending()). The same session_id can therefore
        appear more than once. Processing it twice would double-count fanout and
        caution writes and double-increment the dampening / diversity counters,
        so the records are merged before anything reads them.

        Merge is order-preserving (first occurrence keeps the slot) and is a
        union of evidence: boolean flags OR together, accessed_resources union
        (deduped by resource_id + container_id + signal_type — the container is
        kept in the key because container membership feeds fanout and
        adjacency; the hot weight is deduped separately, per (session,
        resource, signal), in the cycle loop, since the graph's ACCESSED edge
        is keyed by signal only), started_at is the earliest, and source_class is
        taken from the first record that attests one. taint_source keeps the
        first value — it is descriptive evidence only; the poisoning-mitigation
        factors key on the actor origin, not this label (F1).

        Input records are not mutated.
        """
        merged: dict[str, SessionSummary] = {}
        order: list[str] = []
        for s in sessions:
            existing = merged.get(s.session_id)
            if existing is None:
                merged[s.session_id] = dataclasses.replace(
                    s, accessed_resources=list(s.accessed_resources)
                )
                order.append(s.session_id)
                continue
            seen = {
                (a.resource_id, a.container_id, a.signal_type)
                for a in existing.accessed_resources
            }
            existing.accessed_resources.extend(
                a for a in s.accessed_resources
                if (a.resource_id, a.container_id, a.signal_type) not in seen
            )
            existing.had_taint = existing.had_taint or s.had_taint
            existing.had_export_attempt = existing.had_export_attempt or s.had_export_attempt
            existing.had_failed_export = existing.had_failed_export or s.had_failed_export
            existing.had_escalation = existing.had_escalation or s.had_escalation
            existing.started_at = min(existing.started_at, s.started_at)
            if not existing.source_class and s.source_class:
                existing.source_class = s.source_class
            if not existing.agent_id and s.agent_id:
                existing.agent_id = s.agent_id
        return [merged[sid] for sid in order]

    def _run_once_locked(
        self,
        sessions: list[SessionSummary],
        resource_scores: dict[str, float] | None,
        container_members: dict[str, list[str]] | None,
    ) -> ReputationSnapshot:
        now = time.time()
        # Collapse records that share a session_id (e.g. the core-derived list
        # merged with axor-probe's drift buffer) so one session is never
        # double-counted in fanout, caution, or the dampening counters.
        sessions = self._dedupe_sessions(sessions)
        # seed_scores only seeds BRAND-NEW Resource nodes (construct's ON CREATE);
        # existing nodes keep their persisted Neo4j score. Neo4j is authoritative —
        # the snapshot is read back from it at the end, not re-accumulated here.
        seed_scores = dict(resource_scores) if resource_scores else {}
        cmembers = dict(container_members) if container_members else {}

        # Step 0 — materialise the graph for this cycle. The scoring Cypher below
        # only reads/updates nodes; without this producer it matched an empty
        # graph and did nothing. Upserts Agent/Session/Resource + ACCESSED/
        # IN_SESSION and derives ADJACENT_TO from container co-membership so the
        # caution and slow-and-low queries operate on real data. Runs before decay
        # so freshly created nodes carry a current last_decay_at.
        construct.upsert_graph(
            self._neo4j,
            sessions,
            seed_scores,
            cmembers,
            flag_threshold=FLAG_THRESHOLD,
        )

        # Step 1 — apply time decay first (invariant A-4). Decay runs entirely in
        # Neo4j against each resource's own last_decay_at; no in-memory decay exists
        # to drift from it or to risk an A-3 violation.
        q.apply_decay(self._neo4j, flag_threshold=FLAG_THRESHOLD)

        # Window the poisoning-mitigation counters BEFORE this cycle reads them:
        # dampening and diversity must describe the same window the evidence
        # does. Unwindowed, a year of old signals kept origin_dampening at
        # 0.5^n ≈ 0 for good — the actor's fresh signals weighed nothing
        # although every fact behind the count had long expired.
        self._prune_counters(now)

        self._reputation_events.clear()
        self._fanout_signals.clear()
        # rid → fanout fact for this cycle's verdict bump (session-scoped burst;
        # the windowed staging predicates P3/P4 carry the cross-cycle memory).
        fanout_facts: dict[str, str] = {}

        # Step 2 — process each tainted session
        for session in sessions:
            if not session.had_taint:
                continue

            # 2b — fanout detection
            fanout = self._check_fanout(session, now)
            if fanout is not None:
                self._fanout_signals.append(fanout)
                fact = (
                    f"F1:fanout:session={session.session_id}"
                    f":containers={fanout.unique_containers}"
                )
                for rid in fanout.affected_resources:
                    fanout_facts.setdefault(rid, fact)

            # Poisoning-mitigation factors key on the actor identity (source_class or
            # agent_id), NOT the attacker-controllable taint_source label — rotating
            # that label must not reset dampening/diversity (F1).
            origin = session.mitigation_origin

            # Deterministic verdict layer: record this session's typed facts.
            # Dedup by (session, rank) inside the store; predicates count
            # distinct origins/sessions, so replays cannot inflate verdicts.
            for rid, ev in evidence_from_session(
                origin=origin,
                session_id=session.session_id,
                started_at=session.started_at,
                tainted=session.had_taint,
                accesses=session.accessed_resources,
                # knowledge time: attestation supersession compares against
                # max(session start, this), so a late-reported session that
                # started before an attestation still ends its discount
                ingested_at=now,
            ):
                self._evidence.add(rid, ev)

            # 2c — apply hot weights per accessed resource.
            # One application per (session, resource, signal): that is the key
            # of the graph's ACCESSED edge (construct.UPSERT_ACCESS_QUERY), so
            # the same resource+signal listed under two containers is ONE
            # access. _dedupe_sessions keeps the container in its key (fanout
            # and adjacency need membership), and without this guard such a
            # pair got two apply_hot_weight calls and bumped the dampening /
            # diversity counters twice.
            applied: set[tuple[str, SignalType]] = set()
            for access in session.accessed_resources:
                rid = access.resource_id
                # Defensive: "" is "no resource" (graph.normalizer); producers
                # no longer emit it, and a weight on it would land on nothing
                # while still bumping the counters.
                if not rid or (rid, access.signal_type) in applied:
                    continue
                applied.add((rid, access.signal_type))
                raw_weight = compute_hot_weight(access.signal_type)
                history = self._signal_history.get(rid, [])
                prior = self._prior_counts.get((rid, origin), 0)
                # Single source for both weight views — the in-memory `effective`
                # (fed to accumulate) and the Cypher `without_confidence` (the query
                # multiplies by r.canonical_confidence) can no longer drift.
                wf = compute_weight_factors(
                    raw_weight=raw_weight,
                    canonical_confidence=access.canonical_confidence,
                    signal_history=history,
                    current_source=origin,
                    prior_count_from_source=prior,
                )
                eff_weight = wf.effective

                # Apply the hot weight in Neo4j (the authoritative store) and take
                # the before/after score straight from it — no parallel in-memory
                # accumulate to drift from. The Cypher multiplies $raw_weight by
                # r.canonical_confidence, so we hand it wf.without_confidence
                # (raw * diversity * dampening). The fanout flat weight is applied
                # separately below (A-10) and lands in the read-back snapshot.
                result = q.apply_hot_weight(
                    self._neo4j,
                    session_id=session.session_id,
                    signal_type=access.signal_type.value,
                    raw_weight=wf.without_confidence,
                    flag_threshold=FLAG_THRESHOLD,
                    resource_id=rid,
                )

                # Record evidence when the write matched (real graph). score_after
                # is the post-hot value; the fanout contribution is evidenced by the
                # FanoutSignal, not folded into this per-signal event.
                if result is not None:
                    score_before, score_after = result
                    event = ReputationEvent.create(
                        resource_id=rid,
                        session_id=session.session_id,
                        taint_source=session.taint_source,
                        signal_type=access.signal_type.value,
                        raw_weight=raw_weight,
                        effective_weight=eff_weight,
                        score_before=score_before,
                        score_after=score_after,
                        reason=(
                            f"hot signal {access.signal_type.value} "
                            f"from tainted session {session.session_id}"
                        ),
                        timestamp=now,
                    )
                    self._reputation_events.append(event)

                # Update signal history and prior counts — keyed on the actor origin
                # (see `origin` above), so source-label rotation cannot reset them.
                # Each entry is stamped with the cycle clock (when the weight was
                # applied) so _prune_counters can expire it with the window.
                self._signal_history.setdefault(rid, []).append(origin)
                self._signal_history_at.setdefault(rid, []).append(now)
                key = (rid, origin)
                self._prior_counts[key] = self._prior_counts.get(key, 0) + 1
                self._prior_counts_at.setdefault(key, []).append(now)

            # 2d(fanout) — write fanout flat weight to Neo4j (invariant A-10).
            # Applied as a separate accumulate on top of the hot weights; it lands
            # in the read-back snapshot below since that reads Neo4j after all writes.
            if fanout is not None:
                q.apply_fanout_weight(
                    self._neo4j,
                    resource_ids=fanout.affected_resources,
                    fanout_weight=FANOUT_WEIGHT,
                    flag_threshold=FLAG_THRESHOLD,
                )

            # 2e — caution weights to adjacent resources
            q.apply_caution_adjacent(
                self._neo4j,
                session_id=session.session_id,
                flag_threshold=FLAG_THRESHOLD,
            )

        # Read scores back from Neo4j — the authoritative store. This is what folds
        # decay, hot weights, the fanout boost AND caution (which is written only to
        # the graph, never computed in Python) into one consistent snapshot.
        final_scores = q.read_resource_scores(self._neo4j)

        # Windowed evidence is final for this cycle once the sessions above
        # have been folded in; prune it now so attestation supersession (just
        # below) and the verdict layer read the SAME fact set.
        self._evidence.prune(now, self._policy.window_days)

        # Which attestation applies per resource this cycle: the newest active
        # (unrevoked) one, unless evidence newer than it has arrived — then it
        # is superseded and discounts nothing (attestation.is_superseded).
        applied_attestations: dict[str, AttestationRecord] = {}
        superseded_attestations: dict[str, AttestationRecord] = {}
        for rid, records in self._attestations.items():
            active = active_attestation(records)
            if active is None:
                continue
            # known_at = max(session start, ingest time): what the operator
            # could have seen at created_at is decided by when Sentinel learned
            # of a fact, not only when the session began (Evidence docstring).
            if is_superseded(
                active, (e.known_at for e in self._evidence.evidence_for(rid))
            ):
                superseded_attestations[rid] = active
            else:
                applied_attestations[rid] = active

        # Apply operator attestations as a downward recompute over the read-back
        # scores (UI spec 8.1.1): an attested branch reads its post-attestation
        # residue. Neo4j is untouched — the evidence stays, only the exported
        # reputation descends. Container scores below fold this in for free.
        # A superseded attestation leaves the raw score standing, same as the
        # level path below.
        if applied_attestations:
            final_scores = {
                rid: (
                    effective_score(score, [applied_attestations[rid]])
                    if rid in applied_attestations else score
                )
                for rid, score in final_scores.items()
            }

        # 2f — recompute container scores from the read-back scores (invariant A-9).
        container_scores: dict[str, float] = {}
        for cid, member_ids in cmembers.items():
            member_scores = [final_scores.get(rid, 0.0) for rid in member_ids]
            container_scores[cid] = compute_container_score(member_scores)

        # Deterministic verdict layer (dual-run): windowed evidence → decidable
        # levels + facts, published alongside the scalar maps. The scalar path
        # above stays authoritative for the wire values in this phase; the
        # levels are the predicate verdicts being validated against it.
        # (Evidence was pruned above, before the attestation step.)
        resource_verdicts: dict[str, Verdict] = {
            rid: evaluate_resource(self._evidence.evidence_for(rid), self._policy, now)
            for rid in self._evidence.resource_ids()
        }
        # Fanout floor: every resource touched by a quota-exceeding session is
        # at least WATCH this cycle, with the fanout fact attached.
        for rid, fact in fanout_facts.items():
            v = resource_verdicts.get(rid, Verdict(ReputationLevel.CLEAN))
            level = v.level if v.level >= ReputationLevel.WATCH else ReputationLevel.WATCH
            resource_verdicts[rid] = Verdict(level, v.facts + (fact,))

        resource_levels = {rid: v.level for rid, v in resource_verdicts.items()}

        # Deterministic adjacency (replaces the numeric caution bleed): sharing
        # a container with a FLAGGED resource is a structural fact worth WATCH —
        # a label, not an arithmetic contribution.
        for cid, member_ids in cmembers.items():
            if any(
                resource_levels.get(r, ReputationLevel.CLEAN) == ReputationLevel.FLAGGED
                for r in member_ids
            ):
                for r in member_ids:
                    if resource_levels.get(r, ReputationLevel.CLEAN) < ReputationLevel.WATCH:
                        prior_verdict = resource_verdicts.get(
                            r, Verdict(ReputationLevel.CLEAN)
                        )
                        resource_verdicts[r] = Verdict(
                            ReputationLevel.WATCH,
                            prior_verdict.facts + (f"A1:adjacent_to_flagged:{cid}",),
                        )

        # Operator attestations in the level codomain (UI spec 8.1.1): an
        # active (unrevoked) attestation descends the EXPORTED verdict one
        # level — but only while no evidence newer than the attestation exists
        # for the resource. It vouches for what the operator saw, not for
        # whatever comes next: re-descending on every cycle turned one
        # attestation into a permanent one-level discount, so a fresh
        # export-denied FLAGGED read WATCH (0.4, under core's 0.3 floor) and
        # never re-flagged. With newer evidence the full verdict is exported
        # and the superseded attestation is named in the facts — history stays
        # either way (append-only; evidence windows and Neo4j are untouched):
        # "I checked, resume watching", never "trust this forever". The
        # fact names the APPLIED attestation, not records[0], which may be a
        # revocation record. The scalar effective_score above applied the
        # same decision to the telemetry map.
        for rid, active in applied_attestations.items():
            attested = resource_verdicts.get(rid)
            if attested is None or attested.level == ReputationLevel.CLEAN:
                continue
            resource_verdicts[rid] = Verdict(
                ReputationLevel(attested.level - 1),
                attested.facts + (f"A2:attested:{active.attestation_id}",),
            )
        for rid, active in superseded_attestations.items():
            sv = resource_verdicts.get(rid)
            if sv is None or sv.level == ReputationLevel.CLEAN:
                continue
            resource_verdicts[rid] = Verdict(
                sv.level,
                sv.facts + (
                    f"A2:attestation_superseded_by_newer_evidence:{active.attestation_id}",
                ),
            )
        resource_levels = {rid: v.level for rid, v in resource_verdicts.items()}

        container_levels: dict[str, ReputationLevel] = {
            cid: evaluate_container(
                (resource_levels.get(r, ReputationLevel.CLEAN) for r in member_ids),
                self._policy,
            ).level
            for cid, member_ids in cmembers.items()
        }

        # Step 3 — write the new snapshot (invariant A-5).
        self._current_version += 1
        # The wire values are DERIVED from the decidable levels (finite
        # codomain, covered by the checksum); the scalar accumulate/decay maps
        # are demoted to telemetry fields.
        snapshot = ReputationSnapshot(
            version=self._current_version,
            generated_at=now,
            resource_reputation={
                rid: LEVEL_SUSPICION[lvl]
                for rid, lvl in resource_levels.items()
                if lvl > ReputationLevel.CLEAN
            },
            container_reputation={
                cid: LEVEL_SUSPICION[lvl]
                for cid, lvl in container_levels.items()
                if lvl > ReputationLevel.CLEAN
            },
            resource_score_telemetry=final_scores,
            container_score_telemetry=container_scores,
            # canonical level names, the vocabulary the wire validates
            # against (`snapshot_from_payload`); lower-cased they were refused
            resource_level={
                rid: lvl.name for rid, lvl in resource_levels.items()
                if lvl > ReputationLevel.CLEAN
            },
            container_level={
                cid: lvl.name for cid, lvl in container_levels.items()
                if lvl > ReputationLevel.CLEAN
            },
            verdict_facts={
                rid: list(v.facts) for rid, v in resource_verdicts.items() if v.facts
            },
        ).with_checksum()

        # Crash-consistency: persist state (which records _current_version) BEFORE
        # making the snapshot visible. If we crash in between, state is AHEAD of the
        # snapshot — the next run derives a fresh higher version — rather than behind
        # it, which would re-emit THIS version with different content (a consumer
        # would see two distinct snapshots at the same version).
        #
        # If the save FAILS (logged at error level, not raised) the cycle still
        # publishes. That is safe for the version sequence: the old state file is
        # intact (writes are atomic) but behind, and on restart __init__ resumes
        # at max(state version, latest_snapshot_version(dir)) — the snapshot
        # written below is itself the witness, so no version is ever reused.
        # What a failed save costs is only the counters/evidence gathered since
        # the last good save, which is the cold-start trade the design already
        # accepts; withholding the snapshot would cost the node its reputation.
        self._save_state_locked()
        atomic_swap(self._snapshot_dir, snapshot)
        self._publish_snapshot(snapshot)

        log.info(
            "sentinel cycle complete: version=%d resources=%d containers=%d events=%d",
            self._current_version,
            len(final_scores),
            len(container_scores),
            len(self._reputation_events),
        )

        return snapshot

    # ── Fanout detection ───────────────────────────────────────────────────────

    def _check_fanout(
        self,
        session: SessionSummary,
        now: float,
    ) -> FanoutSignal | None:
        """
        Deterministic fanout quota (declared policy) — replaces the self-trained
        z-score baseline as the trigger. A tainted session touching more than
        policy.fanout_containers DISTINCT containers at rank >= READ_SUMMARIZE
        is a fanout fact: exact counting against a declared quota. No cold
        start (a quota needs no history) and no baseline an attacker can walk
        upward — closes limitation F5 by construction.

        The taint and signal-rank gates (invariant A-15) are unchanged. The
        z-score against the smoothed per-agent baseline is still computed on an
        emitted signal, but as TELEMETRY only (0.0 when no baseline exists) —
        it never gates the trigger. ``unique_containers`` is the qualifying
        count the quota compared; the z-score keeps the all-containers count,
        the measure update_baseline records, so it compares like with like.
        """
        # Only containers touched at rank >= READ_SUMMARIZE count, and "" (no
        # container) never does — fanout_containers. Counting every container
        # let 8 plain READs plus one READ_SUMMARIZE fire the quota.
        containers = fanout_containers(
            (a.container_id, a.signal_type) for a in session.accessed_resources
        )
        signal_values = [a.signal_type for a in session.accessed_resources]
        max_signal = max(signal_values) if signal_values else None
        if not fanout_exceeded(
            session.had_taint, containers, max_signal, self._policy,
            source_class=session.source_class,
        ):
            return None

        baseline = self._baselines.get(session.agent_id)
        mean = baseline.mean_containers_per_session if baseline is not None else 0.0
        touched = {a.container_id for a in session.accessed_resources}
        if baseline is not None and baseline.std_containers_per_session >= 0.01:
            z_score = (len(touched) - mean) / baseline.std_containers_per_session
        else:
            z_score = 0.0

        return FanoutSignal(
            origin_session_id=session.session_id,
            agent_id=session.agent_id,
            taint_source=session.taint_source,
            # Every resource the session touched (A-10: the flat weight lands
            # on all of them, not just the qualifying ones), once each and
            # never "": one resource under two containers/signals is one
            # fanout write, not two.
            affected_resources=list(dict.fromkeys(
                a.resource_id for a in session.accessed_resources if a.resource_id
            )),
            unique_containers=len(containers),
            baseline_mean=mean,
            z_score=z_score,
            window_minutes=0.0,
        )

    # ── State persistence ──────────────────────────────────────────────────────

    def save_state(self) -> None:
        """
        Persist signal_history, prior_counts, and baselines to disk.

        Written to ``sentinel_state.json`` in snapshot_dir.  Called automatically
        at the end of every ``run_once()`` so poisoning-mitigation counters and
        agent baselines survive process restarts.

        Thread-safe: takes the cycle lock, so it cannot serialise the dicts while
        run_once or update_baseline is mutating them. Must not be called from
        inside run_once (the lock is not re-entrant) — that path uses
        _save_state_locked.

        Failures are logged at ERROR and swallowed — a missing state file is
        recoverable (cold-start behaviour, and the version is recovered from the
        snapshot directory); a crash during save must not abort the cycle.
        """
        with self._lock:
            self._save_state_locked()

    def _save_state_locked(self) -> None:
        """save_state body; the caller holds ``self._lock``.

        The file is replaced atomically (write_file_atomic: temp + fsync +
        os.replace), so a crash mid-save leaves the previous state intact
        instead of a truncated file that the next start would read as a cold
        start at version 0.
        """
        try:
            # A caller may have written the counter dicts directly (tests,
            # seeding); give every entry its time so the file is consistent.
            self._reconcile_counter_times(time.time())
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            state: dict = {
                "version": self._current_version,
                "signal_history": self._signal_history,
                # Per-entry apply times for the windowed counters, parallel to
                # signal_history[rid] / one per prior_counts increment. Separate
                # keys (not a new shape for the old ones) so an older reader
                # still parses this file and a newer one reads an older file
                # (missing times → dated at load, _reconcile_counter_times).
                "signal_history_at": self._signal_history_at,
                "prior_counts_at": {
                    f"{rid}\x00{src}": times
                    for (rid, src), times in self._prior_counts_at.items()
                },
                # tuple keys are not JSON-serialisable — encode as "rid\x00src"
                "prior_counts": {
                    f"{rid}\x00{src}": count
                    for (rid, src), count in self._prior_counts.items()
                },
                "baselines": {
                    aid: dataclasses.asdict(b)
                    for aid, b in self._baselines.items()
                },
                # Deterministic verdict layer: windowed evidence sets survive
                # restarts inside the same signed envelope.
                "evidence": self._evidence.to_json(),
                # Operator attestations (full history, newest first per
                # resource) ride in the same signed envelope: a forged or
                # injected attestation would lower exported reputation, so it
                # needs exactly the integrity the counters get. Never pruned —
                # history stays.
                "attestations": {
                    rid: [r.to_json() for r in records]
                    for rid, records in self._attestations.items()
                },
            }
            state_file = self._snapshot_dir / "sentinel_state.json"
            serialized = json.dumps(state, sort_keys=True, separators=(",", ":"))
            # Authenticate state when a key is configured: poisoned baselines /
            # prior_counts / signal_history would silently corrupt fanout and
            # dampening after restart.
            sig = sign_blob(serialized)
            if sig is not None:
                out = json.dumps(
                    {"_signed": True, "payload": serialized, "sig": sig},
                    separators=(",", ":"),
                )
            else:
                out = serialized
            write_file_atomic(state_file, out)
            log.debug("sentinel state saved: version=%d", self._current_version)
        except Exception:  # noqa: BLE001 — see save_state / run_once
            log.error(
                "sentinel: failed to save state (version=%d); counters since the "
                "last good save will be lost on restart",
                self._current_version, exc_info=True,
            )

    @staticmethod
    def load_state(
        state_path: Path,
    ) -> tuple[
        dict[str, list[str]],
        dict[tuple[str, str], int],
        dict[str, AgentContainerBaseline],
        int,
        EvidenceStore,
    ]:
        """
        Load persisted sentinel state from ``state_path``.

        Returns ``(signal_history, prior_counts, baselines, version, evidence)``.
        Returns empty dicts and version=0 if the file does not exist or is corrupt.
        Attestations live in the same file; read them with load_attestations
        (kept out of this tuple so existing unpacking callers keep working).
        """
        return SentinelCycle._parse_state(SentinelCycle._read_state_payload(state_path))

    @staticmethod
    def load_attestations(state_path: Path) -> dict[str, list[AttestationRecord]]:
        """Persisted operator attestations from ``state_path``, keyed by
        resource_id, newest first. Same authentication as load_state; a
        missing, corrupt, unauthenticated or pre-attestation state file yields
        ``{}``."""
        return SentinelCycle._parse_attestations(
            SentinelCycle._read_state_payload(state_path)
        )

    @staticmethod
    def _read_state_payload(state_path: Path) -> dict | None:
        """Read and authenticate the state file; the decoded dict, or None for
        a missing / corrupt / unauthenticated file (cold start)."""
        if not state_path.exists():
            return None
        try:
            text = state_path.read_text(encoding="utf-8")
            obj = json.loads(text)
        except Exception as exc:
            log.warning("sentinel: failed to load state from %s: %s", state_path, exc)
            return None

        # Authenticate before trusting. Signed envelope → verify HMAC; legacy
        # flat state → accept only when no key/signature is required (else cold
        # start). A failed check resets to baseline rather than loading poisoned
        # counters.
        if isinstance(obj, dict) and obj.get("_signed"):
            serialized = obj.get("payload", "")
            if not verify_blob(serialized, obj.get("sig")):
                log.warning("sentinel: state signature invalid — cold start")
                return None
            try:
                raw = json.loads(serialized)
            except Exception:
                return None
        else:
            if not verify_blob(text, None):
                log.warning(
                    "sentinel: unsigned state rejected (key/signature required) — cold start"
                )
                return None
            raw = obj
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _parse_state(
        raw: dict | None,
    ) -> tuple[
        dict[str, list[str]],
        dict[tuple[str, str], int],
        dict[str, AgentContainerBaseline],
        int,
        EvidenceStore,
    ]:
        """Counters/baselines/version/evidence from an authenticated payload."""
        if raw is None:
            return {}, {}, {}, 0, EvidenceStore()

        signal_history: dict[str, list[str]] = raw.get("signal_history", {})

        prior_counts: dict[tuple[str, str], int] = {}
        for key, count in raw.get("prior_counts", {}).items():
            rid, sep, src = key.partition("\x00")
            if sep:
                prior_counts[(rid, src)] = int(count)

        baselines: dict[str, AgentContainerBaseline] = {}
        for aid, bdata in raw.get("baselines", {}).items():
            # schema mismatch after upgrade — skip the stale entry
            with contextlib.suppress(TypeError):
                baselines[aid] = AgentContainerBaseline(**bdata)

        version = int(raw.get("version", 0))
        evidence = EvidenceStore.from_json(raw.get("evidence", {}))
        return signal_history, prior_counts, baselines, version, evidence

    @staticmethod
    def _parse_attestations(raw: dict | None) -> dict[str, list[AttestationRecord]]:
        """Attestations from an authenticated payload. State written before
        attestations were persisted has no key → ``{}`` (backward compatible).
        Each record is re-run through validate(): one that no longer passes
        (e.g. an older, laxer writer) is skipped and logged rather than
        failing the whole restore — the rest of the history still applies.
        Order is preserved (newest first, as attest() keeps it)."""
        if raw is None:
            return {}
        stored = raw.get("attestations", {})
        if not isinstance(stored, dict):
            return {}
        out: dict[str, list[AttestationRecord]] = {}
        for rid, items in stored.items():
            if not isinstance(items, list):
                continue
            for item in items:
                try:
                    record = AttestationRecord.from_json(item)
                    validate(record)
                except Exception as exc:  # noqa: BLE001 — skip one bad record
                    log.warning(
                        "sentinel: skipping unreadable persisted attestation "
                        "for %s: %s", rid, exc,
                    )
                    continue
                out.setdefault(record.resource_id, []).append(record)
        return out

    @staticmethod
    def _parse_counter_times(
        raw: dict | None,
    ) -> tuple[dict[str, list[float]], dict[tuple[str, str], list[float]]]:
        """Per-entry apply times of the windowed counters from an authenticated
        payload. State written before the counters were windowed has no such
        keys → ``({}, {})``; the entries it does carry are then dated at load
        by _reconcile_counter_times (backward compatible, and they still expire
        one window later). Malformed values are dropped the same way."""
        if raw is None:
            return {}, {}
        sh_at: dict[str, list[float]] = {}
        stored = raw.get("signal_history_at", {})
        if isinstance(stored, dict):
            for rid, times in stored.items():
                with contextlib.suppress(TypeError, ValueError):
                    sh_at[str(rid)] = [float(t) for t in times]
        pc_at: dict[tuple[str, str], list[float]] = {}
        stored = raw.get("prior_counts_at", {})
        if isinstance(stored, dict):
            for key, times in stored.items():
                rid, sep, src = str(key).partition("\x00")
                if not sep:
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    pc_at[(rid, src)] = [float(t) for t in times]
        return sh_at, pc_at

    def _reconcile_counter_times(self, now: float) -> None:
        """Make every counter entry carry exactly one apply time.

        signal_history[rid] and _signal_history_at[rid] are parallel lists;
        prior_counts[key] is the length of _prior_counts_at[key]. Entries that
        arrived without a time — explicit constructor seeds, state from before
        the counters were windowed, or a caller writing the dicts directly —
        are dated ``now``. The times are prepended, because the undated
        entries are the oldest ones (the front of each list); dating them now
        is the conservative choice — a count is never dropped early, only kept
        for at most one more window. Surplus
        times (a count that was lowered by hand) are trimmed oldest-first.
        Keys whose count reached zero are removed.
        """
        for rid, history in self._signal_history.items():
            times = self._signal_history_at.setdefault(rid, [])
            if len(times) < len(history):
                times[:0] = [now] * (len(history) - len(times))
            elif len(times) > len(history):
                del times[: len(times) - len(history)]
        for rid in list(self._signal_history_at):
            if rid not in self._signal_history:
                del self._signal_history_at[rid]
        for key, count in list(self._prior_counts.items()):
            if count <= 0:
                del self._prior_counts[key]
                continue
            times = self._prior_counts_at.setdefault(key, [])
            if len(times) < count:
                times[:0] = [now] * (count - len(times))
            elif len(times) > count:
                del times[: len(times) - count]
        for key in list(self._prior_counts_at):
            if key not in self._prior_counts:
                del self._prior_counts_at[key]

    def _prune_counters(self, now: float) -> None:
        """Drop counter entries applied before the evidence window.

        Uses the same length as the evidence window (policy.window_days), so
        the dampening / diversity factors describe the same period the verdict
        layer does. The clock is the time the cycle APPLIED the weight (not the
        session start): the counters record weight applications, which happen
        at ingest. Called with the lock held, at the start of each cycle.
        """
        self._reconcile_counter_times(now)
        horizon = now - self._policy.window_days * 86400.0
        for rid in list(self._signal_history):
            pairs = [
                (o, t) for o, t in zip(
                    self._signal_history[rid], self._signal_history_at[rid],
                    strict=True,
                )
                if t >= horizon
            ]
            if pairs:
                self._signal_history[rid] = [o for o, _ in pairs]
                self._signal_history_at[rid] = [t for _, t in pairs]
            else:
                del self._signal_history[rid]
                del self._signal_history_at[rid]
        for key in list(self._prior_counts):
            kept = [t for t in self._prior_counts_at[key] if t >= horizon]
            if kept:
                self._prior_counts[key] = len(kept)
                self._prior_counts_at[key] = kept
            else:
                del self._prior_counts[key]
                del self._prior_counts_at[key]

    def update_baseline(
        self,
        agent_id: str,
        recent_sessions: list[SessionSummary],
    ) -> None:
        """
        Recompute AgentContainerBaseline from the last BASELINE_WINDOW_SESSIONS sessions.

        Called after each completed session. Exponential smoothing prevents a single
        anomalous session from sharply shifting the baseline.

        Takes the cycle lock: callers run this from session-completion hooks on
        other threads, concurrently with run_once / save_state iterating
        ``_baselines``.
        """
        if len(recent_sessions) < 2:
            return
        window = recent_sessions[-BASELINE_WINDOW_SESSIONS:]
        counts = [
            len({a.container_id for a in s.accessed_resources})
            for s in window
        ]
        n = len(counts)
        mean = sum(counts) / n
        # Sample variance (Bessel's correction: divide by n-1) gives an unbiased
        # std estimate.  max(n-1, 1) avoids ZeroDivisionError when n=1.
        variance = sum((c - mean) ** 2 for c in counts) / max(n - 1, 1)
        std = math.sqrt(variance) if variance > 0 else 0.0

        # Read-modify-write of the existing baseline under the lock, so two
        # concurrent updates for one agent cannot lose one's smoothing step.
        with self._lock:
            existing = self._baselines.get(agent_id)
            if existing is not None:
                # Exponential smoothing: blend new stats with existing baseline
                alpha = 0.3
                mean = alpha * mean + (1 - alpha) * existing.mean_containers_per_session
                std = alpha * std + (1 - alpha) * existing.std_containers_per_session

            self._baselines[agent_id] = AgentContainerBaseline(
                agent_id=agent_id,
                mean_containers_per_session=mean,
                std_containers_per_session=std,
                session_count=n,
                last_updated=time.time(),
            )
