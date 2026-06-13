from __future__ import annotations

import dataclasses
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axor_sentinel.graph.model import SignalType
from axor_sentinel.graph import construct
from axor_sentinel.graph import queries as q
from axor_sentinel.sentinel.events import (
    AgentContainerBaseline,
    FanoutSignal,
    ReputationEvent,
)
from axor_sentinel.sentinel.snapshot import (
    ReputationSnapshot,
    atomic_swap,
    sign_blob,
    validate_snapshot_dir,
    verify_blob,
)
from axor_sentinel.sentinel.weight import (
    FLAG_THRESHOLD,
    compute_weight_factors,
    compute_hot_weight,
    compute_container_score,
)

log = logging.getLogger("axor.sentinel.cycle")

# Fanout detection parameters
FANOUT_Z_THRESHOLD: float = 2.5
FANOUT_MIN_SESSIONS: int = 10       # cold-start guard (invariant A-14)
FANOUT_MIN_DELTA: int = 3           # absolute minimum above mean for low-std agents
BASELINE_WINDOW_SESSIONS: int = 50  # sessions used to recompute baseline
FANOUT_WEIGHT: float = 0.5          # flat weight added to all touched resources (A-10)

# Signal type that determines minimum threshold for fanout trigger (A-15)
_FANOUT_MIN_SIGNAL = SignalType.READ_SUMMARIZE


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
    ) -> None:
        """
        Args:
            neo4j_session:    live neo4j.Session for graph operations
            snapshot_dir:     directory for atomic snapshot writes
            agent_baselines:  agent_id → AgentContainerBaseline (updated in-place)
            signal_history:   resource_id → list[taint_source] (for diversity factor)
            prior_counts:     (resource_id, taint_source) → count (for dampening)
        """
        self._neo4j = neo4j_session
        self._snapshot_dir = Path(snapshot_dir)
        self._reputation_events: list[ReputationEvent] = []
        self._fanout_signals: list[FanoutSignal] = []
        # Serialise cycles: run_once mutates _signal_history / _prior_counts /
        # _baselines / _current_version, none of which is safe under overlap.
        self._lock = threading.Lock()

        # If explicit state is provided (tests / controlled init), use it directly.
        # Otherwise try to restore persisted state from disk so poisoning-mitigation
        # counters and agent baselines survive process restarts.
        if agent_baselines is not None or signal_history is not None or prior_counts is not None:
            self._baselines: dict[str, AgentContainerBaseline] = agent_baselines or {}
            self._signal_history: dict[str, list[str]] = signal_history or {}
            self._prior_counts: dict[tuple[str, str], int] = prior_counts or {}
            self._current_version: int = 0
        else:
            sh, pc, bl, ver = SentinelCycle.load_state(
                self._snapshot_dir / "sentinel_state.json"
            )
            self._baselines = bl
            self._signal_history = sh
            self._prior_counts = pc
            self._current_version = ver
            if ver > 0:
                log.info("sentinel: restored persisted state version=%d", ver)

        # Warn early if snapshot_dir is a network mount (invariant A-17).
        # Done in __init__ so operators learn about the misconfiguration at startup,
        # not at the first write hours later.
        validate_snapshot_dir(self._snapshot_dir)

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
        (deduped by resource_id + container_id + signal_type so a repeated access
        is not weighted twice), started_at is the earliest, and source_class is
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

        self._reputation_events.clear()
        self._fanout_signals.clear()

        # Step 2 — process each tainted session
        for session in sessions:
            if not session.had_taint:
                continue

            # 2b — fanout detection
            fanout = self._check_fanout(session, now)
            if fanout is not None:
                self._fanout_signals.append(fanout)

            # Poisoning-mitigation factors key on the actor identity (source_class or
            # agent_id), NOT the attacker-controllable taint_source label — rotating
            # that label must not reset dampening/diversity (F1).
            origin = session.mitigation_origin

            # 2c — apply hot weights per accessed resource
            for access in session.accessed_resources:
                rid = access.resource_id
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
                self._signal_history.setdefault(rid, []).append(origin)
                key = (rid, origin)
                self._prior_counts[key] = self._prior_counts.get(key, 0) + 1

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

        # 2f — recompute container scores from the read-back scores (invariant A-9).
        container_scores: dict[str, float] = {}
        for cid, member_ids in cmembers.items():
            member_scores = [final_scores.get(rid, 0.0) for rid in member_ids]
            container_scores[cid] = compute_container_score(member_scores)

        # Step 3 — write the new snapshot (invariant A-5).
        self._current_version += 1
        snapshot = ReputationSnapshot(
            version=self._current_version,
            generated_at=now,
            resource_reputation=final_scores,
            container_reputation=container_scores,
        ).with_checksum()

        # Crash-consistency: persist state (which records _current_version) BEFORE
        # making the snapshot visible. If we crash in between, state is AHEAD of the
        # snapshot — the next run derives a fresh higher version — rather than behind
        # it, which would re-emit THIS version with different content (a consumer
        # would see two distinct snapshots at the same version).
        self.save_state()
        atomic_swap(self._snapshot_dir, snapshot)

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
        Check whether the session's container access pattern exceeds the agent's
        historical baseline by FANOUT_Z_THRESHOLD standard deviations.

        All three conditions must be true (invariant A-15):
          1. session.had_taint
          2. z_score > FANOUT_Z_THRESHOLD
          3. max signal_type >= READ_SUMMARIZE
        Cold start: returns None if baseline.session_count < FANOUT_MIN_SESSIONS (A-14).
        """
        # Invariant A-15: had_taint is required for fanout detection
        if not session.had_taint:
            return None

        baseline = self._baselines.get(session.agent_id)
        # Cold-start guard (invariant A-14)
        if baseline is None or baseline.session_count < FANOUT_MIN_SESSIONS:
            return None

        # Determine unique containers touched
        containers = {a.container_id for a in session.accessed_resources}
        unique_containers = len(containers)

        # Signal type gate (invariant A-15): need at least READ_SUMMARIZE.
        # max() uses SignalType.__gt__ which compares by explicit _rank — safe
        # against enum reordering.
        signal_values = [a.signal_type for a in session.accessed_resources]
        if not signal_values:
            return None
        max_signal = max(signal_values)
        if max_signal < _FANOUT_MIN_SIGNAL:
            return None

        # z-score calculation
        std = baseline.std_containers_per_session
        mean = baseline.mean_containers_per_session

        # Low-std guard: require absolute delta > FANOUT_MIN_DELTA
        if std < 0.01:
            if unique_containers <= mean + FANOUT_MIN_DELTA:
                return None
            z_score = float("inf")
        else:
            z_score = (unique_containers - mean) / std

        if z_score <= FANOUT_Z_THRESHOLD:
            return None

        affected = [a.resource_id for a in session.accessed_resources]
        return FanoutSignal(
            origin_session_id=session.session_id,
            agent_id=session.agent_id,
            taint_source=session.taint_source,
            affected_resources=affected,
            unique_containers=unique_containers,
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

        Failures are logged and swallowed — a missing state file is recoverable
        (cold-start behaviour); a crash during save must not abort the cycle.
        """
        try:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
            state: dict = {
                "version": self._current_version,
                "signal_history": self._signal_history,
                # tuple keys are not JSON-serialisable — encode as "rid\x00src"
                "prior_counts": {
                    f"{rid}\x00{src}": count
                    for (rid, src), count in self._prior_counts.items()
                },
                "baselines": {
                    aid: dataclasses.asdict(b)
                    for aid, b in self._baselines.items()
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
            state_file.write_text(out, encoding="utf-8")
            log.debug("sentinel state saved: version=%d", self._current_version)
        except Exception as exc:  # pragma: no cover
            log.warning("sentinel: failed to save state: %s", exc)

    @staticmethod
    def load_state(
        state_path: Path,
    ) -> tuple[
        dict[str, list[str]],
        dict[tuple[str, str], int],
        dict[str, AgentContainerBaseline],
        int,
    ]:
        """
        Load persisted sentinel state from ``state_path``.

        Returns ``(signal_history, prior_counts, baselines, version)``.
        Returns empty dicts and version=0 if the file does not exist or is corrupt.
        """
        if not state_path.exists():
            return {}, {}, {}, 0
        try:
            text = state_path.read_text(encoding="utf-8")
            obj = json.loads(text)
        except Exception as exc:
            log.warning("sentinel: failed to load state from %s: %s", state_path, exc)
            return {}, {}, {}, 0

        # Authenticate before trusting. Signed envelope → verify HMAC; legacy
        # flat state → accept only when no key/signature is required (else cold
        # start). A failed check resets to baseline rather than loading poisoned
        # counters.
        if isinstance(obj, dict) and obj.get("_signed"):
            serialized = obj.get("payload", "")
            if not verify_blob(serialized, obj.get("sig")):
                log.warning("sentinel: state signature invalid — cold start")
                return {}, {}, {}, 0
            try:
                raw = json.loads(serialized)
            except Exception:
                return {}, {}, {}, 0
        else:
            if not verify_blob(text, None):
                log.warning(
                    "sentinel: unsigned state rejected (key/signature required) — cold start"
                )
                return {}, {}, {}, 0
            raw = obj

        signal_history: dict[str, list[str]] = raw.get("signal_history", {})

        prior_counts: dict[tuple[str, str], int] = {}
        for key, count in raw.get("prior_counts", {}).items():
            rid, sep, src = key.partition("\x00")
            if sep:
                prior_counts[(rid, src)] = int(count)

        baselines: dict[str, AgentContainerBaseline] = {}
        for aid, bdata in raw.get("baselines", {}).items():
            try:
                baselines[aid] = AgentContainerBaseline(**bdata)
            except TypeError:
                pass  # schema mismatch after upgrade — skip stale entry

        version = int(raw.get("version", 0))
        return signal_history, prior_counts, baselines, version

    def update_baseline(
        self,
        agent_id: str,
        recent_sessions: list[SessionSummary],
    ) -> None:
        """
        Recompute AgentContainerBaseline from the last BASELINE_WINDOW_SESSIONS sessions.

        Called after each completed session. Exponential smoothing prevents a single
        anomalous session from sharply shifting the baseline.
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
