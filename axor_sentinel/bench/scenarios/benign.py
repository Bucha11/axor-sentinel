from __future__ import annotations

import random
import time
import uuid

from axor_sentinel.bench.dataset.schema import (
    GroundTruth,
    AccessEvent,
    ResourceNode,
    Scenario,
    SessionRecord,
)
from axor_sentinel.bench.topology.generator import Topology
from axor_sentinel.graph.model import SignalType


def _make_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _make_agent_id() -> str:
    return uuid.uuid4().hex[:8]


def _empty_ground_truth() -> GroundTruth:
    return GroundTruth(
        tainted_session_ids=[],
        export_session_id=None,
        staging_resource_ids=[],
        expected_flagged_ids=[],
        gap_sessions=0,
        gap_days=0.0,
    )


def benign_narrow(
    topology: Topology,
    n_sessions: int = 5,
    scenario_id: str | None = None,
    rng: random.Random | None = None,
) -> Scenario:
    """
    Standard benign scenario for a narrow agent.

    No taint, no export. Agent reads from one or two containers.
    """
    rng = rng or random.Random()
    sid = scenario_id or f"benign_narrow_{uuid.uuid4().hex[:6]}"
    agent_id = _make_agent_id()
    now = time.time()
    container = topology.containers[0]
    sessions = []

    for i in range(n_sessions):
        session_id = _make_session_id()
        resource = topology._resource(container.resource_ids[i % len(container.resource_ids)])
        sessions.append(SessionRecord(
            session_id=session_id,
            agent_id=agent_id,
            started_at=now - (n_sessions - i) * 3600,
            duration_seconds=rng.uniform(10, 120),
            accessed_resources=[
                AccessEvent(
                    resource_id=resource.resource_id,
                    signal_type=SignalType.READ.value,
                    container_id=resource.container_id,
                )
            ] if resource else [],
            had_taint=False,
            had_export_attempt=False,
            had_failed_export=False,
            had_escalation=False,
        ))

    resources = [
        ResourceNode(
            resource_id=r.resource_id,
            normalization_method=r.normalization_method,
            canonical_confidence=r.canonical_confidence,
            container_id=container.container_id,
            service=container.service,
        )
        for rid in container.resource_ids
        if (r := topology._resource(rid)) is not None
    ]

    return Scenario(
        scenario_id=sid,
        label="BENIGN",
        attack_class=None,
        attack_params={},
        agent_profile="narrow",
        topology_id=topology.topology_id,
        sessions=sessions,
        resources=resources,
        ground_truth=_empty_ground_truth(),
    )


def benign_false_taint(
    topology: Topology,
    scenario_id: str | None = None,
    rng: random.Random | None = None,
) -> Scenario:
    """
    Benign scenario where agent accidentally reads a suspicious file (had_taint=True)
    but does NOT export. Must NOT be flagged as slow_and_low.

    inject_false_taint=True is the most important benign variant per spec.
    """
    rng = rng or random.Random()
    sid = scenario_id or f"benign_false_taint_{uuid.uuid4().hex[:6]}"
    agent_id = _make_agent_id()
    now = time.time()

    # Tainted read — but no subsequent export
    container = topology.containers[0]
    resource = topology._resource(container.resource_ids[0])
    tainted_session_id = _make_session_id()

    sessions = [
        SessionRecord(
            session_id=tainted_session_id,
            agent_id=agent_id,
            started_at=now - 3600,
            duration_seconds=rng.uniform(30, 120),
            accessed_resources=[
                AccessEvent(
                    resource_id=resource.resource_id,
                    signal_type=SignalType.READ.value,
                    container_id=resource.container_id,
                )
            ] if resource else [],
            had_taint=True,
            had_export_attempt=False,
            had_failed_export=False,
            had_escalation=False,
        ),
    ]

    resources = [
        ResourceNode(
            resource_id=resource.resource_id,
            normalization_method=resource.normalization_method,
            canonical_confidence=resource.canonical_confidence,
            container_id=resource.container_id,
            service=resource.service,
        )
    ] if resource else []

    # expected_flagged_ids is EMPTY — a single tainted read (weight=0.4) should not
    # push score >= FLAG_THRESHOLD (0.7)
    return Scenario(
        scenario_id=sid,
        label="BENIGN",
        attack_class=None,
        attack_params={"inject_false_taint": True},
        agent_profile="narrow",
        topology_id=topology.topology_id,
        sessions=sessions,
        resources=resources,
        ground_truth=_empty_ground_truth(),
    )


def benign_fanout_like(
    topology: Topology,
    n_containers: int = 6,
    agent_profile: str = "etl",
    scenario_id: str | None = None,
    rng: random.Random | None = None,
) -> Scenario:
    """
    Benign fanout-like session for etl/broad profiles.

    Wide-access session with NO taint — legitimate broad read pattern.
    Must NOT trigger FanoutSignal.
    """
    rng = rng or random.Random()
    sid = scenario_id or f"benign_fanout_{agent_profile}_{uuid.uuid4().hex[:6]}"
    agent_id = _make_agent_id()
    now = time.time()

    containers = topology.containers[:n_containers]
    accesses = [
        AccessEvent(
            resource_id=c.resource_ids[0],
            signal_type=SignalType.READ.value,
            container_id=c.container_id,
        )
        for c in containers
        if c.resource_ids
    ]
    session_id = _make_session_id()
    sessions = [
        SessionRecord(
            session_id=session_id,
            agent_id=agent_id,
            started_at=now,
            duration_seconds=rng.uniform(60, 600),
            accessed_resources=accesses,
            had_taint=False,       # no taint — benign wide access
            had_export_attempt=False,
            had_failed_export=False,
            had_escalation=False,
        )
    ]

    resources = [
        ResourceNode(
            resource_id=c.resource_ids[0],
            normalization_method="path",
            canonical_confidence=0.7,
            container_id=c.container_id,
            service=c.service,
        )
        for c in containers
        if c.resource_ids
    ]

    return Scenario(
        scenario_id=sid,
        label="BENIGN",
        attack_class=None,
        attack_params={"fanout_like_sessions": True, "n_containers": n_containers},
        agent_profile=agent_profile,
        topology_id=topology.topology_id,
        sessions=sessions,
        resources=resources,
        ground_truth=_empty_ground_truth(),
    )
