"""Process-local gateway session-state ownership."""

from __future__ import annotations

import logging
from typing import Any

from gateway.session_state import SessionState
from gateway.turn_lease import SessionTurnLeaseRegistry

logger = logging.getLogger("gateway.run")


class GatewaySessionRuntime:
    """Own the live state map and its current typed access contract."""

    def __init__(self) -> None:
        self.states: dict[str, SessionState] = {}
        self.turn_leases = SessionTurnLeaseRegistry()

    def state(self, session_key: str) -> SessionState:
        state = self.states.get(session_key)
        if state is None:
            state = SessionState()
            self.states[session_key] = state
        return state

    def peek(self, session_key: str) -> SessionState | None:
        return self.states.get(session_key)

    def is_running(self, session_key: str) -> bool:
        state = self.peek(session_key)
        return state is not None and state.turn.agent is not None

    def running_items(self) -> list[tuple[str, object]]:
        return [
            (key, state.turn.agent)
            for key, state in self.states.items()
            if state.turn.agent is not None
        ]

    def running_count(self) -> int:
        return sum(1 for state in self.states.values() if state.turn.agent is not None)

    def enqueue_fifo(self, session_key: str, queued_event: Any, adapter: Any) -> None:
        if adapter is None:
            return
        pending = getattr(adapter, "_pending_messages", None)
        if pending is None:
            return
        if session_key in pending:
            self.state(session_key).conversation.queued_events.append(queued_event)
        else:
            pending[session_key] = queued_event

    def promote_queued_event(
        self, session_key: str, adapter: Any, pending_event: Any | None
    ) -> Any | None:
        state = self.peek(session_key)
        overflow = state.conversation.queued_events if state else None
        if not overflow:
            return pending_event
        next_queued = overflow.pop(0)
        if pending_event is None:
            return next_queued
        pending = getattr(adapter, "_pending_messages", None)
        if pending is None:
            overflow.insert(0, next_queued)
        else:
            pending[session_key] = next_queued
        return pending_event

    def queue_depth(self, session_key: str, adapter: Any = None) -> int:
        state = self.peek(session_key)
        depth = len(state.conversation.queued_events) if state else 0
        if adapter is not None and session_key in getattr(adapter, "_pending_messages", {}):
            depth += 1
        return depth

    def release_turn_lease(self, session_key: str, run_generation: int) -> bool:
        if not session_key:
            return False
        state = self.peek(session_key)
        if state is None:
            return False
        turn = state.turn
        if turn.lease_token is None or turn.lease_generation != run_generation:
            return False
        token = turn.lease_token
        turn.lease_token = None
        turn.lease_generation = None
        try:
            return self.turn_leases.release(token)
        except Exception:
            logger.debug("Failed to release turn lease", exc_info=True)
            return False

    def rebind_turn_lease(
        self, session_key: str, run_generation: int, new_session_id: str
    ) -> bool:
        if not session_key or not new_session_id:
            return False
        state = self.peek(session_key)
        if state is None:
            return False
        turn = state.turn
        if turn.lease_token is None or turn.lease_generation != run_generation:
            return False
        try:
            return self.turn_leases.rebind(turn.lease_token, new_session_id)
        except Exception:
            logger.debug("Failed to rebind turn lease", exc_info=True)
            return False

    def begin_run_generation(self, session_key: str) -> int:
        if not session_key:
            return 0
        persistent = self.state(session_key).persistent
        persistent.run_generation = int(persistent.run_generation) + 1
        return persistent.run_generation

    def invalidate_run_generation(self, session_key: str, reason: str = "") -> int:
        generation = self.begin_run_generation(session_key)
        if reason:
            logger.info(
                "Invalidated run generation for %s → %d (%s)",
                session_key,
                generation,
                reason,
            )
        return generation

    def run_is_current(self, session_key: str, generation: int) -> bool:
        if not session_key:
            return True
        state = self.peek(session_key)
        current = state.persistent.run_generation if state is not None else 0
        return int(current) == int(generation)

    def set_sidecar_notes(self, session_key: str, notes: list[str]) -> None:
        if session_key and notes:
            self.state(session_key).conversation.sidecar_notes = list(notes)

    def consume_sidecar_notes(self, session_key: str) -> list[str]:
        if not session_key:
            return []
        state = self.peek(session_key)
        if state is None:
            return []
        staged = state.conversation.sidecar_notes
        state.conversation.sidecar_notes = []
        return list(staged)
