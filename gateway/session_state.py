"""Per-session gateway state consolidated into one container.

GatewayRunner historically carried ~19 separate ``Dict[str, ...]`` attributes
keyed by session_key, each with its own ad-hoc lifecycle.  Three failure
classes grew out of that shape:

1. Boundary drift — every conversation boundary carried a hand-copied
   pop-list that went stale when a new dict was added (#48031, #58403,
   #10702, #35809).  Mitigated by the ``_CONVERSATION_SCOPED_STATE`` registry,
   now structurally fixed: the fields live in one ``ConversationState``
   dataclass with a single ``clear()``.
2. Turn-release drift — ad-hoc ``del self._running_agents[key]`` sites that
   popped different subsets of the turn dicts.  Mitigated by
   ``_release_running_agent_state``, now ``TurnState.clear()``.
3. Wholesale-reset races — lazy-init paths like
   ``self._session_reasoning_overrides = {}`` replaced the ENTIRE dict,
   discarding concurrent sessions' entries when raced.  Structurally
   impossible now: state is per-session, resets touch one field of one
   ``SessionState``.

Scopes (placement follows where each dict is CLEARED today):

- ``SessionState.turn`` — reset at end of every running turn.
- ``SessionState.conversation`` — reset at conversation boundaries
  (/new, /resume, auto-reset, expiry, compression-exhausted reset).
- ``SessionState.persistent`` — own lifecycles (approval resolution, update
  prompt answer, native-image consumption); ``run_generation`` is monotonic
  and NEVER reset (#28686).

Entries in ``GatewayRunner._sessions`` are never evicted (matching the old
dicts, most of which also leaked empty/stale entries for dead sessions —
see the migration table in the consolidating commit).  Follow-up work may
add eviction of fully-default SessionStates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Presence-sensitive sentinel: /fast stores "priority" or None (explicit
# normal), so key PRESENCE — not value truthiness — decides whether the
# override applies.  ``_UNSET_TIER`` means "no override recorded".
_UNSET_TIER = object()

# Public alias for callers outside this module.
SERVICE_TIER_UNSET = _UNSET_TIER
AGENT_PENDING = object()


@dataclass
class TurnState:
    """State scoped to one running gateway turn.

    Cleared by ``GatewayRunner._release_running_agent_state`` (via
    ``clear()``) at every site that ends a running turn.  ``lease_token`` /
    ``lease_generation`` are deliberately NOT cleared here — they are owned
    by ``_release_turn_lease`` (#64934), which must release the registry
    lease exactly once per acquiring turn.
    """

    # Running create_agent instance (or _AGENT_PENDING_SENTINEL); None = idle.
    agent: Any = None
    # Turn start timestamp (0.0 = not running).
    started_ts: float = 0.0
    # Cross-process active-session slot lease (None = none held).
    lease: Any = None
    # Last busy-ack timestamp (debounce; 0.0 = never acked).
    busy_ack_ts: float = 0.0
    # Held turn-lease token + the run generation that acquired it.  The old
    # ``_turn_lease_tokens`` dict was keyed by (session_key, generation) so a
    # stale unwind could never free a newer turn's lease; the pair encoding
    # preserves that: release/rebind only match when generation is current.
    lease_token: Any = None
    lease_generation: Optional[int] = None

    def clear(self) -> None:
        """Reset the per-turn slot (agent / start ts / lease / busy-ack).

        Mirrors the exact clear set of the old _release_running_agent_state:
        _running_agents, _running_agents_ts, _active_session_leases (popped
        by the caller so it can call ``lease.release()``), _busy_ack_ts.
        """
        self.agent = None
        self.started_ts = 0.0
        self.lease = None
        self.busy_ack_ts = 0.0


@dataclass
class ConversationState:
    """State scoped to one conversation (survives turns, not boundaries)."""

    # /model per-session override (model/provider/api_key/base_url/api_mode).
    model_override: Optional[Dict[str, Any]] = None
    # /model --once restore snapshot.
    one_turn_restore: Optional[Dict[str, Any]] = None
    # /reasoning per-session override.
    reasoning_override: Optional[Dict[str, Any]] = None
    # /fast per-session override: "priority" or None; _UNSET_TIER = absent.
    service_tier_override: Any = _UNSET_TIER
    # Last successfully-resolved non-empty model (#35314 recovery).
    last_resolved_model: str = ""
    # /queue overflow FIFO (adapter slot holds the head).
    queued_events: List[Any] = field(default_factory=list)
    # Per-turn must-deliver sidecar notes (one-shot).
    sidecar_notes: List[str] = field(default_factory=list)
    # Notes prepended to the next user message after runtime changes.
    model_switch_note: Optional[str] = None
    skills_reload_note: Optional[str] = None
    stall_notified: bool = False
    # Pinned session-context bytes: (change_key, text).
    ephemeral_pin: Optional[Tuple[Any, ...]] = None

    def clear(self) -> None:
        """Reset every conversation-scoped field to its default.

        The structural successor of the ``_CONVERSATION_SCOPED_STATE``
        pop-loop: adding a field here means every boundary clears it
        automatically.
        """
        self.model_override = None
        self.one_turn_restore = None
        self.reasoning_override = None
        self.service_tier_override = _UNSET_TIER
        self.last_resolved_model = ""
        self.queued_events = []
        self.sidecar_notes = []
        self.model_switch_note = None
        self.skills_reload_note = None
        self.stall_notified = False
        self.ephemeral_pin = None


@dataclass
class PersistentState:
    """State with its own lifecycle — NOT cleared by turn or boundary resets
    wholesale (approvals/update prompts ARE cleared by the boundary
    *security* funnel, but individually, matching the old behavior)."""

    # Pending exec approval ({"command": ..., "pattern_key": ...}).
    approvals: Optional[Dict[str, Any]] = None
    # /update prompt awaiting a user response.
    update_prompt_pending: bool = False
    # Image paths staged for native (inline) attachment; consumed one-shot.
    native_image_paths: List[str] = field(default_factory=list)
    # Legacy runner-level pending message text (write-mostly; flushed to
    # disk on shutdown — see #72680).  NOTE: distinct from the adapter-level
    # ``_pending_messages`` (Dict[str, MessageEvent]) in gateway/base.py,
    # which is a different store that happens to share the old name.
    pending_command_text: Optional[str] = None
    # Monotonic run-generation counter (#28686).  NEVER reset: clearing it
    # would break stale-run detection.
    run_generation: int = 0
    # Consecutive session-hygiene compression failures for this session
    # (#79624).  The in-agent compressor escalates repeat timeouts via
    # ContextCompressor._consecutive_timeout_failures, but hygiene builds a
    # FRESH create_agent per run and bind_session_state() zeroes that counter, so
    # the in-agent ladder is structurally unreachable from the gateway.
    # Tracking the streak here — outside the per-run agent — lets hygiene
    # escalate its cooldown instead of retrying on a flat interval forever.
    # Reset on a successful compression, not by turn/boundary resets.
    #
    # PROCESS-LOCAL, deliberately: `PersistentState` means "survives turn and
    # boundary resets", NOT "survives a restart" — this field has no disk flush
    # (unlike `pending_command_text` above, #72680), so a gateway restart drops
    # escalation back to rung 1 while the DB-backed deadline itself survives
    # (#74136). Keying on `session_key` rather than `session_id` is what buys
    # correctness across compaction ROTATION (the sid changes, the chat does
    # not). gateway.run mirrors this value to the DB keyed by session_key so
    # the same semantics also survive gateway restarts.
    hygiene_failure_streak: int = 0


@dataclass
class SessionState:
    """All per-session gateway state, grouped by lifecycle scope."""

    turn: TurnState = field(default_factory=TurnState)
    conversation: ConversationState = field(default_factory=ConversationState)
    persistent: PersistentState = field(default_factory=PersistentState)


# ---------------------------------------------------------------------------
