"""Gateway session-hygiene compression policy."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agent.turn_context import compression_made_progress

logger = logging.getLogger("gateway.run")

_GATEWAY_HYGIENE_PLATFORM = "gateway_hygiene"

_HYGIENE_COOLDOWN_LADDER_MULTIPLIERS = (1, 3, 9)

_HYGIENE_COOLDOWN_MAX_SECONDS = 3600.0

def _hygiene_cooldown_for_failure(
    gateway,
    session_key: str,
    base_cooldown_seconds: float,
) -> float:
    """Bump the hygiene failure streak and return the escalated cooldown.

    This is a MULTIPLIER ladder (x1, x3, x9) over the operator's configured
    ``hygiene_failure_cooldown_seconds``, clamped to
    ``_HYGIENE_COOLDOWN_MAX_SECONDS``, so a tuned base is preserved as rung 1.

    It exists because the in-agent equivalent is unreachable from here:
    ``ContextCompressor.record_timeout_failure`` escalates on an absolute
    60 -> 300 -> 900s ladder driven by the in-memory
    ``_consecutive_timeout_failures`` counter, which ``bind_session_state``
    zeroes.  Session hygiene constructs a FRESH ``create_agent`` per run and re-binds
    state every time, so from the gateway that streak is structurally always 0
    and only the flat ``hygiene_failure_cooldown_seconds`` could ever be
    recorded — a session whose summary model always times out retried on that
    same fixed interval forever (#79624).  The streak is mirrored to SQLite by
    rotation-stable ``session_key`` so it outlives both the per-run agent and
    gateway restarts; ``PersistentState`` keeps the hot in-process view.
    """
    streak = 1
    state = None
    try:
        state = gateway.sessions.state(session_key).persistent
    except Exception as exc:
        logger.debug("hygiene failure streak update failed: %s", exc)
    session_db = getattr(gateway, "_session_db", None)
    session_db = getattr(session_db, "_db", session_db)
    increment = getattr(session_db, "increment_hygiene_failure_streak", None)
    if callable(increment):
        try:
            streak = max(1, int(increment(session_key)))
            if state is not None:
                state.hygiene_failure_streak = streak
        except Exception as exc:
            logger.debug("hygiene failure streak persist failed: %s", exc)
            if state is not None:
                state.hygiene_failure_streak += 1
                streak = state.hygiene_failure_streak
    elif state is not None:
        state.hygiene_failure_streak += 1
        streak = state.hygiene_failure_streak
    multiplier = _HYGIENE_COOLDOWN_LADDER_MULTIPLIERS[
        min(streak, len(_HYGIENE_COOLDOWN_LADDER_MULTIPLIERS)) - 1
    ]
    return min(base_cooldown_seconds * multiplier, _HYGIENE_COOLDOWN_MAX_SECONDS)

def _reset_hygiene_failure_streak(gateway, session_key: str) -> None:
    """Clear the hygiene failure streak after a compression that reduced context.

    Peeks rather than get-or-creates: writing a 0 that is already 0 must not
    materialise a ``_sessions`` entry (those are never evicted).
    """
    try:
        state = gateway.sessions.peek(session_key)
        if state is not None:
            state.persistent.hygiene_failure_streak = 0
    except Exception as exc:
        logger.debug("hygiene failure streak reset failed: %s", exc)
    session_db = getattr(gateway, "_session_db", None)
    session_db = getattr(session_db, "_db", session_db)
    reset = getattr(session_db, "reset_hygiene_failure_streak", None)
    if callable(reset):
        try:
            reset(session_key)
        except Exception as exc:
            logger.debug("hygiene failure streak persistent reset failed: %s", exc)

def hygiene_compaction_recovered(
    *,
    aborted: bool,
    rotated: bool,
    in_place: bool,
    msg_count: int,
    new_count: int,
    approx_tokens: int,
    new_tokens: int,
) -> bool:
    """True when a hygiene run actually recovered the session.

    Extracted from ``_handle_message_with_agent`` so the decision is unit
    testable: it previously lived inline in a ~2000-line async method, and the
    only way to pin it was a source-reading test — which AGENTS.md bans
    outright, naming this file.

    "Recovered" requires all three:

    * the compressor did not abort (no summary produced at all);
    * the transcript was actually rewritten — either rotated into a new session
      or compacted in place.  The degenerate "did not rotate or compact in
      place" path (#21301) reuses the pre-compression counts, so relying on the
      numbers alone would read a no-op as success;
    * the request materially shrank, per the canonical
      :func:`compression_made_progress` (#39548) — a row-count drop counts even
      when the summary keeps the token estimate flat, and a sub-5% token wobble
      does not count at all.

    The token arguments are deliberately compared through that shared predicate
    rather than with a bare ``<``: ``approx_tokens`` can be provider-reported
    while ``new_tokens`` is always a rough estimate (documented to run 30-50%
    high on code-heavy sessions), so a bare comparison both misses real wins and
    counts noise as one.
    """
    if aborted:
        return False
    if not (rotated or in_place):
        return False
    return compression_made_progress(
        msg_count, new_count, approx_tokens, new_tokens
    )

def _record_hygiene_cooldown(
    gateway,
    session_id: str,
    cooldown_seconds: float,
    error: Optional[str] = None,
) -> None:
    """Persist a session-hygiene compression-failure cooldown to the state DB.

    Uses the same ``compression_failure_cooldown_until`` column and
    ``record_compression_failure_cooldown`` method that the in-conversation
    compression path (``agent/context_compressor.py``) already uses, so the
    cooldown survives gateway restarts (#74136).

    ``error`` is forwarded because the recorder writes
    ``compression_failure_error`` UNCONDITIONALLY — omitting it clobbers to NULL
    any reason the in-conversation path recorded, and readers surface that
    reason to the user (falling back to "unknown error"). That matters more now
    that an escalated cooldown can last up to an hour.
    """
    import time as _time
    session_db = getattr(gateway, "_session_db", None)
    if session_db is None:
        return
    session_db = getattr(session_db, "_db", session_db)
    recorder = getattr(session_db, "record_compression_failure_cooldown", None)
    if recorder is None:
        return
    try:
        recorder(session_id, _time.time() + cooldown_seconds, error)
    except Exception as exc:
        logger.debug("session hygiene cooldown persist failed: %s", exc)

def _seed_hygiene_system_prompt(
    agent: Any,
    session_row: Optional[Dict[str, Any]],
) -> bool:
    """Keep gateway hygiene from rebuilding a live session's system prompt.

    The hygiene helper intentionally skips memory-provider initialization.
    Compression is allowed to persist a system prompt, so letting that helper
    rebuild one would strip external provider blocks from the live session.
    Seed the exact persisted prompt instead.  When no usable prompt can be
    restored, seed an empty cache entry.  Compression either preserves that
    unusable value or rebuilds with the hygiene-only platform marker; the real
    turn will rebuild either form with its fully initialized providers.
    """
    stored_prompt = ""
    if isinstance(session_row, dict):
        raw_prompt = session_row.get("system_prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            stored_prompt = raw_prompt

    agent._cached_system_prompt = stored_prompt
    return bool(stored_prompt)
