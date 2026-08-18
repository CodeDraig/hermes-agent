"""Inactivity monitoring and process cleanup for gateway turns."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from typing import Callable, Optional

import agent.interruption as interruption

logger = logging.getLogger("gateway.run")

_INTERRUPT_REASON_TIMEOUT = "Execution timed out (inactivity)"

def _reap_gateway_turn_processes(
    task_id: str,
    process_baseline,
    *,
    source: str,
    is_still_current: Optional[Callable[[], bool]] = None,
) -> int:
    """Reap only background processes created by one abandoned turn.

    ``task_id`` is session-scoped (task_id == session_id), not turn-scoped,
    so a *replacement* turn on the same session can start and spawn its own
    legitimate process while this reap is still in flight. ``is_still_current``
    — a closure over the run_generation captured when the reaping turn began
    or was interrupted — lets the caller detect that a newer turn has since
    claimed the session and bail out instead of killing that newer turn's
    process. The newer turn snapshots its own baseline independently, so
    skipping here does not leave anything permanently unreaped.
    """
    if not task_id:
        # ProcessSession.task_id defaults to "" for sessionless callers, so a
        # blank id would match (and kill) every unrelated empty-task process
        # instead of this turn's own. Nothing session-scoped to reap.
        return 0
    if is_still_current is not None:
        try:
            if not is_still_current():
                logger.debug(
                    "Skipping reap for turn %s (%s): a newer turn already "
                    "claimed this session; it owns its own baseline.",
                    task_id,
                    source,
                )
                return 0
        except Exception:
            logger.debug(
                "is_still_current check failed for turn %s (%s); reaping anyway",
                task_id,
                source,
                exc_info=True,
            )

    from tools.process_registry import process_registry

    try:
        killed = process_registry.kill_started_since(
            task_id,
            process_baseline,
            source=source,
        )
    except Exception:
        # Runs on a detached daemon thread (interrupt and timeout call
        # sites both fire-and-forget it) — an uncaught exception here
        # would only surface via threading.excepthook, bypassing the
        # app's logger. Swallow and log through the normal channel instead.
        logger.warning(
            "Failed to reap background processes for turn %s (%s)",
            task_id,
            source,
            exc_info=True,
        )
        return 0
    if killed:
        logger.warning(
            "Reaped %d background process(es) created by abandoned turn %s (%s)",
            killed,
            task_id,
            source,
        )
    return killed

_TURN_STACK_DUMP_FRAME_MARKERS = (
    "run_conversation",
    "run_sync",
    "_run_sync_with_timeout_lifecycle",
    "finalize_turn",
    "end_turn",
    "run_in_session",
)

def _dump_wedged_turn_stacks(task_id: str) -> None:
    """Log the stack of every thread that looks like turn work, at reap time.

    When the inactivity reaper fires, the model loop is usually long done and
    the worker thread is wedged somewhere in post-turn finalization — but the
    reaper's hard interrupt frees it, so the blocked frame is gone before
    anyone can attach a profiler. A live incident (Aug 2026, WhatsApp session
    on a Relay-corrupted scope stack) wedged EVERY turn for exactly the
    1800s timeout between "Turn ended" and run_sync returning, and the wedge
    point was unrecoverable post-mortem. Dumping the stacks here, BEFORE the
    interrupt, names the frame.

    Best-effort and bounded: pure in-process frame walking (no signals, no
    external tools), only threads whose stack mentions a turn-machinery
    marker are logged, output capped per thread. Must never raise into the
    reaper.
    """
    try:
        frames = sys._current_frames()
        names = {t.ident: t.name for t in threading.enumerate()}
        dumped = 0
        for ident, frame in frames.items():
            if ident == threading.get_ident():
                continue  # the reaper itself
            stack = traceback.format_stack(frame)
            joined = "".join(stack)
            if not any(marker in joined for marker in _TURN_STACK_DUMP_FRAME_MARKERS):
                continue
            dumped += 1
            if dumped > 8:
                logger.error(
                    "Wedged-turn stack dump for task %s truncated: more than "
                    "8 candidate threads",
                    task_id,
                )
                break
            logger.error(
                "Wedged-turn stack dump (task=%s thread=%s ident=%s):\n%s",
                task_id,
                names.get(ident, "?"),
                ident,
                "".join(stack[-25:]),
            )
        if dumped == 0:
            logger.error(
                "Wedged-turn stack dump for task %s: no thread with "
                "turn-machinery frames found (worker may have already exited)",
                task_id,
            )
    except Exception:
        logger.debug("Wedged-turn stack dump failed", exc_info=True)

def _abandon_timed_out_gateway_turn(
    *,
    agent_holder,
    task_id: str,
    process_baseline,
    worker_done: threading.Event,
    timeout_fired: threading.Event,
    cleanup_lock: threading.Lock,
    is_still_current: Optional[Callable[[], bool]] = None,
) -> bool:
    """Interrupt one timed-out turn and reap only processes it created."""
    with cleanup_lock:
        if worker_done.is_set() or timeout_fired.is_set():
            return False
        timeout_fired.set()

    # Capture the wedged worker's stack BEFORE interrupting it — the
    # interrupt frees the blocked frame, destroying the only evidence of
    # where the turn was stuck (see _dump_wedged_turn_stacks).
    _dump_wedged_turn_stacks(task_id)

    agent = agent_holder[0] if agent_holder else None
    if agent is not None:
        try:
            interruption.hard_interrupt(agent, _INTERRUPT_REASON_TIMEOUT)
        except Exception:
            logger.debug("Timed-out agent interrupt failed", exc_info=True)

    try:
        _reap_gateway_turn_processes(
            task_id,
            process_baseline,
            source="gateway_turn_timeout",
            is_still_current=is_still_current,
        )
    except Exception:
        logger.warning(
            "Failed to reap background processes for timed-out turn %s",
            task_id,
            exc_info=True,
        )
    return True

def _watch_gateway_turn_inactivity(
    *,
    agent_holder,
    task_id: str,
    process_baseline,
    timeout: float,
    worker_done: threading.Event,
    timeout_fired: threading.Event,
    cleanup_lock: threading.Lock,
    poll_interval: float = 5.0,
    is_still_current: Optional[Callable[[], bool]] = None,
) -> None:
    """Thread watchdog that remains runnable when gateway asyncio is starved."""
    import agent.status_output as status_output
    while not worker_done.wait(max(0.01, poll_interval)):
        agent = agent_holder[0] if agent_holder else None
        if agent is None or not hasattr(agent, "get_activity_summary"):
            continue
        try:
            idle_seconds = float(
                status_output.get_activity_summary(agent).get("seconds_since_activity", 0.0)
            )
        except Exception:
            continue
        if idle_seconds < timeout:
            continue
        _abandon_timed_out_gateway_turn(
            agent_holder=agent_holder,
            task_id=task_id,
            process_baseline=process_baseline,
            worker_done=worker_done,
            timeout_fired=timeout_fired,
            cleanup_lock=cleanup_lock,
            is_still_current=is_still_current,
        )
        return
