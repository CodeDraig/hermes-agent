"""
Session mirroring for cross-platform message delivery.

When a message is sent to a platform (via send_message or cron delivery),
this module appends a "delivery-mirror" record to the target session's
transcript so the receiving-side agent has context about what was sent.

Standalone -- works from CLI, cron, and gateway contexts without needing
the full SessionStore machinery.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    role: str = "assistant",
) -> bool:
    """
    Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session that matches the given platform + chat_id,
    then writes a mirror entry to both the JSONL transcript and SQLite DB.

    ``role`` defaults to ``"assistant"`` — correct for the interactive
    ``send_message`` mirror, where the mirrored text is the agent's own
    outgoing reply (a genuine assistant turn). Callers mirroring text that is
    NOT the agent speaking — e.g. a cron brief delivered out-of-band — must
    pass ``role="user"``: the ``mirror``/``mirror_source`` metadata is dropped
    at the SQLite boundary (only role+content persist), so on replay an
    assistant-role mirror is indistinguishable from a real assistant turn and
    produces ``assistant → assistant`` pairs that break strict-alternation
    providers (issue #2221). A user-role mirror collapses safely via
    ``repair_message_sequence``'s consecutive-user merge on every provider.

    Returns True if mirrored successfully, False if no matching session or error.
    All errors are caught -- this is never fatal.
    """
    try:
        session_id = _find_session_id(
            platform,
            str(chat_id),
            thread_id=thread_id,
            user_id=user_id,
        )
        if not session_id:
            logger.debug(
                "Mirror: no session found for %s:%s:%s:%s",
                platform,
                chat_id,
                thread_id,
                user_id,
            )
            return False

        mirror_msg = {
            "role": role,
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        }

        _append_to_sqlite(session_id, mirror_msg)

        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True

    except Exception as e:
        logger.debug(
            "Mirror failed for %s:%s:%s:%s: %s",
            platform,
            chat_id,
            thread_id,
            user_id,
            e,
        )
        return False


def _find_session_id(
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """Find the active gateway session for a persisted message origin."""
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            finder = getattr(db, "find_session_by_origin", None)
            if not callable(finder):
                return None
            session_id = finder(
                platform=platform,
                chat_id=chat_id,
                thread_id=thread_id,
                user_id=user_id,
            )
            return str(session_id) if session_id else None
        finally:
            db.close()
    except Exception as exc:
        logger.debug("Mirror state.db session lookup failed: %s", exc)
        return None



def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database."""
    db = None
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        db.append_message(
            session_id=session_id,
            role=message.get("role", "assistant"),
            content=message.get("content"),
        )
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
    finally:
        if db is not None:
            db.close()
