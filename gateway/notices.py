"""Gateway notice rendering and status-message delivery."""

from __future__ import annotations

from typing import Any, Optional


def non_conversational_metadata(metadata=None, *, platform=None):
    """Return delivery metadata unchanged for retained platforms."""
    return metadata

def render_notice_line(notice) -> str:
    """Render an AgentNotice to a single plaintext line for messaging platforms.

    Messaging has no persistent status bar (unlike the TUI), so a notice is a
    one-shot standalone push. The notice policy already bakes the level glyph
    (⚠ / • / ✕ / ✓) into the text, and the TUI + CLI REPL render that text
    verbatim — so we emit it as-is here too. Prepending a per-level glyph would
    DOUBLE it ("⚠ ⚠ Credits 90% used", "⛔ ✕ Credit access paused"). Plaintext
    only — no markdown — so it renders uniformly across retained platforms
    without per-platform escaping. Fail-soft: a malformed/empty notice
    degrades to "" rather than raising on the agent's callback path.
    """
    return str(getattr(notice, "text", "") or "").strip()

async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status message through adapter.send_or_update_status when supported.

    Issue #30045: adapters that implement send_or_update_status (currently
    Telegram) edit the previous bubble for the same status_key instead of
    appending a new one. Adapters without the method fall back to plain send.
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)

def _resolve_progress_thread_id(
    platform: Any,
    source_thread_id: Any,
    event_message_id: Any,
    *,
    reply_in_thread: bool = True,
) -> Optional[str]:
    """Return thread/root ID that progress/status bubbles should target.

    ``reply_in_thread=False`` disables the synthetic-thread fallback: progress messages must not create
    a thread the final flat reply would then inherit. A source.thread_id equal
    to the event's own message id is the adapter's synthetic session-keying
    thread, not a real thread — treat it as "no thread" too (#18859).
    """
    platform_value = getattr(platform, "value", platform)
    platform_key = str(platform_value or "").lower()
    if not reply_in_thread:
        if (
            source_thread_id
            and event_message_id
            and str(source_thread_id) == str(event_message_id)
        ):
            return None
        return str(source_thread_id) if source_thread_id else None
    if source_thread_id:
        return str(source_thread_id)
    if platform_key == "mattermost" and event_message_id:
        return str(event_message_id)
    return None
