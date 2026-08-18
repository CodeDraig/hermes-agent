"""Responsibility-owned agent session runtime behavior."""
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from agent.context_compressor import (  # noqa: F401
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
)
from agent.init_session import safe_session_filename_component
from agent.memory_manager import sanitize_context
from agent.message_protocol import is_ephemeral_scaffolding
from agent.redact import redact_sensitive_text
from agent.session_activity import ActivityProvenance
from agent.tool_dispatch_helpers import (
    _is_multimodal_tool_result,
    _multimodal_text_summary,
    _trajectory_normalize_msg,
)
from agent.trajectory import (
    convert_scratchpad_to_think,
    save_trajectory as _save_trajectory_to_file,
)
from agent.turn_context import drop_stale_api_content
from utils import atomic_json_write


logger = logging.getLogger(__name__)

_INFLIGHT_TURNS_BY_SESSION: Dict[str, Tuple[str, float]] = {}
_INFLIGHT_TURNS_LOCK = threading.Lock()

_DB_PERSISTED_MARKER = "_db_persisted"

def _launch_cwd_for_session(source: str) -> Optional[str]:
    """Working directory to stamp on a new session row, or None.

    Only local CLI sessions get a recorded cwd: the directory the process was
    launched from is meaningful for ``hermes -c`` / ``--resume`` (relaunch
    where you left off). Gateway/cron/remote-backend sessions have no stable
    host cwd to restore, so they record nothing.

    ``TERMINAL_ENV`` is set by the CLI's config bridge (``load_cli_config``);
    a non-"local" backend (docker/ssh/modal/...) means the host cwd is
    irrelevant to the agent's tools, so we skip it there too.
    """
    if source != "cli":
        return None
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend and backend != "local":
        return None
    try:
        return os.getcwd()
    except OSError:
        # cwd was unlinked out from under us — nothing meaningful to record.
        return None

def _session_source_for_agent(platform: Optional[str]) -> str:
    try:
        from gateway.session_context import get_session_env

        source = get_session_env("HERMES_SESSION_SOURCE", "")
    except Exception:
        source = os.environ.get("HERMES_SESSION_SOURCE", "")
    source = str(source or "").strip()
    if source:
        return source
    return platform or "cli"

def _get_session_db_for_recall(self):
    """Return a SessionDB for recall, lazily creating it if an entrypoint forgot.

    Most frontends pass ``session_db`` into ``AgentState`` explicitly, but recall
    is important enough that a missing constructor argument should degrade by
    opening the default state DB instead of making the advertised
    ``session_search`` tool unusable.
    """
    # Persistence-isolated forks (background review) must not lazily open the
    # canonical state DB: doing so would re-arm _flush_messages_to_session_db
    # to write the fork's harness turn into the user's real session. Recall
    # degrades to None for them (they don't use session_search anyway).
    if getattr(self, "_persist_disabled", False):
        return None
    if self._session_db is not None:
        return self._session_db
    try:
        from hermes_state import SessionDB

        self._session_db = SessionDB()
        # We opened it here, so nothing else holds a reference — this agent
        # is its only owner and lifecycle.close() must release it.
        self._owns_session_db = True
        return self._session_db
    except Exception:
        logger.debug("SessionDB unavailable for recall", exc_info=True)
        return None

def _ensure_db_session(self) -> None:
    """Create session DB row on first use. Disables _session_db on failure."""
    if getattr(self, "_persist_disabled", False):
        return
    if self._session_db_created or not self._session_db:
        return
    source = _session_source_for_agent(self.platform)
    try:
        try:
            from hermes_cli.profiles import get_active_profile_name
            _profile_for_session = get_active_profile_name()
            if _profile_for_session == "default":
                _profile_for_session = None
        except Exception:
            _profile_for_session = None
        # Carry the live YOLO bypass into the creation-time model_config so
        # a session whose /yolo was toggled BEFORE the row existed (the row
        # is created lazily on the first turn) still persists the flag for
        # `hermes --resume`. set_session_yolo() no-ops on a missing row, so
        # this is the only chance to record a pre-first-turn toggle.
        _init_model_config = self._session_init_model_config

        try:
            from tools.approval import is_session_yolo_enabled
            if is_session_yolo_enabled(self.session_id):
                _init_model_config = dict(_init_model_config or {})
                _init_model_config["yolo_mode"] = True
        except Exception:
            pass
        self._session_db.create_session(
            session_id=self.session_id,
            source=source,
            model=self.model,
            model_config=_init_model_config,
            system_prompt=self._cached_system_prompt,
            user_id=None,
            parent_session_id=self._parent_session_id,
            cwd=_launch_cwd_for_session(source),
            profile_name=_profile_for_session,
        )
        self._session_db_created = True
    except Exception as e:
        # Transient failure (e.g. SQLite lock). Keep _session_db alive —
        # _session_db_created stays False so next lifecycle.run_conversation() retries.
        logger.warning(
            "Session DB creation failed (will retry next turn): %s", e
        )

def _transition_context_engine_session(
    self,
    *,
    old_session_id: Optional[str] = None,
    new_session_id: Optional[str] = None,
    previous_messages: Optional[list] = None,
    carry_over_context: bool = False,
    reset_engine: bool = True,
    **extra_context,
) -> None:
    """Notify the active context engine about a host session transition.

    Generic host-side lifecycle helper. The built-in compressor keeps its
    existing reset behavior; plugin engines that implement richer hooks
    (``on_session_end``, ``on_session_reset``, ``on_session_start``,
    ``carry_over_new_session_context``) can flush old-session state,
    reset runtime counters, bind to the new session, and optionally
    carry retained context forward.
    """
    engine = getattr(self, "context_compressor", None)
    if not engine:
        return

    if old_session_id and previous_messages is not None and hasattr(engine, "on_session_end"):
        try:
            engine.on_session_end(old_session_id, previous_messages)
        except Exception as exc:
            logger.debug("context engine on_session_end during transition: %s", exc)

    if reset_engine and hasattr(engine, "on_session_reset"):
        try:
            engine.on_session_reset()
        except Exception as exc:
            logger.debug("context engine on_session_reset during transition: %s", exc)

    should_start = bool(
        old_session_id
        or previous_messages is not None
        or carry_over_context
        or extra_context
    )
    target_session_id = new_session_id or getattr(self, "session_id", "") or ""
    if should_start and target_session_id and hasattr(engine, "on_session_start"):
        start_context = {
            "old_session_id": old_session_id,
            "carry_over_context": carry_over_context,
            "platform": _session_source_for_agent(getattr(self, "platform", None)),
            "model": getattr(self, "model", ""),
            "context_length": getattr(engine, "context_length", None),
            "conversation_id": getattr(self, "_gateway_session_key", None),
        }
        start_context.update(extra_context)
        start_context = {k: v for k, v in start_context.items() if v not in (None, "")}
        try:
            engine.on_session_start(target_session_id, **start_context)
        except Exception as exc:
            logger.debug("context engine on_session_start during transition: %s", exc)

    if (
        carry_over_context
        and old_session_id
        and target_session_id
        and hasattr(engine, "carry_over_new_session_context")
    ):
        try:
            engine.carry_over_new_session_context(old_session_id, target_session_id)
        except Exception as exc:
            logger.debug("context engine carry_over_new_session_context during transition: %s", exc)

def reset_session_state(
    self,
    previous_messages: Optional[list] = None,
    old_session_id: Optional[str] = None,
    carry_over_context: bool = False,
):
    """Reset all session-scoped token counters to 0 for a fresh session.

    This method encapsulates the reset logic for all session-level metrics
    including:
    - Token usage counters (input, output, total, prompt, completion)
    - Cache read/write tokens
    - API call count
    - Reasoning tokens
    - Estimated cost tracking
    - Context compressor internal counters

    The method safely handles optional attributes (e.g., context compressor)
    using ``hasattr`` checks.

    When ``previous_messages`` / ``old_session_id`` / ``carry_over_context``
    are provided, the active context engine is notified through the
    full transition lifecycle (``_transition_context_engine_session``)
    instead of a bare reset. Default callers pass nothing and keep the
    existing reset-only behavior.
    """
    # Token usage counters
    self.session_total_tokens = 0
    self.session_input_tokens = 0
    self.session_output_tokens = 0
    self.session_prompt_tokens = 0
    self.session_completion_tokens = 0
    self.session_cache_read_tokens = 0
    self.session_cache_write_tokens = 0
    self.session_reasoning_tokens = 0
    self.session_api_calls = 0
    self.session_estimated_cost_usd = 0.0
    self.session_cost_status = "unknown"
    self.session_cost_source = "none"

    # Turn counter (added after reset_session_state was first written — #2635)
    self._user_turn_count = 0

    # Copilot x-initiator: True for the first API call of a user turn,
    # False for tool-loop follow-ups (#3040).
    self._is_user_initiated_turn = False

    # Context engine reset/transition (works for built-in compressor and plugins)
    _transition_context_engine_session(self,
        old_session_id=old_session_id,
        new_session_id=getattr(self, "session_id", None),
        previous_messages=previous_messages,
        carry_over_context=carry_over_context,
        reset_engine=True,
    )

    # Reset-only session switches (/new, /resume, /branch) update
    # agent.session_id before calling reset_session_state(). The built-in
    # compressor keeps durable cooldown state keyed by its bound session,
    # so rebind it when the active session changed but no full start hook ran.
    engine = getattr(self, "context_compressor", None)
    target_session_id = getattr(self, "session_id", "") or ""
    bound_session_id = getattr(engine, "_session_id", "") if engine is not None else ""
    if (
        engine is not None
        and hasattr(engine, "bind_session_state")
        and target_session_id
        and target_session_id != bound_session_id
    ):
        try:
            engine.bind_session_state(getattr(self, "_session_db", None), target_session_id)
        except Exception as exc:
            logger.debug("context engine bind_session_state during reset: %s", exc)

def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
    """Save session state to both JSON log and SQLite on any exit path.

    Ensures conversations are never lost, even on errors or early returns.

    Trailing empty-response scaffolding is dropped from the live list in
    place (it is ephemeral junk the real transcript should shed). The
    persist user-message *override* is NOT applied here — it is resolved
    inside ``_flush_messages_to_session_db`` and written only to the DB row,
    never mutating the live message list used by the API call (#48677 is
    thus closed for every persist caller, not just this one).
    """
    # Scaffolding removal mutates the live list (desired — ephemeral
    # retry/failure sentinels must not survive into the real transcript).
    # Close and turn-start persistence can run on separate CLI threads; the
    # marker test-and-append below must be one critical section or both can
    # observe the same unmarked dict and write duplicate durable rows.
    persist_lock = getattr(self, "_session_persist_lock", None)

    def _persist_and_drain() -> None:
        _drop_trailing_empty_response_scaffolding(self, messages)
        self._session_messages = messages
        _save_session_log(self, messages)
        _flush_messages_to_session_db(self, messages, conversation_history)
        # Drain async token-accounting deltas at every persist point (turn
        # finalize + error exits) so a crash after this line loses at most
        # the in-flight API call's delta. Cheap no-op when nothing queued.
        if self._session_db is not None:
            self._session_db.flush_token_counts()
        note_turn_persisted(self)

    if persist_lock is None:
        _persist_and_drain()
        return

    with persist_lock:
        _persist_and_drain()

def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
    """Remove private empty-response retry/failure scaffolding from transcript tails.

    Also rewinds past any trailing tool-result / assistant(tool_calls) pair
    that the failed iteration left hanging. Without this, the tail ends at
    a raw ``tool`` message and the next user turn lands as
    ``...tool, user, user`` — a protocol-invalid sequence that most
    providers silently reject (returns empty content), causing the
    empty-retry loop to fire forever. (issue number to be backfilled once filed)
    """
    # Pass 1: strip the flagged scaffolding messages themselves.
    dropped_scaffolding = False
    while (
        messages
        and isinstance(messages[-1], dict)
        and (
            messages[-1].get("_empty_recovery_synthetic")
            or messages[-1].get("_empty_terminal_sentinel")

        )
    ):
        messages.pop()
        dropped_scaffolding = True

    # Pass 2: if we stripped scaffolding, rewind through any trailing
    # tool-result messages plus the assistant(tool_calls) message that
    # produced them. This preserves role alternation so the next user
    # message follows a user or assistant message, not an orphan tool
    # result. Only runs when scaffolding was actually present — normal
    # conversation tails (real tool loops mid-progress) are untouched.
    if not dropped_scaffolding:
        return

    # Drop any trailing tool-result messages
    while (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "tool"
    ):
        messages.pop()

    # Drop the assistant message that issued the tool calls, if the tail
    # now ends in an assistant-with-tool_calls (the pair that owned the
    # just-popped tool results). Without this, the tail is
    # ``assistant(tool_calls=...)`` with no tool answers, which some
    # providers also reject.
    if (
        messages
        and isinstance(messages[-1], dict)
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("tool_calls")
    ):
        messages.pop()


def _flush_messages_to_session_db(
    self,
    messages: List[Dict],
    conversation_history: Optional[List[Dict]] = None,
):
    """Serialize direct and turn-boundary session flushes per agent."""
    persist_lock = getattr(self, "_session_persist_lock", None)
    if persist_lock is None:
        return _flush_messages_to_session_db_unlocked(self, messages, conversation_history)
    with persist_lock:
        return _flush_messages_to_session_db_unlocked(self, messages, conversation_history)

def _flush_messages_to_session_db_unlocked(
    self,
    messages: List[Dict],
    conversation_history: Optional[List[Dict]] = None,
    _adoption_budget: int = 1,
):
    """Persist any un-flushed messages to the SQLite session store.

    Deduplicates via an intrinsic ``_DB_PERSISTED_MARKER`` stamped on each
    written message dict, so repeated calls (from multiple exit paths) only
    write truly new messages — preventing the duplicate-write bug (#860)
    without relying on positional slices that can drift after
    message-sequence repair, and without a retained ``id(msg)`` set that
    CPython could alias onto a freed-then-reused address (#50372). The
    ``_flushed_db_message_ids`` attribute is now only a one-shot seed
    (translated to markers, then cleared each flush), not a persisted set.

    Note: the marker is stamped on the live/shared conversation dict, which
    correctly makes re-persistence idempotent across turns. No code path
    edits a persisted message's content/role in place expecting a re-write
    (in-place compaction resets the seed and re-diffs by identity).
    """
    # Persistence-isolated agents (e.g. the background skill/memory review
    # fork) must NEVER write into the canonical session store. The fork
    # shares the parent's session_id for prompt-cache warmth, so any write
    # here would land its harness turn ("Review the conversation above and
    # update the skill library…") inside the user's real session history,
    # where the next live turn re-reads it as an instruction and the agent
    # "becomes" the curator. Hard-stop before any DB touch.
    if getattr(self, "_persist_disabled", False):
        return None
    if not self._session_db:
        return None
    # Persist user-message override (#48677 chokepoint): historically this
    # mutated the live `messages` list in place, which — on the early
    # crash-resilience persist that runs BEFORE the API call is built —
    # stripped observed group-chat context off the live user message and
    # silently dropped it. Instead, resolve the override here and apply it
    # ONLY to the value written to the DB (see the write loop below); the
    # live dict is never mutated, so every caller (early persist, mid-loop
    # flush, /resume, /branch) is protected uniformly. Timestamp override is
    # metadata and is likewise applied only to the written row.
    _ov_idx = getattr(self, "_persist_user_message_idx", None)
    _ov_content = getattr(self, "_persist_user_message_override", None)
    _ov_timestamp = getattr(self, "_persist_user_message_timestamp", None)
    try:
        # Retry row creation if the earlier attempt failed transiently.
        if not self._session_db_created:
            _ensure_db_session(self)
        # Positional flushing used to slice at
        # max(len(conversation_history), _last_flushed_db_idx). That
        # assumes the live `messages` list is the original history plus a
        # new tail. repair_message_sequence can shrink/merge the history
        # copy before the final flush, making len(conversation_history)
        # larger than len(messages); the slice is then empty and delivered
        # assistant responses never reach state.db (#46053).
        #
        # Track persistence with an intrinsic per-message marker rather than
        # id(msg). `messages` is a shallow copy of `conversation_history`, so
        # history dicts are skipped by identity, and new dicts appended
        # during this turn are written once even if repair compacts the list
        # around them. Unlike an id()-keyed set, a marker bound to the dict
        # cannot be aliased onto a freed-then-reused address, so a real turn
        # can never be silently skipped (see _DB_PERSISTED_MARKER).
        #
        # `self._flushed_db_message_ids` is still honoured as a *one-shot*
        # seed: external callers (gateway shutdown, tests) populate it with
        # {id(m) for m in already_persisted} immediately before the flush,
        # while those objects are alive — so the ids are valid at that
        # instant. We translate the seed into durable markers and then clear
        # the set, so stale ids can never accumulate across turns and alias a
        # future message.
        current_session_id = getattr(self, "session_id", None)
        flushed_session_id = getattr(self, "_flushed_db_message_session_id", None)
        if flushed_session_id != current_session_id or self._last_flushed_db_idx == 0:
            seed_ids = set()
        else:
            seed_ids = getattr(self, "_flushed_db_message_ids", None)
            if not isinstance(seed_ids, set):
                seed_ids = set()
        self._flushed_db_message_session_id = current_session_id
        history_ids = {
            id(item) for item in (conversation_history or [])
            if isinstance(item, dict)
        }

        # Bounded scan: skip the longest identity-matched prefix of the
        # list snapshot taken at the end of the previous successful flush.
        # Every message in that snapshot was already given its final
        # disposition (written+stamped, stamped as durable history, or
        # skipped as ephemeral scaffolding / non-dict), and no code path
        # pops _DB_PERSISTED_MARKER from a live dict in place (compression
        # strips markers on fresh copies, which breaks identity here and
        # forces a full re-scan). Identity match ⇒ identical skip decision,
        # so starting after the matched prefix is behavior-preserving.
        _scan_start = 0
        _prev_prefix = getattr(self, "_db_flush_scan_prefix", None)
        if isinstance(_prev_prefix, list):
            _limit = min(len(_prev_prefix), len(messages))
            while (
                _scan_start < _limit
                and messages[_scan_start] is _prev_prefix[_scan_start]
            ):
                _scan_start += 1

        # Collect this flush's new rows and write them in ONE transaction
        # at the end of the scan (see append_messages_batch).
        _batch_rows: List[Dict[str, Any]] = []
        _batch_msgs: List[Dict] = []
        for _msg_idx in range(_scan_start, len(messages)):
            msg = messages[_msg_idx]
            if not isinstance(msg, dict):
                continue
            # Never write ephemeral recovery scaffolding to the session
            # store. The flush is append-only (it only advances
            # _last_flushed_db_idx via identity tracking), so a synthetic
            # message committed by a mid-turn persist cannot be un-written
            # when the end-of-turn drop removes it from the in-memory list —
            # the resumed transcript would then replay synthetic
            # "(empty)"/nudge/thinking-prefill turns as if they were genuine
            # context. Skip regardless of position: an answered nudge leaves
            # the synthetic pair buried mid-list, not just at the tail.
            if is_ephemeral_scaffolding(msg):
                continue
            if msg.get(_DB_PERSISTED_MARKER):
                continue
            # Already-durable messages: either carried over from the loaded
            # history copy, or seeded by a caller. Stamp them so future
            # flushes skip them without consulting any id() set again.
            if id(msg) in history_ids or id(msg) in seed_ids:
                msg[_DB_PERSISTED_MARKER] = True
                continue
            role = msg.get("role", "unknown")
            content = msg.get("content")
            # api_content sidecar: the exact bytes sent to the API when
            # they differ from the clean content (stamped by the turn
            # prologue for prefetch/plugin injections). Written verbatim
            # so replay can reproduce the sent prefix byte-for-byte.
            _row_api_content = msg.get("api_content")
            if not isinstance(_row_api_content, str):
                _row_api_content = None
            _row_timestamp = msg.get("timestamp")
            # Apply the persist override to THIS row's written values only
            # (never to the live dict). A multimodal override is a complete
            # clean replacement for an API-local noted payload. Preserve the
            # historical text-only guard for a list payload, though: a plain
            # text override must not erase its image/audio transcript summary.
            # The close safety-net may flush a shortened snapshot while
            # turn setup still owns its staged CLI dict. In that shape the
            # normal turn index refers to the full history, not this list;
            # preserve the API-local override by recognizing the same dict.
            pending_cli_message = getattr(self, "_pending_cli_user_message", None)
            is_current_turn_user = (
                _ov_idx == _msg_idx or msg is pending_cli_message
            )
            if is_current_turn_user and msg.get("role") == "user":
                # Preflight compaction can re-anchor the override index at
                # a message whose content was MERGED with the compaction
                # summary (merge-summary-into-tail). Overwriting that with
                # the clean gateway text would silently drop the summary
                # from the durable transcript. The wire is already
                # consistent — the merge popped the sidecar and the merged
                # content is what gets sent — so keep it.
                if (
                    _ov_content is not None
                    and (not isinstance(content, list) or isinstance(_ov_content, list))
                    and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                ):
                    # The live content is what the API call sends; the
                    # override is the cleaned transcript value. If they
                    # differ and no injection already stamped the sidecar,
                    # keep the sent bytes in api_content so replay matches
                    # the wire (#48677 divergence, closed for the cache
                    # prefix too).
                    if (
                        _row_api_content is None
                        and isinstance(content, str)
                        and content != _ov_content
                    ):
                        _row_api_content = content
                    content = _ov_content
                if _ov_timestamp is not None:
                    _row_timestamp = _ov_timestamp
            # Store the sidecar only when it actually differs.
            if _row_api_content == content:
                _row_api_content = None
            # Load-time sanitize divergence: get_messages_as_conversation
            # replays user/assistant rows through
            # ``sanitize_context(content).strip()``, so content that
            # sanitize would rewrite (echoed/pasted <memory-context>
            # fences or system notes) replays different bytes after a
            # session reload even though THIS turn sent it verbatim.
            # Capture the sent bytes in the sidecar so a reloaded session
            # replays what was actually on the wire. Compared in wire form
            # (both sides .strip()-ed — the api_messages build strips
            # every outgoing content string) so plain surrounding
            # whitespace doesn't grow redundant sidecars.
            if (
                _row_api_content is None
                and role in ("user", "assistant")
                and isinstance(content, str)
                and content
                and sanitize_context(content).strip() != content.strip()
            ):
                _row_api_content = content
            # Persist multimodal tool results as their text summary only —
            # base64 images would bloat the session DB and aren't useful
            # for cross-session replay.
            if _is_multimodal_tool_result(content):
                content = _multimodal_text_summary(content)
            elif isinstance(content, list):
                # List of OpenAI-style content parts: strip images, keep text.
                _txt = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        _txt.append(str(p.get("text", "")))
                    elif isinstance(p, dict) and p.get("type") in {"image", "image_url", "input_image"}:
                        _txt.append("[screenshot]")
                content = "\n".join(_txt) if _txt else None
            tool_calls_data = None
            if hasattr(msg, "tool_calls") and isinstance(msg.tool_calls, list) and msg.tool_calls:
                tool_calls_data = [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in msg.tool_calls
                ]
            elif isinstance(msg.get("tool_calls"), list):
                tool_calls_data = msg["tool_calls"]
            _batch_rows.append({
                "role": role,
                "content": content,
                "tool_name": msg.get("tool_name"),
                "tool_calls": tool_calls_data,
                "tool_call_id": msg.get("tool_call_id"),
                "finish_reason": msg.get("finish_reason"),
                # Reasoning/codex fields are role-gated (assistant-only)
                # inside _insert_message_rows — pass through untouched.
                "reasoning": msg.get("reasoning"),
                "reasoning_content": msg.get("reasoning_content"),
                "reasoning_details": msg.get("reasoning_details"),
                "codex_reasoning_items": msg.get("codex_reasoning_items"),
                "codex_message_items": msg.get("codex_message_items"),
                "timestamp": _row_timestamp,
                "api_content": _row_api_content,
                # Standalone reference handoffs are always hidden, even
                # when the summarized transcript contained a user turn —
                # otherwise they occupy the active user slot in
                # retry/undo/session dispatch (#80622). Merge-into-tail
                # carriers keep prior visibility rules so preserved tail
                # content stays readable.
                "display_kind": (

                    "hidden"
                    if (
                        msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
                        and (
                            ContextCompressor.classify_summary_content(
                                msg.get("content")
                            )
                            == "standalone"
                            or not msg.get(
                                "_compressed_summary_has_user_turn"
                            )
                        )
                    )
                    else msg.get("display_kind")
                ),
                "display_metadata": msg.get("display_metadata"),
            })
            _batch_msgs.append(msg)
        # One transaction for the whole turn's new rows (typically 3-8
        # messages): one BEGIN IMMEDIATE / commit — and, off WAL, one
        # fsync — instead of one per row. All-or-nothing pairs exactly
        # with the marker stamping below: on failure NO rows landed and
        # NO markers were stamped, so the next flush re-scans and
        # re-writes the whole tail (same recovery contract as before,
        # minus the partial-prefix case that could double-pay counters).
        if _batch_rows:
            self._session_db.append_messages_batch(
                session_id=self.session_id,
                messages=_batch_rows,
                compression_lock_holder=getattr(
                    self, "_active_compression_lock_holder", None
                ),
                turn_lease_holder=getattr(
                    self, "_active_session_turn_lease_holder", None
                ),
                turn_lease_ttl_seconds=getattr(
                    self, "_active_session_turn_lease_ttl_seconds", 300.0
                )
                or 300.0,
            )
            for _written in _batch_msgs:
                _written[_DB_PERSISTED_MARKER] = True
        # The intrinsic markers are now the sole source of truth. Reset the
        # one-shot seed so no id() outlives this flush to alias a message
        # allocated next turn at a recycled address.
        self._flushed_db_message_ids = set()
        self._last_flushed_db_idx = len(messages)

        # Snapshot for the bounded scan above — only on full success, so
        # a partially-processed list can never be treated as settled.
        self._db_flush_scan_prefix = messages[:]
        return True
    except Exception as e:
        # Force a full re-scan on the next flush: an exception mid-loop
        # leaves messages with mixed dispositions.
        self._db_flush_scan_prefix = None
        # This is the one place the underlying SQLite error is visible
        # before it is swallowed into a bare ``False`` — classify it here
        # so the turn-end explanation can distinguish lock contention
        # ("storage was busy, send it again") from disk-full/read-only.
        from hermes_state import (
            CompressionSessionClosedError,
            classify_persistence_error,
        )

        self._last_persistence_error_cause = classify_persistence_error(e)
        if isinstance(e, CompressionSessionClosedError):
            # Compression race: another path rotated this session while
            # this turn was still writing against it. The store resolves
            # the continuation chain transitively via the canonical API
            # ``get_compression_tip`` (bounded walk, excludes branch/
            # delegate/tool children, prefers live children over stale
            # closed siblings such as ``ws_orphan_reap``). Adopt the tip
            # ONLY when it is a different row AND still live, and retry
            # the flush exactly once (adoption budget) — a second
            # closed-parent write must fail closed, never loop. The tip
            # walk returns the input id when no continuation exists, so
            # ``tip == session_id`` means fail closed.
            if _adoption_budget > 0:
                old_id = self.session_id
                tip = None
                try:
                    tip = self._session_db.get_compression_tip(old_id)
                except Exception as tip_exc:
                    logger.warning(
                        "compression tip lookup failed for %s: %s",
                        old_id,
                        tip_exc,
                    )
                if tip and tip != old_id:
                    tip_row = None
                    try:
                        tip_row = self._session_db.get_session(tip)
                    except Exception:
                        tip_row = None
                    if tip_row is not None and tip_row.get("ended_at") is None:
                        logger.warning(
                            "Adopted live compression tip %s for closed "
                            "session %s; retrying flush once",
                            tip,
                            old_id,
                        )
                        self.session_id = tip
                        self._flushed_db_message_ids = set()
                        self._last_flushed_db_idx = 0
                        self._compression_adoption_failed = False
                        return _flush_messages_to_session_db_unlocked(self,
                            messages,
                            conversation_history,
                            _adoption_budget=0,
                        )
            # No live tip (or budget exhausted): fail closed — never guess
            # a target session. The per-turn diagnostic flag lets the
            # turn-completion explanation name compression rotation
            # instead of the historical (misleading) full-disk advice.
            self._compression_adoption_failed = True
            logger.warning("Session DB append_message failed: %s", e)
            return False
        logger.warning("Session DB append_message failed: %s", e)
        return False

def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
    """
    Get messages up to (but not including) the last assistant turn.

    This is used when we need to "roll back" to the last successful point
    in the conversation, typically when the final assistant message is
    incomplete or malformed.

    Args:
        messages: Full message list

    Returns:
        Messages up to the last complete assistant turn (ending with user/tool message)
    """
    if not messages:
        return []

    # Find the index of the last assistant message
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        # No assistant message found, return all messages
        return messages.copy()

    # Return everything up to (not including) the last assistant message
    return messages[:last_assistant_idx]

def _format_tools_for_system_message(self) -> str:
    """Forwarder — see ``agent.system_prompt.format_tools_for_system_message``."""
    from agent.system_prompt import format_tools_for_system_message
    return format_tools_for_system_message(self)


def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
    """
    Save conversation trajectory to JSONL file.

    Args:
        messages (List[Dict]): Complete message history
        user_query (str): Original user query
        completed (bool): Whether the conversation completed successfully
    """
    if not self.save_trajectories:
        return

    trajectory = _convert_to_trajectory_format(self, messages, user_query, completed)
    _save_trajectory_to_file(trajectory, self.model, completed)

def _save_session_log(self, messages: List[Dict[str, Any]] = None):
    """Optional per-session JSON snapshot writer.

    Gated by ``sessions.write_json_snapshots`` (default False).  state.db
    is the canonical message store; this writer exists only for users
    whose external tooling consumes ``~/.hermes/sessions/session_{sid}.json``
    directly.  When the flag is off this is a fast no-op.

    When enabled, rewrites the snapshot after every persistence point with
    the full message list (assistant content normalized via
    ``_clean_session_content`` to convert REASONING_SCRATCHPAD to think
    tags).  The truncation guard ("don't overwrite a larger log with

    fewer messages") is preserved so resume + branch don't clobber a
    fuller existing snapshot.
    """
    import agent.error_reporting as error_reporting
    if not getattr(self, "_session_json_enabled", False):
        return
    messages = messages or self._session_messages
    if not messages:
        return

    # Re-derive the target path each call so /branch and /compress
    # session-id changes land in the right file without any re-point
    # bookkeeping at the call sites.  Sanitize the session ID into a
    # single traversal-free path segment — session IDs can come from
    # untrusted input (X-Hermes-Session-Id header) and must not escape
    # the sessions directory.
    try:
        safe_sid = safe_session_filename_component(self.session_id)
        log_file = self.logs_dir / f"session_{safe_sid}.json"
    except Exception:
        return

    try:
        cleaned = []
        for msg in messages:
            # Mirror the SQLite flush: ephemeral recovery scaffolding is
            # internal retry state, never durable transcript content.
            if is_ephemeral_scaffolding(msg):
                continue
            if msg.get("role") == "assistant" and msg.get("content"):
                msg = dict(msg)
                msg["content"] = error_reporting._clean_session_content(msg["content"])
            # Defence-in-depth: redact credentials from every message
            # content before persistence. Catches PATs / API keys / Bearer
            # tokens that may have leaked into assistant responses, tool
            # output, or user paste. Respects HERMES_REDACT_SECRETS via
            # redact_sensitive_text — no-op when disabled. (#19798, #19845)
            if "content" in msg:
                msg = dict(msg)
                msg["content"] = error_reporting._redact_message_content(msg.get("content"))
            cleaned.append(msg)

        # Guard: never overwrite a larger session log with fewer messages.
        # Protects against data loss when a resumed agent starts with
        # partial history and would otherwise clobber the full JSON log.
        if log_file.exists():
            try:
                existing = json.loads(log_file.read_text(encoding="utf-8"))
                existing_count = existing.get("message_count", len(existing.get("messages", [])))
                if existing_count > len(cleaned):
                    logging.debug(
                        "Skipping session log overwrite: existing has %d messages, current has %d",
                        existing_count, len(cleaned),
                    )
                    return
            except Exception:
                pass  # corrupted existing file — allow the overwrite

        entry = {
            "session_id": self.session_id,
            "model": self.model,
            "base_url": self.base_url,
            "platform": self.platform,
            "session_start": self.session_start.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "system_prompt": redact_sensitive_text(self._cached_system_prompt or ""),
            "tools": self.tools or [],
            "message_count": len(cleaned),
            "messages": cleaned,
        }

        atomic_json_write(
            log_file,
            entry,
            indent=2,
            default=str,
        )

    except Exception as e:
        if self.verbose_logging:
            logging.warning(f"Failed to save session log: {e}")

def _touch_activity(
    self,
    desc: str,
    *,
    provenance: Optional[ActivityProvenance] = None,
    force_persist: bool = False,
) -> None:
    """Update the last-activity timestamp and description (thread-safe).

    Also bridges to the kanban board's heartbeat fields when this
    process is a dispatcher-spawned worker (HERMES_KANBAN_TASK set),
    so the dispatcher watchdog doesn't reclaim an actively-running
    worker as stale (#31752). Bridge is rate-limited (60s) and
    best-effort — it never raises into the agent loop.

    Separately, rate-limits a durable SessionDB activity projection
    (``last_activity_at`` + bounded description/provenance) so
    CLI/Gateway consumers share one observation source (#72016 / #72039).

    ``provenance`` defaults to ``unknown`` (the ordinary agent activity
    clock). Named values are for special writers (e.g. compression);
    ordinary call sites should leave the default.

    ``force_persist`` bypasses the 60s SessionDB rate limit so a
    terminal stamp (e.g. compression completed) is not dropped.
    """
    from agent.session_activity import (
        bound_activity_description,
        normalize_activity_provenance,
        reset_session_activity_persist_window,
    )

    self._last_activity_ts = time.time()
    self._last_activity_desc = bound_activity_description(desc)
    self._last_activity_provenance = normalize_activity_provenance(provenance)
    if os.environ.get("HERMES_KANBAN_TASK"):
        try:
            from tools.kanban_tools import (
                heartbeat_current_worker_from_env,
                inject_new_comments_from_env,
            )
            heartbeat_current_worker_from_env()
            # Fold any new operator notes into the running turn (OUT-OF-BAND
            # steer) so the user can talk to a live task without a restart.
            inject_new_comments_from_env(self)
        except Exception:
            # Never let the bridge break the agent loop.  The function
            # already swallows exceptions internally; this outer guard
            # covers import-time failures (kanban_tools unavailable,
            # etc.) on niche deployment surfaces.
            pass
    if force_persist:
        reset_session_activity_persist_window(self)
    _persist_session_activity_if_due(self)

def _persist_session_activity_if_due(self) -> None:
    """Best-effort durable activity heartbeat for SessionDB consumers.

    Cadence is pinned by SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS
    (>=30s per session, config-independent — see agent/session_activity.py).
    The write rides the standard SessionDB ``_execute_write`` patience
    path via ``touch_session_activity``. Fail-open: a failed heartbeat
    write must NEVER raise into the agent loop (swallow + debug-log).
    """
    session_id = getattr(self, "session_id", None)
    session_db = getattr(self, "_session_db", None)
    if not session_id or session_db is None:
        return
    touch = getattr(session_db, "touch_session_activity", None)
    if not callable(touch):
        return
    from agent.session_activity import (
        SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS,
        normalize_activity_provenance,
    )

    now_mono = time.monotonic()
    last_mono = getattr(self, "_session_activity_last_persist_mono", 0.0)
    if (now_mono - last_mono) < SESSION_ACTIVITY_HEARTBEAT_MIN_INTERVAL_SECONDS:
        return
    self._session_activity_last_persist_mono = now_mono
    try:
        touch(
            session_id,
            getattr(self, "_last_activity_ts", None),
            description=getattr(self, "_last_activity_desc", None),
            provenance=normalize_activity_provenance(
                getattr(self, "_last_activity_provenance", None)
            ),
        )
    except Exception:
        # Never let durable heartbeat I/O break the agent loop. The
        # heartbeat is an observation-only projection; the next due
        # window retries naturally.
        logger.debug(
            "session activity heartbeat write failed (ignored)",
            exc_info=True,
        )

def _reset_activity_labels_after_turn(self) -> None:
    """Drop mid-turn activity labels once the turn is no longer running.

    Keeps ``_last_activity_ts`` so idle/watchdog clocks stay continuous
    across interrupt-recursive turns (#15654) and between turns. Clears
    description + provenance so idle cached agents / SessionDB listings
    do not keep advertising the last mid-turn stamp (e.g. compression
    or tool execution) after the turn ended (#72039).
    """
    from agent.session_activity import ActivityProvenance

    self._last_activity_desc = ""
    self._last_activity_provenance = ActivityProvenance.UNKNOWN
    session_id = getattr(self, "session_id", None)
    session_db = getattr(self, "_session_db", None)
    if not session_id or session_db is None:
        return
    clear = getattr(session_db, "clear_session_activity_labels", None)
    if not callable(clear):
        return
    try:
        clear(session_id)
    except Exception:
        # Never let durable cleanup I/O break turn teardown.
        pass

def convert_to_trajectory_format(agent, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
    """
    Convert internal message format to trajectory format for saving.

    Args:
        messages (List[Dict]): Internal message history
        user_query (str): Original user query
        completed (bool): Whether the conversation completed successfully

    Returns:
        List[Dict]: Messages in trajectory format
    """
    import agent.session_runtime as session_runtime
    # Normalize multimodal tool results — trajectories are text-only, so
    # replace image-bearing tool messages with their text_summary to avoid
    # embedding ~1MB base64 blobs into every saved trajectory.
    messages = [_trajectory_normalize_msg(m) for m in messages]
    trajectory = []

    # Add system message with tool definitions
    system_msg = (
        "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
        "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
        "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
        "into functions. After calling & executing the functions, you will be provided with function results within "
        "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
        f"<tools>\n{session_runtime._format_tools_for_system_message(agent)}\n</tools>\n"
        "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
        "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
        "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
        "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
        "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
    )

    trajectory.append({
        "from": "system",
        "value": system_msg
    })

    # Add the actual user prompt (from the dataset) as the first human message
    trajectory.append({
        "from": "human",
        "value": user_query
    })

    # Skip the first message (the user query) since we already added it above.
    # Prefill messages are injected at API-call time only (not in the messages
    # list), so no offset adjustment is needed here.
    i = 1

    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "assistant":
            # Check if this message has tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                # Format assistant message with tool calls
                # Add <think> tags around reasoning for trajectory storage
                content = ""

                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"

                if msg.get("content") and msg["content"].strip():
                    # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                    # (used when native thinking is disabled and model reasons via XML)
                    content += convert_scratchpad_to_think(msg["content"]) + "\n"

                # Add tool calls wrapped in XML tags
                for tool_call in msg["tool_calls"]:
                    if not tool_call or not isinstance(tool_call, dict): continue
                    # Parse arguments - should always succeed since we validate during conversation
                    # but keep try-except as safety net
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                    except json.JSONDecodeError:
                        # This shouldn't happen since we validate and retry during conversation,
                        # but if it does, log warning and use empty dict
                        logger.warning("Unexpected invalid JSON in trajectory conversion: %s", tool_call['function']['arguments'][:100])
                        arguments = {}

                    tool_call_json = {
                        "name": tool_call["function"]["name"],
                        "arguments": arguments
                    }
                    content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"

                # Ensure every gpt turn has a <think> block (empty if no reasoning)
                # so the format is consistent for training data
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content

                trajectory.append({
                    "from": "gpt",
                    "value": content.rstrip()
                })

                # Collect all subsequent tool responses
                tool_responses = []
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    tool_msg = messages[j]
                    # Format tool response with XML tags
                    tool_response = "<tool_response>\n"

                    # Try to parse tool content as JSON if it looks like JSON
                    tool_content = tool_msg["content"]
                    try:
                        if tool_content.strip().startswith(("{", "[")):
                            tool_content = json.loads(tool_content)
                    except (json.JSONDecodeError, AttributeError):
                        pass  # Keep as string if not valid JSON

                    tool_index = len(tool_responses)
                    tool_name = (
                        msg["tool_calls"][tool_index]["function"]["name"]
                        if tool_index < len(msg["tool_calls"])
                        else "unknown"
                    )
                    tool_response += json.dumps({
                        "tool_call_id": tool_msg.get("tool_call_id", ""),
                        "name": tool_name,
                        "content": tool_content
                    }, ensure_ascii=False)
                    tool_response += "\n</tool_response>"
                    tool_responses.append(tool_response)
                    j += 1

                # Add all tool responses as a single message
                if tool_responses:
                    trajectory.append({
                        "from": "tool",
                        "value": "\n".join(tool_responses)
                    })
                    i = j - 1  # Skip the tool messages we just processed

            else:
                # Regular assistant message without tool calls
                # Add <think> tags around reasoning for trajectory storage
                content = ""

                # Prepend reasoning in <think> tags if available (native thinking tokens)
                if msg.get("reasoning") and msg["reasoning"].strip():
                    content = f"<think>\n{msg['reasoning']}\n</think>\n"

                # Convert any <REASONING_SCRATCHPAD> tags to <think> tags
                # (used when native thinking is disabled and model reasons via XML)
                raw_content = msg["content"] or ""
                content += convert_scratchpad_to_think(raw_content)

                # Ensure every gpt turn has a <think> block (empty if no reasoning)
                if "<think>" not in content:
                    content = "<think>\n</think>\n" + content

                trajectory.append({
                    "from": "gpt",
                    "value": content.strip()
                })

        elif msg["role"] == "user":
            trajectory.append({
                "from": "human",
                "value": msg["content"]
            })

        i += 1

    return trajectory

def note_turn_start(agent, turn_id: str):
    """Tripwire: detect a turn starting while a previous turn of the same
    agent — or of the same underlying *session* on a different agent object —
    has not completed its turn-end persist.

    Two turns interleaving on one session corrupt the durable transcript:
    their flushes race (user rows can persist out of arrival order), a row
    can be swallowed by the identity-marker dedup over shared history dicts,
    and the second turn runs on a history base that never saw the first
    turn's exchange. This helper does NOT prevent any of that — it names the
    occurrence, with both turn ids, so the dispatch route that let the
    second turn through the busy guard can be identified from logs.

    Returns the previous in-flight turn_id when an overlap is detected,
    else None. Takes ownership of the in-flight slot either way, so a turn
    that crashed before its persist produces at most one warning."""
    prev = getattr(agent, "_inflight_turn_id", None)
    prev_started = getattr(agent, "_inflight_turn_started", 0.0)
    agent._inflight_turn_id = turn_id
    agent._inflight_turn_started = time.time()
    overlap = None
    if prev and prev != turn_id:
        logger.warning(
            "turn %s starting while turn %s (started %.0fs ago) has not "
            "completed its turn-end persist (session=%s) — concurrent turns "
            "on one session; transcript writes may interleave",
            turn_id,
            prev,
            time.time() - prev_started if prev_started else -1.0,
            getattr(agent, "session_id", None) or "-",
        )
        overlap = prev

    # Cross-agent leg: same session_id in flight under a different agent
    # object means two routing keys resolve to one durable session — the
    # busy guard (keyed by routing key) cannot see this overlap at all.
    # Persist-disabled agents (background-review forks) deliberately share
    # the live parent's session_id for prompt-cache warmth but can never
    # write to the transcript — they must not register here (would warn a
    # false overlap against the parent's real turn) nor pop the parent's
    # slot at their persist (note_turn_persisted skips them symmetrically).
    session_id = getattr(agent, "session_id", None)
    if session_id and not getattr(agent, "_persist_disabled", False):
        now = time.time()
        with _INFLIGHT_TURNS_LOCK:
            entry = _INFLIGHT_TURNS_BY_SESSION.get(session_id)
            _INFLIGHT_TURNS_BY_SESSION[session_id] = (turn_id, now)
        # Stamp the session id this turn registered under: compression can
        # rotate agent.session_id mid-turn, and the persist-time clear must
        # pop the slot the turn actually holds, not the rotated id.
        agent._inflight_turn_session_id = session_id
        if entry and entry[0] not in (turn_id, prev):
            logger.warning(
                "turn %s starting while turn %s (started %.0fs ago) is still "
                "in flight on session %s under a different agent object — "
                "two routing keys are mapped to one session_id; concurrent "
                "turns on one session; transcript writes may interleave",
                turn_id,
                entry[0],
                now - entry[1] if entry[1] else -1.0,
                session_id,
            )
            overlap = overlap or entry[0]
    return overlap

def note_turn_persisted(agent):
    """Clear the in-flight marker at turn-end persist (see note_turn_start).

    Called from the single persist funnel; unconditional by design — when two
    turns genuinely overlap, the first persist clears the second turn's slot
    and the tripwire under-reports instead of double-reporting. A diagnostic
    must never be noisier than the defect it hunts."""
    agent._inflight_turn_id = None
    # Symmetric with note_turn_start's cross-agent leg: persist-disabled
    # forks never registered a session slot, and their persist funnel still
    # runs — popping here would steal the live parent turn's slot and make
    # the tripwire under-report the real overlap it exists to catch.
    if not getattr(agent, "_persist_disabled", False):
        session_id = getattr(agent, "_inflight_turn_session_id", None) or getattr(
            agent, "session_id", None
        )
        if session_id:
            with _INFLIGHT_TURNS_LOCK:
                _INFLIGHT_TURNS_BY_SESSION.pop(session_id, None)
    agent._inflight_turn_session_id = None

def repair_message_sequence(agent, messages: List[Dict]) -> int:
    """Collapse malformed role-alternation left in the live history.

    Providers (OpenAI, OpenRouter, Anthropic) expect strict alternation:
    after the system message, user/tool alternates with assistant, with
    no two consecutive user messages and no tool-result that doesn't
    follow an assistant-with-tool_calls. Violations cause silent empty
    responses on most providers, which triggers the empty-retry loop.

    This runs right before the API call as a defensive belt — by the
    time it fires, the scaffolding strip should already have prevented
    most shapes, but external callers (gateway multi-queue replay,
    session resume, cron, explicit conversation_history passed in by
    host code) can feed in already-broken histories.

    Repairs applied:
      0. Consecutive ``assistant`` messages with no intervening
         ``tool``/``user`` turn — merged into a single assistant turn
         (union of ``tool_calls``, concatenated ``content``). Strict
         OpenAI-compatible providers (DeepSeek v4, Moonshot/Kimi) reject
         a history where an ``assistant`` message carrying ``tool_calls``
         is immediately followed by another ``assistant`` message instead
         of its ``tool`` results — HTTP 400 "An assistant message with
         'tool_calls' must be followed by tool messages…". The split
         shape is produced by recovery/continuation paths that append an
         interim assistant turn (thinking-prefill, codex
         incomplete-continuation) or by host-fed / legacy-persisted /
         resumed histories. Refs #29148, #49147.
      1. Stray ``tool`` messages whose ``tool_call_id`` doesn't match
         any preceding assistant tool_call — dropped.
      2. Consecutive ``user`` messages — merged with newline separator
         so no user input is lost.

    Deliberately does NOT rewind orphan ``assistant(tool_calls)+tool``
    pairs that precede a user message — that pattern IS valid when the
    previous turn completed normally and the user jumped in to redirect
    before the model got a continuation turn (the ongoing dialog
    pattern). The empty-response scaffolding stripper handles the
    genuinely-broken variant via its flag-gated rewind.

    Returns the number of repairs made (for logging/telemetry).
    """
    if not messages:
        return 0

    repairs = 0

    # Pass 0: merge consecutive assistant messages. Runs BEFORE Pass 1 so
    # the merged turn's union of tool_call ids is known when Pass 1
    # validates which tool-result messages are orphans. Two assistant
    # messages are only adjacent here when nothing (no tool result, no
    # user turn) separates them — an intervening ``tool`` message means
    # two distinct, valid tool-call rounds that must NOT be merged.
    #
    # Codex Responses interim turns are exempt: the codex_responses
    # api_mode legitimately keeps multiple consecutive incomplete
    # assistant turns in history, each carrying its own encrypted
    # continuation state (codex_reasoning_items / codex_message_items)
    # that must be replayed verbatim. Collapsing them corrupts the
    # Responses replay chain (the duplicate-detection logic at
    # conversation_loop.py already de-dups identical codex interims).
    def _is_codex_interim(m: Dict) -> bool:
        return bool(
            m.get("codex_reasoning_items")
            or m.get("codex_message_items")
            or m.get("finish_reason") == "incomplete"
        )

    def _is_verification_candidate(m: Dict) -> bool:
        return m.get("finish_reason") in {
            "verification_required",
            "verify_hook_continue",
        }

    collapsed: List[Dict] = []
    for msg in messages:
        if (
            collapsed
            and isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(collapsed[-1], dict)
            and collapsed[-1].get("role") == "assistant"
            and not _is_codex_interim(msg)
            and not _is_codex_interim(collapsed[-1])
        ):
            prev = collapsed[-1]
            # Verification candidate collapsing: when the earlier assistant
            # message is a provisional candidate (finish_reason =
            # verification_required / verify_hook_continue), the later
            # response supersedes it for model replay — replace rather than
            # union. Both remain durable in state.db; this only affects the
            # in-memory sequence sent to the model. (#65919 §7)
            if _is_verification_candidate(prev):
                collapsed[-1] = msg
                repairs += 1
                continue
            # Union tool_calls (preserve order, both may carry them).
            prev_calls = list(prev.get("tool_calls") or [])
            new_calls = list(msg.get("tool_calls") or [])
            if new_calls:
                prev["tool_calls"] = prev_calls + new_calls
            elif prev_calls:
                prev["tool_calls"] = prev_calls
            else:
                # Neither turn carries tool calls, but the surviving turn may
                # still carry a stale ``tool_calls: []`` from the earlier
                # message.  An empty array is semantically "no tool calls",
                # yet strict OpenAI-compatible providers (DeepSeek v4,
                # Moonshot/Kimi) reject it with HTTP 400 ("Invalid
                # 'messages[N].tool_calls': empty array...").  Drop the key
                # HERE, at the source: ``sanitize_api_messages`` only fixes
                # the per-call wire copy, so a ``[]`` left on the repaired
                # turn survives in the live/persisted trajectory returned to
                # callers (gateway/WebUI transcripts, session resume,
                # subagents, cron) and is replayed on the next turn — which
                # is how #58755 kept reproducing after the chokepoint fix
                # (#77921).  Popping is non-destructive: an empty array
                # carries no information.
                prev.pop("tool_calls", None)
            # Concatenate plain-text content; leave multimodal (list)
            # content on either side alone to avoid mangling attachment
            # blocks — fall back to keeping the existing content.
            prev_content = prev.get("content")
            new_content = msg.get("content")
            if isinstance(prev_content, str) and isinstance(new_content, str):
                joined = "\n".join(
                    p for p in (prev_content.strip(), new_content.strip()) if p
                )
                prev["content"] = joined
            elif not prev_content and new_content is not None:
                prev["content"] = new_content
            # Carry reasoning_content from the later turn only if the
            # earlier turn lacks it (strict thinking providers require a
            # reasoning_content on the merged tool-call turn; the first
            # non-empty one suffices).
            if not prev.get("reasoning_content") and msg.get("reasoning_content"):
                prev["reasoning_content"] = msg["reasoning_content"]
            repairs += 1
            continue
        collapsed.append(msg)

    # Pass 1: drop stray tool messages that don't follow a known
    # assistant tool_call_id. Uses a rolling set of known ids refreshed
    # on each assistant message.
    #
    # Both ``id`` AND ``call_id`` are registered for every assistant
    # tool_call. In the Codex Responses API format the two differ
    # (``id`` = ``fc_...`` response-item id, ``call_id`` = ``call_...``
    # the function-call id), and a tool result's ``tool_call_id`` may be
    # matched against *either* depending on which code path built it
    # (the OpenAI-compatible path stores ``tc.id``; codex paths store
    # ``call_id``). Registering only ``id`` — as this pass did before —
    # made a valid tool result look orphaned whenever the assistant
    # tool_call carried a distinct ``call_id`` (or only ``call_id``); the
    # pass then dropped it, leaving the assistant tool_call unanswered and
    # producing an HTTP 400 on strict providers (DeepSeek, Kimi). Matching
    # on the *superset* of both keys achieves the same tolerance as
    # ``_get_tool_call_id_static``'s ``call_id || id`` — a match set must
    # accept every legitimate reference, not just the canonical one (#58168).
    known_tool_ids: set = set()
    filtered: List[Dict] = []
    for msg in collapsed:
        if not isinstance(msg, dict):
            filtered.append(msg)
            continue
        role = msg.get("role")
        if role == "assistant":
            known_tool_ids = set()
            for tc in (msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                for key in ("id", "call_id"):
                    tc_id = tc.get(key)
                    if tc_id:
                        known_tool_ids.add(tc_id)
            filtered.append(msg)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id in known_tool_ids:
                filtered.append(msg)
                # Consume the id so a SECOND tool result carrying the same
                # tool_call_id (duplicate from a retry/crash/session-resume
                # glitch) falls into the drop branch below instead of being
                # replayed — strict providers (DeepSeek) reject a duplicate
                # tool_call_id with HTTP 400 (#58327). Credit: #55436.
                known_tool_ids.discard(tc_id)
            else:
                repairs += 1
        else:
            if role == "user":
                # A user turn closes the tool-result run; subsequent
                # tool messages without a fresh assistant tool_call
                # are orphans.
                known_tool_ids = set()
            filtered.append(msg)

    # Pass 2: merge consecutive user messages. Preserves all user input
    # so nothing the user typed is lost.
    merged: List[Dict] = []
    for msg in filtered:
        if (
            merged
            and isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(merged[-1], dict)
            and merged[-1].get("role") == "user"
        ):
            prev = merged[-1]
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")
            # Only merge plain-text content; leave multimodal (list)
            # content alone — collapsing image/audio blocks risks
            # mangling the attachment structure.
            if isinstance(prev_content, str) and isinstance(new_content, str):
                prev["content"] = (
                    (prev_content + "\n\n" + new_content)
                    if prev_content and new_content
                    else (prev_content or new_content)
                )
                # Merged content invalidates the api_content sidecar (exact
                # bytes previously sent for the pre-merge message) — drop it
                # so replay can't substitute stale bytes.
                drop_stale_api_content(prev)
                repairs += 1
                continue
        merged.append(msg)

    if repairs > 0:
        # Rewrite in place so downstream paths (persistence, return
        # value, session DB flush) see the repaired sequence.
        messages[:] = merged

    return repairs

def repair_message_sequence_with_cursor(agent, messages: List[Dict]) -> int:
    """Run :func:`repair_message_sequence` and keep the SessionDB flush
    cursor consistent with the compacted list (#44837).

    ``repair_message_sequence`` merges/drops messages in place, shrinking
    the list. ``_last_flushed_db_idx`` (the DB-write cursor) indexes into
    that list, so after compaction it can point past the new end — the
    turn-end flush would then skip the assistant/tool chain entirely — or
    past unflushed messages shifted to lower indexes.

    Repair preserves object identity for surviving messages, so counting
    the survivors from the previously-flushed prefix gives the exact new
    cursor even when messages are dropped/merged at indexes *before* the
    cursor — a plain ``min()`` clamp would silently skip that many
    unflushed rows. Falls back to the clamp when no prefix snapshot is
    available.

    Returns the number of repairs made (same as ``repair_message_sequence``).
    """
    pre_repair_flushed_ids = None
    flush_cursor = getattr(agent, "_last_flushed_db_idx", None)
    if isinstance(flush_cursor, int) and flush_cursor > 0:
        pre_repair_flushed_ids = {id(m) for m in messages[:flush_cursor]}

    repairs = repair_message_sequence(agent, messages)

    if repairs > 0 and hasattr(agent, "_last_flushed_db_idx"):
        if pre_repair_flushed_ids is not None:
            agent._last_flushed_db_idx = sum(
                1 for m in messages if id(m) in pre_repair_flushed_ids
            )
        else:
            agent._last_flushed_db_idx = min(
                agent._last_flushed_db_idx, len(messages)
            )

    return repairs
