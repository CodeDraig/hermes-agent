"""Execution of one gateway agent turn and its progress/status callbacks."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

from agent.async_utils import safe_schedule_threadsafe
from gateway.config import Platform
from gateway.agent_cache import AgentCacheEntry
from gateway.history import (
    _build_gateway_agent_history,
    _collect_auto_append_media_tags,
    _collect_history_media_paths,
    _is_fresh_gateway_interruption,
    _last_transcript_timestamp,
    _message_timestamps_enabled,
    _select_cached_agent_history,
    _wrap_current_message_with_observed_context,
    build_resume_recovery_note,
    strip_stale_dangerous_confirmations,
)
from gateway.notices import (
    _send_or_update_status_coro,
    non_conversational_metadata as _non_conversational_metadata,
    render_notice_line,
)
from gateway.session import auto_continue_freshness_window
from gateway.platforms.base import BasePlatformAdapter
from gateway.response_filters import (
    _format_exec_approval_fallback,
    _normalize_empty_agent_response,
    _prepare_gateway_status_message,
    _redact_approval_command,
    _redact_gateway_user_facing_secrets,
    _sanitize_gateway_final_response,
)
from gateway.runtime_config import (
    _checkpoint_agent_kwargs,
    _current_max_iterations,
    _load_gateway_config,
)
from gateway.turn_context import TurnContext
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home
from utils import is_truthy_value

from gateway.session_state import AGENT_PENDING as _AGENT_PENDING_SENTINEL

logger = logging.getLogger("gateway.run")
_hermes_home = get_hermes_home()

class TurnRunner:
    """Execute one turn using explicit runtime values and callbacks."""

    def __init__(
        self,
        ctx: TurnContext,
        *,
        session_db: Any,
        session_store: Any,
        sessions: Any,
        agent_cache: Any,
        prefill_messages: Any,
        provider_routing: Any,
        reasoning_config: Any,
        service_tier: Any,
        streaming_config: Any,
        adapter_for_source: Callable[..., Any],
        agent_config_signature: Callable[..., Any],
        apply_fallback_chain_to_agent: Callable[..., Any],
        build_stream_consumer_config: Callable[..., Any],
        consume_pending_native_image_paths: Callable[..., Any],
        deliver_platform_notice: Callable[..., Any],
        extract_cache_busting_config: Callable[..., Any],
        get_system_prompt_for_channel: Callable[..., Any],
        is_telegram_topic_lane: Callable[..., Any],
        refresh_fallback_providers: Callable[..., Any],
        resolve_session_agent_runtime: Callable[..., Any],
        resolve_session_reasoning_config: Callable[..., Any],
        resolve_session_service_tier: Callable[..., Any],
        resolve_turn_agent_config: Callable[..., Any],
        schedule_telegram_topic_title_rename: Callable[..., Any],
        sync_session_model_from_agent: Callable[..., Any],
        sync_telegram_topic_binding: Callable[..., Any],
    ) -> None:
        self._ctx = ctx
        self._session_db = session_db
        self.session_store = session_store
        self.sessions = sessions
        self.agent_cache = agent_cache
        self._prefill_messages = prefill_messages
        self._provider_routing = provider_routing
        self._reasoning_config = reasoning_config
        self._service_tier = service_tier
        self._streaming_config = streaming_config
        self._adapter_for_source = adapter_for_source
        self._agent_config_signature = agent_config_signature
        self._apply_fallback_chain_to_agent = apply_fallback_chain_to_agent
        self._build_stream_consumer_config = build_stream_consumer_config
        self._consume_pending_native_image_paths = consume_pending_native_image_paths
        self._deliver_platform_notice = deliver_platform_notice
        self._extract_cache_busting_config = extract_cache_busting_config
        self._get_system_prompt_for_channel = get_system_prompt_for_channel
        self._is_telegram_topic_lane = is_telegram_topic_lane
        self._refresh_fallback_providers = refresh_fallback_providers
        self._resolve_session_agent_runtime = resolve_session_agent_runtime
        self._resolve_session_reasoning_config = resolve_session_reasoning_config
        self._resolve_session_service_tier = resolve_session_service_tier
        self._resolve_turn_agent_config = resolve_turn_agent_config
        self._schedule_telegram_topic_title_rename = schedule_telegram_topic_title_rename
        self._sync_session_model_from_agent = sync_session_model_from_agent
        self._sync_telegram_topic_binding = sync_telegram_topic_binding

    def progress_callback(self, event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
        """Callback invoked by agent on tool lifecycle events."""
        ctx = self._ctx
        # Live status line (Slack's assistant status): stash the current
        # tool phrase on the adapter; the _keep_typing refresh renders it
        # within a couple of seconds. Handled before every other gate
        # because it's independent of progress bubbles and queues (Slack
        # keeps tool_progress off by default, but the ephemeral status
        # line is always safe). Plain dict write — safe from the agent's
        # sync worker thread, no event-loop hop needed.
        if (
            ctx._live_status_adapter is not None
            and ctx._live_status_mode != "off"
            and tool_name != "_thinking"
        ):
            try:
                if event_type == "tool.started" and tool_name and ctx._run_still_current():
                    from agent.display import build_status_phrase
                    _phrase = build_status_phrase(
                        tool_name,
                        args if ctx._live_status_mode == "full" else None,
                    )
                    ctx._live_status_adapter.set_status_text(ctx.source.chat_id, _phrase)
                elif event_type == "tool.completed":
                    # Between tools the model is genuinely "thinking"
                    # again — revert to the static default.
                    ctx._live_status_adapter.set_status_text(ctx.source.chat_id, None)
            except Exception as _ls_err:
                logger.debug("live status update failed: %s", _ls_err)
        # "log" mode: append tool.started lines to the log queue and stay
        # silent in chat. Handled before the progress_queue guard because
        # log mode runs without a chat progress queue.
        if ctx.log_queue is not None:
            if event_type == "tool.started" and tool_name and tool_name != "_thinking":
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                preview_str = f' "{preview}"' if preview else ""
                ctx.log_queue.put(f"{ts}  {tool_name}:{preview_str}".rstrip())
            if not ctx.progress_queue:
                return
        if not ctx.progress_queue or not ctx._run_still_current():
            return

        # First-touch onboarding: the first time a tool takes longer than
        # _LONG_TOOL_THRESHOLD_S during a run that's streaming every tool
        # (progress_mode == "all"), append a one-time hint suggesting
        # /verbose.  We only fire when (a) the user hasn't seen the hint
        # before and (b) /verbose is actually usable on this platform
        # (gateway gate must be open).  The CLI has its own trigger.
        if event_type == "tool.completed" and not ctx.long_tool_hint_fired[0]:
            try:
                duration = kwargs.get("duration") or 0
                if duration >= ctx._LONG_TOOL_THRESHOLD_S and ctx.progress_mode == "all":
                    from agent.onboarding import (
                        TOOL_PROGRESS_FLAG,
                        is_seen,
                        mark_seen,
                        tool_progress_hint_gateway,
                    )
                    _cfg = _load_gateway_config()
                    gate_on = is_truthy_value(
                        cfg_get(_cfg, "display", "tool_progress_command"),
                        default=False,
                    )
                    if gate_on and not is_seen(_cfg, TOOL_PROGRESS_FLAG):
                        ctx.long_tool_hint_fired[0] = True
                        ctx.progress_queue.put(tool_progress_hint_gateway())
                        mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
            except Exception as _hint_err:
                logger.debug("tool-progress onboarding hint failed: %s", _hint_err)
            return

        # "_thinking" is assistant scratch text between tool calls.  It
        # is never ordinary tool progress: only relay it when the platform
        # explicitly opted into thinking_progress.  Handle both legacy
        # callback shapes: ("_thinking", text) and
        # ("reasoning.available", "_thinking", text, ...).
        if event_type == "_thinking" or tool_name == "_thinking":
            if not ctx._thinking_enabled:
                return
            thinking_text = preview if tool_name == "_thinking" else tool_name
            msg = f"💬 {thinking_text}" if thinking_text else None
            if msg:
                ctx.progress_queue.put(msg)
            return

        # If tool_progress is off, only _thinking passes through (above).
        # Regular tool calls are suppressed.
        if not ctx.tool_progress_enabled:
            return

        # Only act on tool.started events (ignore tool.completed, reasoning.available, etc.)
        if event_type not in {"tool.started",}:
            return

        # Never render a progress bubble for the clarify tool.  The
        # adapter's send_clarify IS the user-facing rendering (interactive
        # buttons or the numbered-text fallback), so a progress bubble is
        # pure duplication — and in verbose mode it dumps the raw
        # tool-call args JSON ({"question": ..., "choices": [...]}) into
        # the chat.  Because the progress queue drains on a background
        # task, that raw JSON typically lands right underneath the
        # rendered prompt (#52374).
        if tool_name == "clarify":
            return

        # Suppress tool-progress bubbles once the user has sent `stop`.
        # When the LLM response carries N parallel tool calls, the agent
        # fires N "tool.started" events back-to-back before checking for
        # interrupts — without this guard, a late `stop` still renders
        # all N as 🔍 bubbles, making the interrupt feel ignored.
        # (agent lives in run_sync's scope; agent_holder[0] is the shared
        # handle across nested scopes — see line ~9607.)
        try:
            _agent_for_interrupt = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent_for_interrupt is not None and getattr(
                _agent_for_interrupt, "is_interrupted", False
            ):
                return
        except Exception:
            pass

        # "new" mode: only report when tool changes
        if ctx.progress_mode == "new" and tool_name == ctx.last_tool[0]:
            return
        ctx.last_tool[0] = tool_name

        # Build progress message with primary argument preview
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")

        # Markdown-capable platforms render a terminal command as a fenced
        # code block instead of the compact `terminal: "cmd…"` preview.
        # Gated on the adapter's ``supports_code_blocks`` capability so
        # plain-text platforms keep the short line.  No language tag is
        # emitted — Slack mrkdwn renders the tag as a literal first code
        # line ("bash"), and a bare fence renders correctly everywhere
        # that supports blocks.
        #
        # Verbose mode shows the FULL command.  Non-verbose ("all"/"new")
        # modes still wrap in a fence but truncate to a single line capped
        # at ``tool_preview_length`` (default 40) so a long or multi-line
        # command doesn't render as a huge block — matching the budget the
        # non-terminal preview path already applies (#42634).
        _code_block_full = None
        _code_block_short = None
        try:
            _progress_adapter = self._adapter_for_source(ctx.source)
        except Exception:
            _progress_adapter = None
        if (
            getattr(_progress_adapter, "supports_code_blocks", False)
            and tool_name == "terminal"
            and isinstance(args, dict)
            and isinstance(args.get("command"), str)
            and args["command"].strip()
        ):
            from agent.display import get_tool_preview_max_len
            _cmd_full = args["command"].rstrip()
            # Consecutive terminal calls: drop the repeated
            # "💻 terminal" header so back-to-back commands render as
            # adjacent code blocks under a single header.
            _block_header = (
                "" if ctx.last_was_terminal_block[0] else f"{emoji} {tool_name}\n"
            )
            _code_block_full = f"{_block_header}```\n{_cmd_full}\n```"
            # Single-line, capped preview for non-verbose modes.
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _lines = _cmd_full.splitlines()
            _cmd_short = _lines[0] if _lines else _cmd_full
            _multiline = len(_lines) > 1
            if len(_cmd_short) > _cap:
                _cmd_short = _cmd_short[:_cap - 3] + "..."
            elif _multiline:
                _cmd_short = _cmd_short + " ..."
            _code_block_short = f"{_block_header}```\n{_cmd_short}\n```"

        # Verbose mode: show detailed arguments, respects tool_preview_length
        if ctx.progress_mode == "verbose":
            if _code_block_full is not None:
                ctx.last_was_terminal_block[0] = True
                ctx.progress_queue.put(_code_block_full)
                return
            ctx.last_was_terminal_block[0] = False
            if args:
                from agent.display import get_tool_preview_max_len
                _pl = get_tool_preview_max_len()
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                # When tool_preview_length is 0 (default), don't truncate
                # in verbose mode — the user explicitly asked for full
                # detail.  Platform message-length limits handle the rest.
                if _pl > 0 and len(args_str) > _pl:
                    args_str = args_str[:_pl - 3] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
            elif preview:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            ctx.progress_queue.put(msg)
            return

        # "all" / "new" modes: short preview, respects tool_preview_length
        # config (defaults to 40 chars when unset to keep gateway messages
        # compact — unlike CLI spinners, these persist as permanent messages).
        # Terminal commands on markdown platforms get a single-line capped
        # fenced block (built above) instead of the truncated preview.
        if _code_block_short is not None:
            msg = _code_block_short
            ctx.last_was_terminal_block[0] = True
        elif preview:
            from agent.display import (
                get_tool_preview_max_len,
                get_tool_verb,
                prepare_tool_preview,
                tool_verb_connector,
                verb_drops_preview,
            )
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _prepared_preview = prepare_tool_preview(
                tool_name,
                args,
                fallback=preview,
                max_len=_cap,
            )
            if _progress_adapter is not None:
                preview = _progress_adapter.format_tool_preview(_prepared_preview)
            else:
                preview = _prepared_preview.text
            # Friendly labels: render a human-phrased line for built-in
            # tools ("🔍 Searching the web for ...") by prefixing the verb
            # onto the preview the callback already computed (so the
            # command/url/query is preserved).  Custom/plugin/MCP tools
            # have no verb and fall back to the raw "tool_name: ..." form.
            _verb = get_tool_verb(tool_name)
            if _verb:
                if verb_drops_preview(tool_name):
                    msg = f"{emoji} {_verb}"
                else:
                    msg = f"{emoji} {_verb}{tool_verb_connector(tool_name)}{preview}"
            else:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            ctx.last_was_terminal_block[0] = False
        else:
            msg = f"{emoji} {tool_name}..."
            ctx.last_was_terminal_block[0] = False

        # Dedup: collapse consecutive identical progress messages.
        # Common with execute_code where models iterate with the same
        # code (same boilerplate imports → identical previews).
        if msg == ctx.last_progress_msg[0]:
            ctx.repeat_count[0] += 1
            # Update the last line in progress_lines with a counter
            # via a special "dedup" queue message.
            ctx.progress_queue.put(("__dedup__", msg, ctx.repeat_count[0]))
            return
        ctx.last_progress_msg[0] = msg
        ctx.repeat_count[0] = 0

        ctx.progress_queue.put(msg)


    async def send_progress_messages(self):
        ctx = self._ctx
        if not ctx.progress_queue:
            return

        adapter = self._adapter_for_source(ctx.source)
        if not adapter:
            return

        # Skip tool progress for platforms that don't support message
        # editing (e.g. iMessage/BlueBubbles) — each progress update
        # would become a separate message bubble, which is noisy.
        # getattr, not attribute access: duck-typed adapters (test fakes,
        # minimal plugin adapters) may not define edit_message at all —
        # "missing" means the same thing as "base no-op": can't edit.
        _adapter_edit = getattr(type(adapter), "edit_message", None)
        if _adapter_edit is None or _adapter_edit is BasePlatformAdapter.edit_message:
            while not ctx.progress_queue.empty():
                try:
                    ctx.progress_queue.get_nowait()
                except Exception:
                    break
            return

        progress_lines = []      # Accumulated tool lines for the CURRENT editable bubble
        progress_msg_id = None   # ID of the current progress message to edit
        can_edit = ctx.progress_grouping != "separate"  # "separate" = one message per tool (pre-v0.9 behavior)
        _last_edit_ts = 0.0      # Throttle edits to avoid Telegram flood control
        _PROGRESS_EDIT_INTERVAL = 1.5  # Minimum seconds between edits

        _progress_len_fn = (
            adapter.message_len_fn
            if isinstance(adapter, BasePlatformAdapter)
            else len
        )
        try:
            _raw_progress_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
        except Exception:
            _raw_progress_limit = 4000
        # Per-chat resolution (relay adapter fronting N platforms): the cap
        # and length unit follow the chat's underlying platform. Native
        # adapters return their scalar/property unchanged.
        if isinstance(adapter, BasePlatformAdapter):
            try:
                _raw_progress_limit = int(
                    adapter.max_message_length_for_chat(ctx.source.chat_id) or 4000
                )
                _progress_len_fn = adapter.message_len_fn_for_chat(ctx.source.chat_id)
            except Exception:
                pass
        # Leave a little room for platform quirks / formatting.  For tiny
        # test adapters keep the limit usable instead of clamping to 500+.
        _PROGRESS_TEXT_LIMIT = max(
            1,
            _raw_progress_limit - (64 if _raw_progress_limit > 128 else 0),
        )

        # Detect whether the adapter's edit_message accepts metadata so
        # overflow edits preserve Telegram topic/thread routing (#27487).
        _edit_accepts_metadata = False
        if ctx._progress_metadata:
            try:
                _edit_params = inspect.signature(adapter.edit_message).parameters
                _edit_accepts_metadata = (
                    "metadata" in _edit_params
                    or any(
                        param.kind is inspect.Parameter.VAR_KEYWORD
                        for param in _edit_params.values()
                    )
                )
            except (TypeError, ValueError):
                _edit_accepts_metadata = False

        async def _edit_progress_message(message_id: str, content: str):
            kwargs = {
                "chat_id": ctx.source.chat_id,
                "message_id": message_id,
                "content": content,
            }
            if getattr(adapter, "REQUIRES_EDIT_FINALIZE", False):
                kwargs["finalize"] = True
            if _edit_accepts_metadata:
                kwargs["metadata"] = ctx._progress_metadata
            return await adapter.edit_message(**kwargs)

        def _progress_text(lines: list) -> str:
            return "\n".join(str(line) for line in lines)

        def _split_progress_groups(lines: list) -> list[list]:
            """Partition progress lines into platform-sized editable bubbles."""
            groups: list[list] = []
            current: list = []
            for line in lines:
                candidate = current + [line]
                if current and _progress_len_fn(_progress_text(candidate)) > _PROGRESS_TEXT_LIMIT:
                    groups.append(current)
                    current = [line]
                else:
                    current = candidate
            if current:
                groups.append(current)
            return groups

        def _track_progress_result(result) -> None:
            if (
                ctx._cleanup_progress
                and getattr(result, "success", False)
                and getattr(result, "message_id", None)
            ):
                ctx._cleanup_msg_ids.append(str(result.message_id))

        async def _send_progress_text(text: str):
            result = await adapter.send(
                chat_id=ctx.source.chat_id,
                content=text,
                reply_to=ctx._progress_reply_to,
                metadata=ctx._progress_metadata,
            )
            _track_progress_result(result)
            return result

        async def _roll_progress_overflow_if_needed() -> bool:
            """Start fresh editable progress bubbles before a bubble exceeds limit.

                Returns True when it delivered/split the current buffer, or when
                a transient edit failure left the buffer and message identity
                intact for a later retry.  In either case the caller should skip
                the normal send/edit path for this tick.
                """
            nonlocal progress_msg_id, progress_lines, can_edit
            if not progress_lines or not can_edit:
                return False
            groups = _split_progress_groups(progress_lines)
            if len(groups) <= 1:
                return False

            first_text = _progress_text(groups[0])
            if progress_msg_id is not None:
                result = await _edit_progress_message(progress_msg_id, first_text)
                if not result.success:
                    if getattr(result, "retryable", False):
                        logger.debug(
                            "[%s] Transient overflow edit failure — keeping can_edit=True",
                            adapter.name,
                        )
                        return True
                    can_edit = False
                    # Fall back to the existing non-edit behavior below.
                    return False
            else:
                result = await _send_progress_text(first_text)
                if result.success and result.message_id:
                    progress_msg_id = result.message_id

            for group in groups[1:]:
                result = await _send_progress_text(_progress_text(group))
                if result.success and result.message_id:
                    progress_msg_id = result.message_id

            # The newest continuation is now the only mutable bubble.  Keep
            # just its lines so subsequent edits update it instead of
            # replaying the full historical transcript into new messages.
            progress_lines = groups[-1]
            return True

        while True:
            try:
                if not ctx._run_still_current():
                    while not ctx.progress_queue.empty():
                        try:
                            ctx.progress_queue.get_nowait()
                        except Exception:
                            break
                    return

                raw = ctx.progress_queue.get_nowait()

                # Drain silently when interrupted: events queued in the
                # window between tool parse and interrupt processing
                # should not render as bubbles.  The "⚡ Interrupting
                # current task" message is sent separately and is the
                # last progress-flavored bubble the user should see.
                try:
                    _agent_for_interrupt = ctx.agent_holder[0] if ctx.agent_holder else None
                    if _agent_for_interrupt is not None and getattr(
                        _agent_for_interrupt, "is_interrupted", False
                    ):
                        # Drop this event and continue draining.
                        await asyncio.sleep(0)
                        continue
                except Exception:
                    pass

                # Handle dedup messages: update last line with repeat counter
                if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                    _, base_msg, count = raw
                    if progress_lines:
                        progress_lines[-1] = f"{base_msg} (×{count + 1})"
                    msg = progress_lines[-1] if progress_lines else base_msg
                elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                    # Content bubble just landed on the platform — close off
                    # the current tool-progress bubble so the next tool
                    # starts a fresh bubble below the content. Without this,
                    # tool lines keep editing the ORIGINAL progress message
                    # above the new content, making the chat appear out of
                    # order. Mirrors GatewayStreamConsumer.on_segment_break
                    # on the content side. (Issue: tool + content
                    # linearization regression after PR #7885.)
                    progress_msg_id = None
                    progress_lines = []
                    ctx.last_progress_msg[0] = None
                    ctx.repeat_count[0] = 0
                    continue
                else:
                    msg = raw
                    progress_lines.append(msg)

                if await _roll_progress_overflow_if_needed():
                    _last_edit_ts = time.monotonic()
                    await asyncio.sleep(0.3)
                    if ctx._run_still_current():
                        await adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)
                    continue

                # Throttle edits: batch rapid tool updates into fewer
                # API calls to avoid hitting Telegram flood control.
                # (grammY auto-retry pattern: proactively rate-limit
                # instead of reacting to 429s.)
                _now = time.monotonic()
                _remaining = _PROGRESS_EDIT_INTERVAL - (_now - _last_edit_ts)
                if _remaining > 0:
                    # Wait out the throttle interval, then loop back to
                    # drain any additional queued messages before sending
                    # a single batched edit.
                    await asyncio.sleep(_remaining)
                    continue

                if not ctx._run_still_current():
                    return

                if can_edit and progress_msg_id is not None:
                    # Try to edit the existing progress message
                    full_text = "\n".join(progress_lines)
                    result = await _edit_progress_message(progress_msg_id, full_text)
                    if not result.success:
                        _err = (getattr(result, "error", "") or "").lower()
                        # Transient network errors (ConnectError, timeouts)
                        # must not permanently disable progress-message
                        # editing — the next cycle can catch up.  Only
                        # permanent failures (flood control, message not
                        # found, permissions) should set can_edit = False.
                        if getattr(result, "retryable", False):
                            logger.debug(
                                "[%s] Transient edit failure — keeping can_edit=True",
                                adapter.name,
                            )
                            continue
                        if "flood" in _err or "retry after" in _err:
                            # Flood control hit — backoff but keep editing.
                            # Only disable edits for non-recoverable errors.
                            logger.info(
                                "[%s] Progress edit flood control, backing off",
                                adapter.name,
                            )
                            _last_edit_ts = time.monotonic()
                        else:
                            can_edit = False
                        _flood_result = await adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=msg,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                        if (
                            ctx._cleanup_progress
                            and getattr(_flood_result, "success", False)
                            and getattr(_flood_result, "message_id", None)
                        ):
                            ctx._cleanup_msg_ids.append(str(_flood_result.message_id))
                else:
                    if can_edit:
                        # First tool: send all accumulated text as new message
                        full_text = "\n".join(progress_lines)
                        result = await adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=full_text,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                    else:
                        # Editing unsupported: send just this line
                        result = await adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=msg,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id
                        if ctx._cleanup_progress:
                            ctx._cleanup_msg_ids.append(str(result.message_id))

                _last_edit_ts = time.monotonic()

                # Restore typing indicator
                await asyncio.sleep(0.3)
                if ctx._run_still_current():
                    await adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)

            except queue.Empty:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                # Drain remaining queued messages
                while not ctx.progress_queue.empty():
                    try:
                        raw = ctx.progress_queue.get_nowait()
                        if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                            _, base_msg, count = raw
                            if progress_lines:
                                progress_lines[-1] = f"{base_msg} (×{count + 1})"
                                await _roll_progress_overflow_if_needed()
                        elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                            # Content-bubble marker during drain: close off
                            # the current progress bubble and start a fresh
                            # one for any tool lines that arrived after.
                            await _roll_progress_overflow_if_needed()
                            if can_edit and progress_lines and progress_msg_id:
                                _pending_text = _progress_text(progress_lines)
                                try:
                                    await _edit_progress_message(progress_msg_id, _pending_text)
                                except Exception:
                                    pass
                            progress_msg_id = None
                            progress_lines = []
                            ctx.last_progress_msg[0] = None
                            ctx.repeat_count[0] = 0
                        else:
                            progress_lines.append(raw)
                            await _roll_progress_overflow_if_needed()
                    except Exception:
                        break
                # Final edit with all remaining tools (only if editing works)
                if can_edit and progress_lines and progress_msg_id:
                    await _roll_progress_overflow_if_needed()
                if can_edit and progress_lines and progress_msg_id:
                    full_text = _progress_text(progress_lines)
                    try:
                        await _edit_progress_message(progress_msg_id, full_text)
                    except Exception:
                        pass
                return
            except Exception as e:
                logger.error("Progress message error: %s", e)
                await asyncio.sleep(1)


    def _step_callback_sync(self, iteration: int, prev_tools: list) -> None:
        ctx = self._ctx
        if not ctx._run_still_current():
            return
        # prev_tools may be list[str] or list[dict] with "name"/"result"
        # keys.  Normalise to keep "tool_names" backward-compatible for
        # user-authored hooks that do ', '.join(tool_names)'.
        _names: list[str] = []
        for _t in (prev_tools or []):
            if isinstance(_t, dict):
                _names.append(_t.get("name") or "")
            else:
                _names.append(str(_t))
        safe_schedule_threadsafe(
            ctx._hooks_ref.emit("agent:step", {
                "platform": ctx.source.platform.value if ctx.source.platform else "",
                "user_id": ctx.source.user_id,
                "session_id": ctx.session_id,
                "iteration": iteration,
                "tool_names": _names,
                "tools": prev_tools,
            }),
            ctx._loop_for_step,
            logger=logger,
            log_message="agent:step hook scheduling error",
        )

    def _event_callback_sync(self, event_type: str, context: dict) -> None:
        ctx = self._ctx
        try:
            asyncio.run_coroutine_threadsafe(
                ctx._hooks_ref.emit(event_type, context),
                ctx._loop_for_step,
            )
        except Exception as _e:
            logger.debug("event_callback hook error: %s", _e)

    def _attach_session_title_callback(self, agent, ctx) -> None:
        """Wire the platform thread-rename lane onto the agent as `_on_session_title`.

        The session titler runs inside the turn prologue now (it derives the
        title from the user's first message, so it no longer needs the
        response), which means the callback has to be attached before the run
        rather than registered after it. The lane predicates and their
        rationale are unchanged from the old post-response registration.
        """
        try:
            # Gateway auto-title failures must NOT be surfaced as user-visible
            # messages (#23246) — they are not actionable to the end user.
            # Overriding the failure sink here keeps CLI mode on the agent's
            # _emit_auxiliary_failure path while the gateway logs at debug.
            def _title_failure_cb(task: str, exc: BaseException) -> None:
                logger.debug(
                    "Gateway auto-title failure suppressed (not user-visible): %s: %s",
                    task, exc,
                )

            agent._title_failure_callback = _title_failure_cb

            session_id = getattr(agent, "session_id", None)
            source = ctx.source

            # Both lanes below spend a rate-limited platform call per title, so
            # they take the model's title and skip the derived one — see
            # TitleCallback. Renaming twice lands on the same name at twice the
            # cost, and Discord's 2-per-10-minutes channel budget can spend
            # itself on the throwaway and drop the one worth showing.
            if self._is_telegram_topic_lane(source):
                agent._on_session_title = lambda title, title_source: (
                    title_source == "llm"
                    and self._schedule_telegram_topic_title_rename(
                        source, session_id, title,
                    )
                )
        except Exception:
            logger.debug("Failed to attach session title callback", exc_info=True)

    def _status_callback_sync(self, event_type: str, message: str) -> None:
        ctx = self._ctx
        if not ctx._status_adapter or not ctx._run_still_current():
            return
        prepared_message = _prepare_gateway_status_message(
            ctx.source.platform,
            event_type,
            message,
        )
        if prepared_message is None:
            logger.debug(
                "status_callback suppressed for %s/%s: %s",
                ctx.source.platform.value if ctx.source.platform else "unknown",
                event_type,
                _redact_gateway_user_facing_secrets(str(message or ""))[:160],
            )
            return
        _fut = safe_schedule_threadsafe(
            _send_or_update_status_coro(ctx._status_adapter, ctx._status_chat_id, event_type, prepared_message, ctx._status_thread_metadata),
            ctx._loop_for_step,
            logger=logger,
            log_message=f"status_callback ({event_type}) scheduling error",
        )
        if _fut is None:
            return
        if ctx._cleanup_progress:
            def _track_status_id(fut) -> None:
                try:
                    res = fut.result()
                except Exception:
                    return
                mid = getattr(res, "message_id", None)
                if getattr(res, "success", False) and mid:
                    ctx._cleanup_msg_ids.append(str(mid))
            _fut.add_done_callback(_track_status_id)

    def run_sync(self):
        import agent.lifecycle as lifecycle
        ctx = self._ctx
        # Historical note: as a nested closure this body declared
        # `nonlocal message` because the conditional re-assignments below
        # (prepending model-switch / resume-recovery notes) would otherwise
        # make `message` function-local and break the earlier read at
        # `_resolve_turn_agent_config(message, …)`.  As a method the turn
        # message lives on the shared TurnContext instead: every rebind
        # writes `ctx.message`, so the outer `_run_agent_inner` body observes
        # the updated value exactly as it did through the closure cell.

        # session_key is propagated via contextvars in _set_session_env()
        # (_SESSION_KEY) and via set_current_session_key() (_approval_session_key)
        # below — both concurrency-safe and inherited by tool worker threads.
        # We deliberately do NOT write os.environ["HERMES_SESSION_KEY"] here:
        # os.environ is process-global, so concurrent gateway sessions (e.g.
        # two Discord threads) would clobber each other's value, and a tool
        # thread whose contextvar is unset would fall back to os.environ and
        # read the wrong session key — misrouting command-approval prompts to
        # the wrong thread (#24100). The non-gateway surfaces don't depend on
        # this write: CLI and cron bind the session via contextvars
        # (set_current_session_key / session context), and only the TUI
        # slash-worker *subprocess* exports HERMES_SESSION_KEY (from its own
        # --session-key argv, a separate process) — so removing this in-process
        # gateway write does not affect any of them.

        # Map platform enum to the platform hint key the agent understands.
        # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
        platform_key = "cli" if ctx.source.platform == Platform.LOCAL else ctx.source.platform.value

        # Combine platform context, YAML channel_prompts hint for this chat,
        # channel_overrides system_prompt (or global ephemeral), and gateway
        # ephemeral prompt from _get_system_prompt_for_channel.
        combined_ephemeral = ctx.context_prompt or ""
        event_channel_prompt = (ctx.channel_prompt or "").strip()
        if event_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + event_channel_prompt).strip()
        cfg_channel_prompt = self._get_system_prompt_for_channel(
            ctx.source.platform,
            ctx.source.chat_id or "",
            thread_id=getattr(ctx.source, "thread_id", None),
            parent_id=getattr(ctx.source, "parent_chat_id", None),
        )
        if cfg_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + cfg_channel_prompt).strip()

        max_iterations = _current_max_iterations()

        try:
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=ctx.source,
                session_key=ctx.session_key,
                user_config=ctx.user_config,
            )
            logger.debug(
                "run_agent resolved: model=%s provider=%s session=%s",
                model, runtime_kwargs.get("provider"), ctx.session_key or "",
            )
        except Exception as exc:
            return {
                "final_response": f"⚠️ Provider authentication failed: {exc}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        pr = self._provider_routing
        reasoning_config = self._resolve_session_reasoning_config(
            source=ctx.source,
            session_key=ctx.session_key,
            model=model,
        )
        self._reasoning_config = reasoning_config
        self._service_tier = self._resolve_session_service_tier(
            source=ctx.source, session_key=ctx.session_key
        )
        # Set up stream consumer for token streaming or interim commentary.
        _stream_consumer = None
        _stream_delta_cb = None
        # #60671 — streaming TTS consumer is created on the outer
        # event-loop thread before run_sync launches.  run_sync only
        # reads it via ``streaming_tts_consumer_holder[0]`` for delta
        # callback wiring.
        _stts_consumer_ref = ctx.streaming_tts_consumer_holder[0]
        _scfg = self._streaming_config

        # Per-platform streaming gate: display.platforms.<plat>.streaming
        # can disable streaming for specific platforms even when the global
        # streaming config is enabled.
        _plat_streaming = ctx.resolve_display_setting(
            ctx.user_config, platform_key, "streaming"
        )
        # None = no per-platform override → follow global config
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )
        _want_stream_deltas = _streaming_enabled
        _want_interim_messages = ctx.interim_assistant_messages_enabled
        _want_interim_consumer = _want_interim_messages
        if _want_stream_deltas or _want_interim_consumer:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                _adapter = self._adapter_for_source(ctx.source)
                if _adapter:
                    _consumer_cfg, _pause_typing_before_finalize = (
                        self._build_stream_consumer_config(
                            ctx.source, _scfg, _adapter,
                            on_missing_cursor="raise",
                        )
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=ctx.source.chat_id,
                        config=_consumer_cfg,
                        metadata=ctx._status_thread_metadata,
                        on_new_message=(
                            (lambda: ctx.progress_queue.put(("__reset__",)))
                            if ctx.progress_queue is not None
                            else None
                        ),
                        on_before_finalize=_pause_typing_before_finalize,
                        initial_reply_to_id=ctx.event_message_id,
                        run_still_current=ctx._run_still_current,
                    )
                    if _want_stream_deltas:
                        def _stream_delta_cb(text: str) -> None:
                            if ctx._run_still_current():
                                _stream_consumer.on_delta(text)
                                # Tee to the streaming-TTS consumer (#60671).
                                if _stts_consumer_ref is not None:
                                    _stts_consumer_ref.on_delta(text)
                    ctx.stream_consumer_holder[0] = _stream_consumer
            except Exception as _sc_err:
                logger.debug("Could not set up stream consumer: %s", _sc_err)

        # When text streaming is off but streaming TTS is active,
        # install a TTS-only delta callback so the consumer still
        # receives LLM deltas for audio synthesis (#60671).
        if _stream_delta_cb is None and _stts_consumer_ref is not None:
            def _stream_delta_cb(text: str) -> None:
                if ctx._run_still_current():
                    _stts_consumer_ref.on_delta(text)

        def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            display_text = text
            if _stream_consumer is not None:
                if already_streamed:
                    _stream_consumer.on_segment_break()
                else:
                    _stream_consumer.on_commentary(display_text)
                return
            if already_streamed or not ctx._status_adapter or not str(display_text or "").strip():
                return
            safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    display_text,
                    metadata=ctx._status_thread_metadata,
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="interim_assistant_callback scheduling error",
            )

        turn_route = self._resolve_turn_agent_config(ctx.message, model, runtime_kwargs)

        # Per-platform skip_context_files — messaging platforms can opt out
        # of filesystem-heavy context-file discovery (SOUL.md, AGENTS.md,
        # .cursorrules) to cut create_agent construction latency. Especially
        # impactful on Windows, where stat() + directory walks are 10-100x
        # slower than Linux. Off by default; soul identity is preserved so
        # the persona survives even with minimal context.
        _platforms_gw_cfg = (ctx.user_config.get("gateway") or {}).get("platforms") or {}
        # ``hermes gateway setup`` writes ``gateway.platforms`` as a LIST of
        # enabled platform names (e.g. ``- telegram``), not a dict.  Treat any
        # non-dict shape as "no per-platform overrides" instead of crashing
        # on ``.get()`` for every incoming turn (#83185).
        if not isinstance(_platforms_gw_cfg, dict):
            _platforms_gw_cfg = {}
        _plat_gw_cfg = _platforms_gw_cfg.get(platform_key) or {}
        _skip_context = _plat_gw_cfg.get("skip_context_files")
        skip_context_files = bool(_skip_context) if _skip_context is not None else False

        # Check agent cache — reuse the create_agent from the previous message
        # in this session to preserve the frozen system prompt and tool
        # schemas for prompt cache hits.
        _sig = self._agent_config_signature(
            turn_route["model"],
            turn_route["runtime"],
            ctx.enabled_toolsets,
            combined_ephemeral,
            cache_keys=self._extract_cache_busting_config(ctx.user_config),
            user_id=getattr(ctx.source, "user_id", None),
            user_id_alt=getattr(ctx.source, "user_id_alt", None),
            skip_context_files=skip_context_files,
        )
        agent = None
        reused_cached_agent = False
        _cache_lock = self.agent_cache.lock
        _cache = self.agent_cache.entries

        # Peek at the cached entry's snapshot session_id (if any) so we can
        # check, OUTSIDE the cache lock, whether THAT session_id is a DEAD
        # session in state.db. This closes a gap in the #54947 fix: that
        # fix treats "cached session_id != current session_id" as an
        # intentional /resume-style switch and reuses the agent unchanged.
        # But the #54878 self-heal produces the exact same tuple shape
        # when it recovers a routing key away from a session that was
        # already ended — the cached create_agent still belongs to the DEAD
        # session, not a valid sibling conversation. Reusing it lets that
        # turn's post-run "session split" sync write the routing key
        # straight back onto the dead session_id, undoing the self-heal
        # and looping every message until an interrupt happens to race in
        # first (the #54878 x #54947 interaction — no existing upstream
        # issue tracks this combination as of 2026-07-12).
        _peek_cached_sid = None
        with _cache_lock:
            _peek_entry = _cache.get(ctx.session_key)
        if _peek_entry:
            _peek_cached_sid = _peek_entry.session_id
        _cached_sid_is_dead = False
        if (
            _peek_cached_sid is not None
            and ctx.session_id is not None
            and _peek_cached_sid != ctx.session_id
        ):
            try:
                _cached_sid_is_dead = self.session_store._is_session_ended_in_db(
                    _peek_cached_sid
                )
            except Exception:
                _cached_sid_is_dead = False

        # Detect cross-process writes: when another process (e.g. hermes
        # dashboard) appends to the same session in the shared SessionDB,
        # the cached agent's in-memory transcript becomes stale.  Compare
        # the session's current message_count against the count recorded
        # when the agent was cached; on mismatch, invalidate the cache
        # so a fresh agent re-reads from disk. (#45966)
        _current_msg_count = None
        if self._session_db is not None and ctx.session_id:
            try:
                # run_sync is off-loop (executor); sync DB is fine.
                _sess_row = self._session_db._db.get_session(ctx.session_id)
                if _sess_row:
                    _current_msg_count = _sess_row.get("message_count", 0)
            except Exception:
                pass

        _xproc_evicted_agent = None
        with _cache_lock:
                cached = _cache.get(ctx.session_key)
                if cached and cached.signature == _sig:
                    # The cached message count becomes stale when a second
                    # process appends rows.  Its session ID identifies which
                    # conversation supplied that snapshot (#54947).
                    _cached_mc = cached.message_count
                    _cached_sid = cached.session_id
                    # If the snapshot belongs to a different session_id
                    # (same session_key, different conversation), the
                    # message_count comparison is meaningless — the
                    # counts track DIFFERENT DB rows.  REUSE the cached
                    # agent rather than rebuild and bust the prompt cache
                    # on every session switch (#54947).
                    _session_id_mismatch = (
                        _cached_sid is not None
                        and ctx.session_id is not None
                        and _cached_sid != ctx.session_id
                    )
                    # Re-validate the OUTSIDE-lock dead-session peek
                    # against the tuple actually read under THIS lock —
                    # the cache entry could have been replaced between
                    # the peek and this lock acquisition, and a stale
                    # "dead" verdict must never be applied to a
                    # different (possibly live) cached agent.
                    _stale_dead_sid_reuse = (
                        _session_id_mismatch
                        and _cached_sid_is_dead
                        and _cached_sid == _peek_cached_sid
                    )
                    if _stale_dead_sid_reuse:
                        # #54878 x #54947 interaction: the routing key
                        # was just self-healed away from a session that
                        # state.db already marked ended, but the cached
                        # create_agent here still belongs to that DEAD
                        # session_id. The #54947 "different session_id
                        # under the same key = intentional switch, reuse
                        # freely" rule does not hold here — this isn't a
                        # sibling conversation, it's a stale agent left
                        # over from before the self-heal. Reusing it lets
                        # this turn's post-run "session split" sync write
                        # the routing key straight back onto the dead
                        # session_id, undoing the self-heal and looping
                        # every message until an interrupt happens to
                        # race in first. Discard and rebuild fresh
                        # instead, same as a genuine cross-process write.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "cached agent's session_id %s is ended in "
                            "state.db (stale self-heal artifact, "
                            "#54878 x #54947) — discarding instead of "
                            "reusing across the routing recovery",
                            ctx.session_key, _cached_sid,
                        )
                        evicted = self.agent_cache.entries.pop(ctx.session_key, None)
                        _ev_agent = evicted.agent if evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Same deferred-cleanup rationale as the
                            # cross-process branch below (#52197): don't
                            # block the event loop / cache lock on
                            # memory-provider shutdown or socket teardown.
                            _xproc_evicted_agent = _ev_agent
                    elif (
                        not _session_id_mismatch
                        and _cached_mc is not None
                        and _current_msg_count is not None
                        and _current_msg_count != _cached_mc
                    ):
                        # Cross-process write detected — discard stale
                        # agent so it rebuilds from fresh DB transcript.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "message_count changed (%s -> %s), "
                            "possible cross-process write",
                            ctx.session_key, _cached_mc, _current_msg_count,
                        )
                        evicted = self.agent_cache.entries.pop(ctx.session_key, None)
                        _ev_agent = evicted.agent if evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Defer cleanup until AFTER the lock is
                            # released — _cleanup_agent_resources /
                            # release_clients can block on memory-provider
                            # shutdown and socket teardown, and running it
                            # here would stall the gateway event loop while
                            # _sweep_idle_cached_agents (session-expiry
                            # watcher) waits on the same lock, blocking
                            # Discord heartbeats (#52197).  The same session
                            # rebuilds a fresh agent immediately below, so
                            # use the SOFT release that preserves the
                            # session's terminal sandbox / browser / bg
                            # processes for the rebuilt agent to inherit —
                            # mirrors _evict_cached_agent / idle-sweep.
                            _xproc_evicted_agent = _ev_agent
                    else:
                        agent = cached.agent
                        # Refresh LRU order so the cap enforcement evicts
                        # truly-oldest entries, not the one we just used.
                        try:
                            _cache.move_to_end(ctx.session_key)
                        except KeyError:
                            pass
                        self.agent_cache.init_for_turn(agent, ctx._interrupt_depth)
                        # Refresh agent max_iterations from current config
                        # (cached agent may have been created with old config)
                        agent.max_iterations = max_iterations
                        logger.debug("Reusing cached agent for session %s", ctx.session_key)
                        reused_cached_agent = True

        # Lock released — refresh the fallback chain from disk for the
        # reused agent OUTSIDE the cache lock (config.yaml read is disk
        # I/O; the idle-sweep watcher contends on this lock and stalls
        # Discord heartbeats — same reasoning as #52197).  A chain
        # configured after this agent was cached (or after gateway start)
        # must reach the next turn (#60955).  Per-session turn
        # serialization (_running_agents) keeps this safe post-lock.
        if reused_cached_agent and agent is not None:
            self._apply_fallback_chain_to_agent(
                agent, self._refresh_fallback_providers(),
            )

        # Lock released — now schedule cleanup of any cross-process-evicted
        # agent on a daemon thread so memory-provider shutdown / socket
        # teardown never blocks the gateway event loop or the cache lock
        # the session-expiry watcher needs (#52197).
        if _xproc_evicted_agent is not None:
            try:
                threading.Thread(
                    target=self.agent_cache.release_soft,
                    args=(_xproc_evicted_agent,),
                    daemon=True,
                    name=f"agent-xproc-evict-{str(ctx.session_key)[:24]}",
                ).start()
            except Exception:
                # Interpreter shutdown or thread-spawn failure — release
                # inline as a best-effort fallback.
                try:
                    self.agent_cache.release_soft(_xproc_evicted_agent)
                except Exception:
                    pass

        if agent is None:
            # Config changed or first message — create fresh agent
            agent = ctx.create_agent(
                model=turn_route["model"],
                **turn_route["runtime"],
                **_checkpoint_agent_kwargs(ctx.user_config),
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=ctx.enabled_toolsets,
                disabled_toolsets=ctx.disabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._prefill_messages or None,
                reasoning_config=reasoning_config,
                service_tier=self._service_tier,
                request_overrides=turn_route.get("request_overrides"),
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=ctx.session_id,
                platform=platform_key,
                user_id=ctx.source.user_id,
                user_id_alt=ctx.source.user_id_alt,
                user_name=ctx.source.user_name,
                chat_id=ctx.source.chat_id,
                chat_name=ctx.source.chat_name,
                chat_type=ctx.source.chat_type,
                thread_id=ctx.source.thread_id,
                gateway_session_key=ctx.session_key,
                session_db=getattr(self._session_db, "_db", self._session_db),
                # Reload from disk — do not reuse the startup snapshot (#60955).
                fallback_providers=self._refresh_fallback_providers(),
                skip_context_files=skip_context_files,
                # Keep the persona even with minimal context: soul identity is
                # a single small file, not part of the expensive walk.
                load_soul_identity=True,
            )
            with _cache_lock:
                    # Record the session_id the snapshot was taken for
                    # alongside the message_count, so the cross-process
                    # guard can skip the (meaningless) count comparison
                    # when the active session_id later switches under
                    # the same session_key (#54947).
                    _cache[ctx.session_key] = AgentCacheEntry(
                        agent=agent,
                        signature=_sig,
                        message_count=_current_msg_count,
                        session_id=ctx.session_id,
                    )
                    self.agent_cache.enforce_cap()
            logger.debug("Created new agent for session %s (sig=%s)", ctx.session_key, _sig)

        # Per-message state — callbacks and reasoning config change every
        # turn and must not be baked into the cached agent constructor.
        # Gate on needs_progress_queue (tool_progress OR thinking_progress)
        # rather than tool_progress alone: the progress_callback also relays
        # _thinking assistant scratch text, which is gated on
        # thinking_progress and is intentionally independent of tool
        # progress. With the old `tool_progress_enabled`-only gate, a user
        # who set thinking_progress:true but kept tool_progress:off got a
        # None callback — so _thinking scratch bubbles never relayed even
        # though the progress queue was created for them.
        agent.tool_progress_callback = (
            ctx.progress_callback
            if (
                ctx.needs_progress_queue
                or ctx.log_mode_enabled
                or ctx._live_status_adapter is not None
            )
            else None
        )
        agent.tool_start_callback = None
        agent.tool_complete_callback = None
        agent.step_callback = ctx._step_callback_sync if ctx._hooks_ref.loaded_hooks else None
        agent.stream_delta_callback = _stream_delta_cb
        agent.interim_assistant_callback = _interim_assistant_cb if _want_interim_messages else None
        agent.status_callback = ctx._status_callback_sync
        # Credits / out-of-band notices (usage bands, depletion, restored).
        # Messaging has no persistent status bar, so each notice is a
        # standalone push: render to a single plaintext line and deliver via
        # the shared _deliver_platform_notice rail (honors private/public +
        # thread metadata). Fires from the agent's sync worker thread, so we
        # hop onto the gateway loop with safe_schedule_threadsafe - same
        # pattern as _status_callback_sync. The fired-once latch lives on the
        # cached agent and persists across turns, so a band crosses -> one
        # push (no per-turn re-nag). Recovery ("✓ Credit access restored")
        # rides the same show path (it's emitted as a success notice, not a
        # clear). The clear callback is a no-op: a sent platform message
        # can't be cleanly retracted, and the band already fired once.
        def _notice_callback_sync(notice) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            try:
                line = render_notice_line(notice)
            except Exception:
                logger.debug("render_notice_line failed", exc_info=True)
                return
            if not line:
                return
            safe_schedule_threadsafe(
                self._deliver_platform_notice(ctx.source, line),
                ctx._loop_for_step,
                logger=logger,
                log_message="notice_callback delivery scheduling error",
            )

        agent.notice_callback = _notice_callback_sync
        agent.notice_clear_callback = None
        agent.event_callback = ctx._event_callback_sync
        agent.reasoning_config = reasoning_config
        agent.service_tier = self._service_tier
        agent.request_overrides = turn_route.get("request_overrides") or {}
        # Must-deliver notes for THIS turn ride the current user message
        # (api_content sidecar), never the system prompt: staged by
        # _handle_message_with_agent (auto-reset note, first-contact
        # intro, and other one-turn notices). Assigned unconditionally so a
        # reused cached agent never replays a stale note.
        agent._gateway_turn_context_notes = "\n\n".join(
            self.sessions.consume_sidecar_notes(ctx.session_key)
        )

        _bg_review_release = threading.Event()
        _bg_review_pending: list[str] = []
        _bg_review_pending_lock = threading.Lock()

        def _deliver_bg_review_message(message: str) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    message,
                    metadata=_non_conversational_metadata(ctx._status_thread_metadata, platform=ctx.source.platform),
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="background_review_callback scheduling error",
            )

        def _release_bg_review_messages() -> None:
            _bg_review_release.set()
            with _bg_review_pending_lock:
                pending = list(_bg_review_pending)
                _bg_review_pending.clear()
            for queued in pending:
                _deliver_bg_review_message(queued)

        # Background review delivery — send "💾 Memory updated" etc. to user
        def _bg_review_send(message: str) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            if not _bg_review_release.is_set():
                with _bg_review_pending_lock:
                    if not _bg_review_release.is_set():
                        _bg_review_pending.append(message)
                        return
            _deliver_bg_review_message(message)

        agent.background_review_callback = _bg_review_send
        # Register the release hook on the adapter so base.py's finally
        # block can fire it after delivering the main response.
        if ctx._status_adapter and ctx.session_key:
            if getattr(type(ctx._status_adapter), "register_post_delivery_callback", None) is not None:
                ctx._status_adapter.register_post_delivery_callback(
                    ctx.session_key,
                    _release_bg_review_messages,
                    generation=ctx.run_generation,
                )
            else:
                _pdc = getattr(ctx._status_adapter, "_post_delivery_callbacks", None)
                if _pdc is not None:
                    _pdc[ctx.session_key] = _release_bg_review_messages
        # Memory update notifications in chat.  Config: display.memory_notifications
        #   off     — no chat notification (still logged to stdout)
        #   on      — generic "💾 Memory updated" (default)
        #   verbose — content preview: "💾 Memory ➕ Hermes Repo..."
        _mem_notif = ctx.user_config.get("display", {}).get("memory_notifications")
        if isinstance(_mem_notif, bool):
            _mem_notif = "on" if _mem_notif else "off"
        agent.memory_notifications = str(_mem_notif).lower() if _mem_notif else "on"

        # ------------------------------------------------------------------
        # Clarify callback: present a clarify prompt and block on a response.
        #
        # Runs on the agent's worker thread (see clarify_tool's synchronous
        # callback contract).  Bridges sync→async by scheduling the
        # adapter's send_clarify on the gateway event loop, then blocks on
        # the clarify primitive's threading.Event with a configurable
        # timeout.  Returns the user's response string, or a sentinel
        # explaining that no response arrived (so the agent can adapt
        # rather than hang forever).
        # ------------------------------------------------------------------
        def _clarify_callback_sync(question: str, choices, multi_select: bool = False) -> str:
            from tools import clarify_gateway as _clarify_mod
            import uuid as _uuid

            if not ctx._status_adapter:
                return ""

            clarify_id = _uuid.uuid4().hex[:10]
            _clarify_mod.register(
                clarify_id=clarify_id,
                session_key=ctx.session_key or "",
                question=question,
                choices=list(choices) if choices else None,
                multi_select=bool(multi_select),
            )

            # Pause typing — like approval, we don't want a "thinking..."
            # status to obscure the prompt or block the user from typing
            # an "Other" response on platforms that disable input while
            # typing is active (Slack Assistant API).
            try:
                ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)
            except Exception:
                pass

            # Ordering barrier (#clarify-ordering): flush any buffered
            # assistant prose (interim commentary / streamed deltas) to the
            # platform BEFORE sending the poll.  The poll is delivered on a
            # separate, agent-thread-blocking path; without this barrier it
            # races ahead of prose still sitting in the stream consumer's
            # queue, so the question renders ABOVE its own explanation.
            # Best-effort + short timeout: never hang the agent thread if
            # the consumer task isn't running.
            try:
                _sc = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
                _flush = getattr(_sc, "flush_pending_sync", None)
                if callable(_flush):
                    _flush(timeout=3.0)
            except Exception:
                logger.debug(
                    "Stream-consumer flush before clarify prompt failed",
                    exc_info=True,
                )

            send_ok = False
            fut = safe_schedule_threadsafe(
                ctx._status_adapter.send_clarify(
                    chat_id=ctx._status_chat_id,
                    question=question,
                    choices=list(choices) if choices else None,
                    clarify_id=clarify_id,
                    session_key=ctx.session_key or "",
                    metadata=ctx._status_thread_metadata,
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="Clarify send failed to schedule",
            )
            if fut is None:
                send_ok = False
            else:
                try:
                    result = fut.result(timeout=15)
                    send_ok = bool(getattr(result, "success", False))
                except Exception as exc:
                    logger.warning("Clarify send failed: %s", exc)
                    send_ok = False

            if not send_ok:
                # Couldn't deliver the prompt — clean up and return
                # sentinel so the agent can fall back to a sensible
                # default rather than hanging.
                _clarify_mod.clear_session(ctx.session_key or "")
                return "[clarify prompt could not be delivered]"

            timeout = _clarify_mod.get_clarify_timeout()
            response = _clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
            if response is None or response == "":
                # Timeout or session-boundary cancellation
                return f"[user did not respond within {int(timeout / 60)}m]"
            return response

        agent.clarify_callback = _clarify_callback_sync

        # Show assistant thinking between tool calls — independent of
        # tool_progress mode. Mattermost needs an explicit per-platform
        # opt-in so global scratch-text display does not leak into threads.
        agent.thinking_progress = ctx._thinking_enabled
        # Store agent reference for interrupt support
        ctx.agent_holder[0] = agent
        # Wire the platform thread-rename lane onto the agent, because the
        # session titler now fires from the turn prologue rather than after
        # the response. Titles are pushed here the moment they land.
        self._attach_session_title_callback(agent, ctx)
        # Publish turn ownership for explicit /stop, /new, disconnect, and
        # shutdown interrupts. Older session processes are outside this
        # baseline and remain alive.
        agent._gateway_turn_process_task_id = ctx.process_task_id
        agent._gateway_turn_process_baseline = ctx.process_baseline
        # Capture the full tool definitions for transcript logging
        ctx.tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None

        # Convert history to agent format.
        # Two cases:
        #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
        #      - Strip timestamps, keep role+content
        #   2. Interrupt path (from agent result["messages"]): full agent messages
        #      that may include tool_calls, tool_call_id, reasoning, etc.
        #      - These must be passed through intact so the API sees valid
        #        assistant→tool sequences (dropping tool_calls causes 500 errors)
        #
        # Telegram observed group context is handled structurally here:
        # observed=True transcript rows are withheld from replayable
        # history and attached to the current addressed message as
        # API-only context, so persisted history stores only the real
        # addressed user turn.
        agent_history, observed_group_context = _build_gateway_agent_history(
            ctx.history,
            channel_prompt=ctx.channel_prompt,
            inject_timestamps=_message_timestamps_enabled(ctx.user_config),
        )

        # FTS write-corruption guard (#50502): when message persistence
        # fails silently through corrupt FTS triggers, the reloaded
        # transcript above is stale/empty even though the SAME cached agent
        # still holds the full live conversation in `_session_messages`.
        # Replacing the live transcript with that shorter copy causes
        # immediate same-session amnesia. Only applies when we reused a
        # cached agent bound to this exact session_id.
        if reused_cached_agent and getattr(agent, "session_id", None) == ctx.session_id:
            _selected = _select_cached_agent_history(
                agent_history, getattr(agent, "_session_messages", None)
            )
            if _selected is not agent_history:
                logger.warning(
                    "Persisted transcript lagged live cached history for "
                    "session %s (disk=%d, memory=%d); preserving live "
                    "conversation context (possible FTS write corruption)",
                    ctx.session_key, len(agent_history), len(_selected),
                )
                # The live in-memory history bypassed the
                # _build_gateway_agent_history cleanup pipeline above —
                # re-apply the stale-confirmation expiry (#59607) so a
                # dangerous confirmation can't slip through this path
                # either. Idempotent; messages without timestamps are
                # untouched.
                agent_history = strip_stale_dangerous_confirmations(
                    _selected, now=time.time()
                )

        # Collect MEDIA paths already in history so we can exclude them
        # from the current turn's extraction. This is compression-safe:
        # even if the message list shrinks, we know which paths are old.
        _history_media_paths: set = _collect_history_media_paths(agent_history)

        # Register per-session gateway approval callback so dangerous
        # command approval blocks the agent thread (mirrors CLI input()).
        # The callback bridges sync→async to send the approval request
        # to the user immediately.
        from tools.approval import (
            register_gateway_notify,
            reset_current_session_key,
            set_current_session_key,
            unregister_gateway_notify,
        )

        def _approval_notify_sync(approval_data: dict) -> None:
            """Send the approval request to the user from the agent thread.

                If the adapter supports interactive button-based approvals
                (e.g. Discord's ``send_exec_approval``), use that for a richer
                UX.  Otherwise fall back to a plain text message with
                ``/approve`` instructions.
                """
            # Pause the typing indicator while the agent waits for
            # user approval.  Critical for Slack's Assistant API where
            # assistant_threads_setStatus disables the compose box — the
            # user literally cannot type /approve while "is thinking..."
            # is active.  The approval message send auto-clears the Slack
            # status; pausing prevents _keep_typing from re-setting it.
            # Typing resumes in _handle_approve_command/_handle_deny_command.
            ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)

            cmd = approval_data.get("command", "")
            desc = approval_data.get("description", "dangerous command")

            # Redact credentials from the command before displaying it in
            # the approval prompt — Tirith's findings are already redacted,
            # but the raw command string still leaks secrets to the chat
            # platform (#48456). Applied here so BOTH the button-based
            # (send_exec_approval) and plain-text fallback paths below use
            # the redacted value.
            cmd = _redact_approval_command(cmd)

            # Prefer button-based approval when the adapter supports it.
            # Check the *class* for the method, not the instance — avoids
            # false positives from MagicMock auto-attribute creation in tests.
            if getattr(type(ctx._status_adapter), "send_exec_approval", None) is not None:
                try:
                    _approval_fut = safe_schedule_threadsafe(
                        ctx._status_adapter.send_exec_approval(
                            chat_id=ctx._status_chat_id,
                            command=cmd,
                            session_key=_approval_session_key,
                            description=desc,
                            metadata=ctx._status_thread_metadata,
                            allow_permanent=approval_data.get("allow_permanent", True),
                            allow_session=approval_data.get("allow_session", True),
                            smart_denied=approval_data.get("smart_denied", False),
                        ),
                        ctx._loop_for_step,
                        logger=logger,
                        log_message="send_exec_approval scheduling error",
                    )
                    if _approval_fut is None:
                        raise RuntimeError("send_exec_approval: loop unavailable")
                    _approval_result = _approval_fut.result(timeout=15)
                    if _approval_result.success:
                        return
                    logger.warning(
                        "Button-based approval failed (send returned error), falling back to text: %s",
                        _approval_result.error,
                    )
                except Exception as _e:
                    logger.warning(
                        "Button-based approval failed, falling back to text: %s", _e
                    )

            # Fallback: plain text approval prompt.  Use the adapter's
            # typed prefix so Slack/Matrix users are told the form they
            # can actually type (`!approve`) — typed "/" is blocked in
            # Slack threads and reserved by Matrix clients.
            _p = getattr(ctx._status_adapter, "typed_command_prefix", "/")
            msg = _format_exec_approval_fallback(
                cmd,
                desc,
                _p,
                allow_permanent=approval_data.get("allow_permanent", True),
                allow_session=approval_data.get("allow_session", True),
                smart_denied=approval_data.get("smart_denied", False),
            )
            try:
                _approval_send_fut = safe_schedule_threadsafe(
                    ctx._status_adapter.send(
                        ctx._status_chat_id,
                        msg,
                        metadata=ctx._status_thread_metadata,
                    ),
                    ctx._loop_for_step,
                    logger=logger,
                    log_message="Approval text-send scheduling error",
                )
                if _approval_send_fut is not None:
                    _approval_send_fut.result(timeout=15)
            except Exception as _e:
                logger.error("Failed to send approval request: %s", _e)

        # Keep real user text separate from API-only recovery guidance.  If
        # an auto-continue note is prepended below, persist the original
        # message so stale guidance never replays as user-authored text.
        _persist_user_message_override: Optional[Any] = ctx.persist_user_message
        _persist_user_timestamp_override: Optional[float] = ctx.persist_user_timestamp

        # Prepend pending model switch note so the model knows about the switch
        _conversation = self.sessions.state(ctx.session_key).conversation if ctx.session_key else None
        _msn = _conversation.model_switch_note if _conversation else None
        if _conversation is not None:
            _conversation.model_switch_note = None
        if _msn:
            ctx.message = _msn + "\n\n" + ctx.message

        # Auto-continue: if the loaded history ends with a tool result,
        # the previous agent turn was interrupted mid-work (gateway
        # restart, crash, SIGTERM).  Prepend a system note so the model
        # finishes processing the pending tool results before addressing
        # the user's new message.  (#4493)
        #
        # Session-level resume_pending (set on drain-timeout shutdown)
        # escalates the wording — the transcript's last role may be
        # anything (tool, assistant with unfinished work, etc.), so we
        # give a stronger, reason-aware instruction that subsumes the
        # tool-tail case.
        #
        # Freshness gate (#16802): both branches are gated on the age
        # of the last persisted transcript row.  That is the correct
        # "when did we last do anything here" signal for both the
        # resume_pending path (restart watchdog) and the tool-tail
        # path (in-flight tool loop killed).  We read ``history[-1]``
        # here because ``agent_history`` has already stripped the
        # ``timestamp`` field off tool/tool_call rows for API purity
        # (see the `k != "timestamp"` filter above).  Rows without a
        # timestamp (legacy transcripts) are treated as fresh so the
        # historical auto-continue behaviour is preserved.
        _freshness_window = auto_continue_freshness_window()
        _interruption_is_fresh = _is_fresh_gateway_interruption(
            _last_transcript_timestamp(ctx.history),
            window_secs=_freshness_window,
        )

        _resume_entry = None
        if ctx.session_key:
            try:
                _resume_entry = self.session_store._entries.get(ctx.session_key)
            except Exception:
                _resume_entry = None

        # resume_pending freshness uses a SECOND signal in addition to the
        # transcript clock above.  The restart watchdog stamps the session
        # with ``last_resume_marked_at`` at interrupt time — that is the
        # correct "when were we interrupted" signal.  The transcript clock
        # (_interruption_is_fresh) can be far older: an active thread you
        # return to may have its last persisted row hours back, even though
        # the interruption itself just happened.  Gating resume_pending on
        # the transcript clock alone makes the recovery note silently drop,
        # and because the startup auto-resume turn carries empty text
        # (_schedule_resume_pending_sessions), the model then receives a
        # blank user message and replies with confused "the message came
        # through blank" noise.  Treat the marker as fresh when
        # EITHER signal is fresh so the two freshness checks agree.
        _resume_mark_is_fresh = False
        if _resume_entry is not None and getattr(_resume_entry, "resume_pending", False):
            _resume_mark_is_fresh = _is_fresh_gateway_interruption(
                getattr(_resume_entry, "last_resume_marked_at", None),
                window_secs=_freshness_window,
            )
        _is_resume_pending = bool(
            _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
            and (_interruption_is_fresh or _resume_mark_is_fresh)
        )
        _has_fresh_tool_tail = bool(
            agent_history
            and agent_history[-1].get("role") == "tool"
            and _interruption_is_fresh
        )

        if _is_resume_pending:
            _reason = getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            _persist_user_message_override = ctx.message
            # The empty-message case is the auto-resume startup turn
            # synthesized by _schedule_resume_pending_sessions — there is
            # no NEW user message to address.  Guidance is adapter-aware:
            # interactive platforms report the restore and ask what next;
            # non-interactive event platforms (webhook, API server)
            # continue the interrupted work instead, because nobody is
            # present to answer and an acknowledgement would silently
            # abandon the task (#57056).
            _resume_adapter = self._adapter_for_source(ctx.source)
            _interactive_resume = bool(
                getattr(_resume_adapter, "interactive_resume", True)
            )
            ctx.message = build_resume_recovery_note(
                _reason, ctx.message, interactive=_interactive_resume,
            )
        elif _has_fresh_tool_tail:
            _persist_user_message_override = ctx.message
            ctx.message = (
                "[System note: A new message has arrived. The conversation "
                "history contains pending tool outputs from an interrupted turn. "
                "IGNORE those pending results. Address the user's NEW message "
                "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                + ctx.message
            )

        # Consume one-shot /reload-skills note (if the user ran
        # /reload-skills since their last turn in this session). Same
        # queue pattern as CLI: prepend to the NEXT user message, then
        # clear. Nothing was written to the transcript out-of-band, so
        # message alternation stays intact.
        _srn = _conversation.skills_reload_note if _conversation else None
        if _conversation is not None:
            _conversation.skills_reload_note = None
        if _srn:
            ctx.message = _srn + "\n\n" + ctx.message

        # Safety net: a startup auto-resume event carries empty
        # text and relies on the resume_pending branch above to supply the
        # recovery note.  If that branch did not fire for any reason (e.g.
        # both freshness signals disagreed, or the marker was cleared
        # between scheduling and dispatch) we must NOT hand the model a
        # blank user turn — it responds with confused "the message came
        # through blank" noise.  Restricted to resume_pending sessions so
        # legitimately empty user turns (e.g. an image with no caption,
        # wrapped as native content below) are untouched.
        if (
            isinstance(ctx.message, str)
            and not ctx.message.strip()
            and _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
        ):
            _sn_reason = (
                getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            )
            _sn_adapter = self._adapter_for_source(ctx.source)
            ctx.message = build_resume_recovery_note(
                _sn_reason,
                "",
                interactive=bool(
                    getattr(_sn_adapter, "interactive_resume", True)
                ),
            )

        _approval_session_key = ctx.session_key or ""
        _approval_session_token = set_current_session_key(_approval_session_key)
        register_gateway_notify(_approval_session_key, _approval_notify_sync)
        try:
            # If _prepare_inbound_message_text buffered image paths for native
            # attachment, wrap the user turn as an OpenAI-style multimodal
            # content list. Consume-and-clear so subsequent turns on the same
            # runner instance don't re-attach stale images.
            _native_imgs = self._consume_pending_native_image_paths(ctx.session_key)
            if _native_imgs:
                try:
                    from agent.image_routing import build_native_content_parts
                    _parts, _skipped = build_native_content_parts(
                        ctx.message,
                        _native_imgs,
                    )
                    if _skipped:
                        logger.warning(
                            "Native image attachment: skipped %d unreadable path(s): %s",
                            len(_skipped), _skipped,
                        )
                    if any(p.get("type") == "image_url" for p in _parts):
                        _run_message: Any = _parts
                    else:
                        # All images failed to read — fall back to plain text.
                        _run_message = ctx.message
                except Exception as _img_exc:
                    logger.warning(
                        "Native image attachment failed, falling back to text: %s",
                        _img_exc,
                    )
                    _run_message = ctx.message
            else:
                _run_message = ctx.message

            _api_run_message = _wrap_current_message_with_observed_context(
                _run_message,
                observed_group_context,
            )
            _conversation_kwargs = {
                "conversation_history": agent_history,
                "task_id": ctx.session_id,
            }
            if _persist_user_message_override is not None:
                _conversation_kwargs["persist_user_message"] = _persist_user_message_override
            elif observed_group_context:
                _conversation_kwargs["persist_user_message"] = ctx.message
            if ctx.persist_user_display_kind:
                # Internal self-injected turn (#82888): type the persisted user
                # row at turn start so UIs render it as a timeline notice, not
                # a user bubble. Role/content are untouched and the key is
                # stripped from provider-bound payloads in conversation_loop.
                _conversation_kwargs["persist_user_display_kind"] = (
                    ctx.persist_user_display_kind
                )
            if ctx.moa_config is not None:
                _conversation_kwargs["moa_config"] = ctx.moa_config
            if _persist_user_timestamp_override is not None:
                _conversation_kwargs["persist_user_timestamp"] = _persist_user_timestamp_override
            result = lifecycle.run_conversation(agent, _api_run_message, **_conversation_kwargs)
        finally:
            unregister_gateway_notify(_approval_session_key)
            # Cancel any pending clarify entries so blocked agent
            # threads don't hang past the end of the run (interrupt,
            # completion, gateway shutdown).  Idempotent.
            try:
                from tools.clarify_gateway import clear_session as _clear_clarify_session
                _clear_clarify_session(_approval_session_key)
            except Exception:
                pass
            reset_current_session_key(_approval_session_token)
        ctx.result_holder[0] = result

        # Signal the stream consumer that the agent is done
        if _stream_consumer is not None:
            _stream_consumer.finish()

        # Signal the streaming-TTS consumer that the agent is done (#60671).
        # finish() is called from the outer event-loop thread after the
        # executor returns, so early returns from run_sync are also
        # finalised.  See the outer finally/completion section below.

        # Return final response, or a message if something went wrong
        final_response = result.get("final_response")

        # Extract actual token counts from the agent instance used for this run
        _last_prompt_toks = 0
        _input_toks = 0
        _output_toks = 0
        _context_length = 0
        _agent = ctx.agent_holder[0]
        if _agent and hasattr(_agent, "context_compressor"):
            _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
            _input_toks = getattr(_agent, "session_prompt_tokens", 0)
            _output_toks = getattr(_agent, "session_completion_tokens", 0)
            _context_length = getattr(_agent.context_compressor, "context_length", 0) or 0
        _resolved_model = getattr(_agent, "model", None) if _agent else None

        # Sync session_id immediately after run_conversation(). Compression
        # can rotate before a follow-up model call fails; the failure return
        # below must still point the gateway at the compressed child.
        agent = ctx.agent_holder[0]
        _session_was_split = False
        # In-place compaction (compression.in_place / #38763) compacts the
        # transcript WITHOUT rotating the id, so the id-change diff below
        # can't detect it. compress_context() sets this rotation-independent
        # flag on the agent; the gateway uses it to re-baseline transcript
        # handling (history_offset=0 + rewrite the JSONL transcript) the
        # same way a split would, even though the session_id is unchanged.
        _compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
        agent_session_id = getattr(agent, 'session_id', ctx.session_id) if agent else ctx.session_id
        if agent and ctx.session_key and agent_session_id != ctx.session_id:
            _session_was_split = True
            logger.info(
                "Session split detected: %s → %s (compression)",
                ctx.session_id, agent_session_id,
            )
            entry = self.session_store._entries.get(ctx.session_key)
            _session_split_entry_persisted = False
            if entry:
                entry_session_id = getattr(entry, "session_id", None)
                if not ctx._run_still_current():
                    logger.info(
                        "Skipping session split sync for stale run %s — "
                        "generation %s is no longer current",
                        ctx.session_key or "?",
                        ctx.run_generation,
                    )
                elif entry_session_id == agent_session_id:
                    _session_split_entry_persisted = True
                elif entry_session_id != ctx.session_id:
                    logger.info(
                        "Skipping session split sync for %s because the "
                        "session binding moved from %s to %s before "
                        "compression finished",
                        ctx.session_key or "?",
                        ctx.session_id,
                        entry_session_id,
                    )
                else:
                    entry.session_id = agent_session_id
                    self.session_store._save()
                    self.session_store._record_gateway_session_peer(
                        agent_session_id,
                        ctx.session_key,
                        ctx.source,
                    )
                    _session_split_entry_persisted = True

            # If this is a Telegram DM and source.thread_id was lost during
            # the session split (synthetic / recovered event), restore it
            # from the binding so _thread_metadata_for_source produces the
            # correct message_thread_id instead of routing to the General
            # thread.  Failure here is non-fatal — we log and continue;
            # worst case the message lands in General, which is the
            # pre-fix behaviour. Only do this after this run successfully
            # published its session split; a stale /stop→/new predecessor
            # must not mutate routing/binding state for the fresh session.
            if _session_split_entry_persisted and (
                getattr(ctx.source, "platform", None) == Platform.TELEGRAM
                and getattr(ctx.source, "chat_type", None) == "dm"
                and getattr(ctx.source, "thread_id", None) is None
                and self._session_db is not None
            ):
                try:
                    # run_sync is off-loop (executor); sync DB is fine.
                    _binding = self._session_db._db.get_telegram_topic_binding_by_session(
                        session_id=agent_session_id,
                    )
                    if _binding and _binding.get("thread_id"):
                        ctx.source.thread_id = str(_binding["thread_id"])
                        logger.debug(
                            "Restored source.thread_id=%s from binding after session split %s → %s",
                            ctx.source.thread_id,
                            ctx.session_id,
                            agent_session_id,
                        )
                except Exception:
                    logger.debug(
                        "Failed to restore thread_id from binding after session split",
                        exc_info=True,
                    )
            if _session_split_entry_persisted:
                self._sync_telegram_topic_binding(
                    ctx.source, entry, reason="agent-run-compression",
                )

        effective_session_id = agent_session_id
        self._sync_session_model_from_agent(effective_session_id, agent)
        # history_offset=0 whenever the agent's message list no longer has
        # the original history prefix — i.e. on rotation (split) OR in-place
        # compaction. In both cases the returned `messages` is the compacted
        # set, so the gateway must persist all of it (offset 0), not slice
        # past the pre-compaction length (which would drop everything).
        _effective_history_offset = (
            0 if (_session_was_split or _compacted_in_place) else len(agent_history)
        )

        if not final_response:
            final_response = _normalize_empty_agent_response(
                result, final_response or "", history_len=len(agent_history),
            )
            final_response = _sanitize_gateway_final_response(ctx.source.platform, final_response)
            if not final_response:
                final_response = f"⚠️ {result['error']}" if result.get("error") else ""
            return {
                "final_response": final_response,
                "messages": result.get("messages", []),
                "api_calls": result.get("api_calls", 0),
                "failed": result.get("failed", False),
                # Sibling of the non-empty-response return below (#64686):
                # the classifier's failure_reason must survive the
                # empty-response normalization path too, or downstream
                # consumers (TUI billing surface, transient-failure
                # persistence) lose the structured reason exactly when
                # the run produced no text.
                "failure_reason": result.get("failure_reason"),
                "partial": result.get("partial", False),
                "completed": result.get("completed"),
                "interrupted": result.get("interrupted", False),
                "interrupt_message": result.get("interrupt_message"),
                "error": result.get("error"),
                "compression_exhausted": result.get("compression_exhausted", False),
                "compression_deferred": result.get("compression_deferred", False),
                "tools": ctx.tools_holder[0] or [],
                "history_offset": _effective_history_offset,
                "compacted_in_place": _compacted_in_place,
                "session_id": effective_session_id,
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "context_length": _context_length,
            }

        # Scan tool results for MEDIA:<path> tags that need to be delivered
        # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
        # in its JSON response, but the model's final text reply usually
        # doesn't include them.  We collect unique tags from tool results and
        # append any that aren't already present in the final response, so the
        # adapter's extract_media() can find and deliver the files exactly once.
        #
        # Scope the scan to THIS turn's tool results only. ``agent_history``
        # was passed into run_conversation as ``conversation_history``, so the
        # agent's returned ``messages`` list is ``agent_history`` followed by
        # the messages produced this turn. Slicing at ``len(agent_history)``
        # isolates the current turn precisely, so a stale MEDIA: path emitted
        # by a tool several turns earlier (still present in the full message
        # list) can never leak onto a later text-only reply. (Fixes #34608)
        #
        # Path-based deduplication against _history_media_paths (collected
        # before run_conversation) is retained as a secondary guard. It is
        # also the sole guard on the fallback branch taken when mid-run
        # context compression shrinks the message list below the original
        # history length, preserving the compression-safe behaviour of #160.
        if "MEDIA:" not in final_response:
            media_tags, has_voice_directive = _collect_auto_append_media_tags(
                result.get("messages", []),
                history_offset=len(agent_history),
                history_media_paths=_history_media_paths,
            )

            if media_tags:
                seen = set()
                unique_tags = []
                for tag in media_tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                if has_voice_directive:
                    unique_tags.insert(0, "[[audio_as_voice]]")
                final_response = final_response + "\n" + "\n".join(unique_tags)

        # Auto-titling runs at TURN START (agent/turn_context.py) from the
        # user's message alone, so it no longer waits on final_response — a
        # failed or interrupted turn still gets a titled session. The
        # platform-specific thread-rename callbacks are attached to the agent
        # as `_on_session_title` before the run starts (see
        # _attach_session_title_callback), because the titler now fires from
        # inside the turn prologue rather than from here.

        return {
            "final_response": final_response,
            "last_reasoning": result.get("last_reasoning"),
            "messages": ctx.result_holder[0].get("messages", []) if ctx.result_holder[0] else [],
            "api_calls": ctx.result_holder[0].get("api_calls", 0) if ctx.result_holder[0] else 0,
            "failed": ctx.result_holder[0].get("failed", False) if ctx.result_holder[0] else False,
            "failure_reason": (
                ctx.result_holder[0].get("failure_reason") if ctx.result_holder[0] else None
            ),
            "completed": ctx.result_holder[0].get("completed") if ctx.result_holder[0] else None,
            "interrupted": ctx.result_holder[0].get("interrupted", False) if ctx.result_holder[0] else False,
            "partial": ctx.result_holder[0].get("partial", False) if ctx.result_holder[0] else False,
            "error": ctx.result_holder[0].get("error") if ctx.result_holder[0] else None,
            "interrupt_message": ctx.result_holder[0].get("interrupt_message") if ctx.result_holder[0] else None,
            "compression_exhausted": (
                ctx.result_holder[0].get("compression_exhausted", False)
                if ctx.result_holder[0] else False
            ),
            # Soft lock-contention defer (#69870 consumer): distinct from
            # compression_exhausted so the gateway never auto-resets a
            # session that a concurrent compressor is about to shrink.
            "compression_deferred": (
                ctx.result_holder[0].get("compression_deferred", False)
                if ctx.result_holder[0] else False
            ),
            "tools": ctx.tools_holder[0] or [],
            "history_offset": _effective_history_offset,
            "compacted_in_place": _compacted_in_place,
            "last_prompt_tokens": _last_prompt_toks,
            "input_tokens": _input_toks,
            "output_tokens": _output_toks,
            "model": _resolved_model,
            "context_length": _context_length,
            "session_id": effective_session_id,
            "response_previewed": result.get("response_previewed", False),
            "response_transformed": result.get("response_transformed", False),
            # Pass through the agent_persisted flag so the persistence block
            # above can correctly determine whether the codex app-server path
            # self-persisted (it didn't — see codex_runtime.py).  Default
            # True preserves the skip-db behaviour for the standard runtime.
            "agent_persisted": (ctx.result_holder[0].get("agent_persisted", True) if ctx.result_holder[0] else True),
        }
