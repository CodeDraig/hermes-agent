"""Background-process and async-delegation notification ownership."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from gateway.config import Platform
from gateway.delivery import resolve_delivery_transport
from gateway.message_router import USER_BOUNDARY_END_REASONS as _USER_BOUNDARY_END_REASONS
from gateway.notices import non_conversational_metadata as _non_conversational_metadata
from gateway.platforms.base import MessageEvent, MessageType
from gateway.response_filters import _redact_gateway_user_facing_secrets

logger = logging.getLogger("gateway.run")


def _shorten_command_for_display(command: str, limit: int = 80) -> str:
    """Collapse a shell command onto one line and cap it for display."""
    one_line = " ".join((command or "").split())
    return one_line[: limit - 1] + "…" if len(one_line) > limit else one_line


def _parse_session_key(session_key: str) -> "dict | None":
    """Parse a session key into its component parts.

    Session keys follow the format
    ``agent:main:{platform}:{chat_type}:{chat_id}[:{extra}...]``.
    Returns a dict with ``platform``, ``chat_type``, ``chat_id``, and
    optionally ``thread_id`` keys, or None if the key doesn't match.

    The 6th element is only returned as ``thread_id`` for chat types where
    it is unambiguous (``dm`` and ``thread``).  For group/channel sessions
    the suffix may be a user_id (per-user isolation) rather than a
    thread_id, so we leave ``thread_id`` out to avoid mis-routing.
    """
    parts = session_key.split(":")
    if len(parts) >= 5 and parts[0] == "agent" and parts[1] == "main":
        result = {
            "platform": parts[2],
            "chat_type": parts[3],
            "chat_id": parts[4],
        }
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None

def _format_concise_process_notification(
    session_id: str,
    command: str,
    exit_code,
    output: str,
    duration_seconds=None,
) -> str:
    """One-line "pretty" completion message for the ``concise`` display mode.

    Success is a single status line; failure appends a short tail of output so
    the user can see what went wrong without the full raw dump. The full
    output always remains available to the agent via process(log/wait).
    """
    ok = exit_code in {0, None}
    icon = "✅" if ok else "❌"
    verb = "finished" if ok else f"failed (exit {exit_code})"
    parts = [f"{icon} Background task {verb}"]
    short_cmd = _shorten_command_for_display(command)
    if short_cmd:
        parts.append(f"— `{short_cmd}`")
    if isinstance(duration_seconds, (int, float)) and duration_seconds >= 0:
        secs = int(duration_seconds)
        if secs >= 3600:
            dur = f"{secs // 3600}h {(secs % 3600) // 60}m"
        elif secs >= 60:
            dur = f"{secs // 60}m {secs % 60}s"
        else:
            dur = f"{secs}s"
        parts.append(f"({dur})")
    text = " ".join(parts)
    if not ok and output:
        tail_lines = [ln for ln in output.strip().splitlines() if ln.strip()][-5:]
        tail = "\n".join(tail_lines)
        if len(tail) > 500:
            tail = tail[-500:]
        if tail:
            text += f"\n```\n{tail}\n```"
    return text

def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a watch pattern event from completion_queue into a [IMPORTANT:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    if evt_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"

    # Overflow events carry their human-readable summary in `message`,
    # like watch_disabled — see the shared formatter in
    # tools/process_registry.py.
    if evt_type in ("watch_overflow_tripped", "watch_overflow_released"):
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\n"
            f"Matched output:\n{_out}"
        )
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        # Reuse the shared rich formatter (self-contained task-source block).
        from tools.process_registry import format_process_notification
        return format_process_notification(evt)

    return None

def _drain_gateway_watch_events(completion_queue) -> "list[dict]":
    """Drain gateway-owned watch events without spinning on requeued events.

    Watch events are handled by the post-turn gateway drain. Process
    completions are owned by their per-process watcher task, and async
    delegation completions are owned by ``_async_delegation_watcher``.
    Requeueing async events inside ``while not queue.empty()`` would make the
    loop non-terminating, so detach the current batch first, then requeue any
    events this drain does not own after the queue is empty.
    """
    watch_events: list[dict] = []
    requeue: list[dict] = []
    while not completion_queue.empty():
        try:
            evt = completion_queue.get_nowait()
        except Exception:
            break
        evt_type = evt.get("type", "completion")
        if evt_type in {
            "watch_match",
            "watch_disabled",
            "watch_overflow_tripped",
            "watch_overflow_released",
        }:
            watch_events.append(evt)
        elif evt_type == "async_delegation":
            requeue.append(evt)
        # else: process completion events are handled by the watcher task
    for evt in requeue:
        completion_queue.put(evt)
    return watch_events

class ProcessNotifications:
    def _build_process_event_source(self, evt: dict):
        """Resolve the canonical source for a synthetic background-process event.

        Prefer the persisted session-store origin for the event's session key.
        Falling back to the currently active foreground event is what causes
        cross-topic bleed, so don't do that.
        """
        from gateway.session import SessionSource

        session_key = str(evt.get("session_key") or "").strip()
        derived_platform = ""
        derived_chat_type = ""
        derived_chat_id = ""

        if session_key:
            try:
                self.session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                if entry and getattr(entry, "origin", None):
                    return entry.origin
            except Exception as exc:
                logger.debug(
                    "Synthetic process-event session-store lookup failed for %s: %s",
                    session_key,
                    exc,
                )

            cached_source = self._get_cached_session_source(session_key)
            if cached_source is not None:
                return cached_source

            _parsed = _parse_session_key(session_key)
            if _parsed:
                derived_platform = _parsed["platform"]
                derived_chat_type = _parsed["chat_type"]
                derived_chat_id = _parsed["chat_id"]

        platform_name = str(evt.get("platform") or derived_platform or "").strip().lower()
        chat_type = str(evt.get("chat_type") or derived_chat_type or "").strip().lower()
        chat_id = str(evt.get("chat_id") or derived_chat_id or "").strip()
        if not platform_name or not chat_type or not chat_id:
            logger.warning(
                "Synthetic event source unresolvable: "
                "session_key=%r platform=%r chat_type=%r chat_id=%r "
                "evt_type=%s",
                session_key, platform_name, chat_type, chat_id,
                evt.get("type", "?"),
            )
            return None

        try:
            platform = Platform(platform_name)
        except Exception:
            logger.warning(
                "Synthetic process event has invalid platform metadata: %r",
                platform_name,
            )
            return None

        scope_id = str(evt.get("scope_id") or "").strip() or None
        if platform == Platform.MATTERMOST and scope_id is None and chat_type not in ("dm", "thread"):
            logger.warning(
                "Synthetic event source for %s chat=%s (%s) reconstructed "
                "without a Mattermost team scope; team-scoped profile routing "
                "cannot be applied.",
                platform_name, chat_id, chat_type,
            )
        return SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=str(evt.get("thread_id") or "").strip() or None,
            user_id=str(evt.get("user_id") or "").strip() or None,
            user_name=str(evt.get("user_name") or "").strip() or None,
            scope_id=scope_id,
        )

    async def _drain_watch_notifications(self, completion_queue) -> None:
        """Consume queued watch events and inject them when notifications are enabled.

        The queue is ALWAYS drained (so watch events don't rot or requeue-spin)
        but injection is skipped entirely when
        ``display.background_process_notifications`` is ``off`` (#9290).
        """
        watch_events = _drain_gateway_watch_events(completion_queue)
        if self._load_background_notifications_mode() == "off":
            return

        for evt in watch_events:
            synth_text = _format_gateway_process_notification(evt)
            if not synth_text:
                continue
            try:
                await self._inject_watch_notification(synth_text, evt)
            except Exception as exc:
                logger.error("Watch notification injection error: %s", exc)

    async def _inject_watch_notification(
        self, synth_text: str, evt: dict,
    ) -> Optional[bool]:
        """Inject a watch/completion notification as a synthetic message event.

        Routing must come from the queued event itself, not from whatever
        foreground message happened to be active when the queue was drained.
        Returns ``True`` after adapter acceptance, ``False`` after a retryable
        adapter failure, and ``None`` when the event has no gateway route. This
        is not a transactional boundary: a process crash after adapter
        acceptance can still cause durable at-least-once replay.
        """
        source = await asyncio.to_thread(self._build_process_event_source, evt)
        if not source:
            # API-server-originated sessions bind a RAW session key (the
            # X-Hermes-Session-Id value — see _bind_api_server_session), not a
            # structured ``agent:main:...`` key, so _build_process_event_source
            # cannot derive routing metadata from it and returns None above.
            # Recover the raw session id and wake the real session via the API
            # server's own /v1/chat/completions entry point instead of
            # dropping the event.
            raw_sid = str(evt.get("origin_session_id") or "").strip()
            if not raw_sid:
                _sk = str(evt.get("session_key") or "").strip()
                if _sk and _parse_session_key(_sk) is None:
                    raw_sid = _sk
            if raw_sid:
                adapter = self.adapters.get(Platform.API_SERVER)
                from gateway.wake import adapter_supports_push, deliver_wake
                if adapter is not None and not adapter_supports_push(adapter):
                    try:
                        logger.info(
                            "Watch pattern notification — waking api_server "
                            "session %s via self-post",
                            raw_sid,
                        )
                        await deliver_wake(adapter, text=synth_text, session_id=raw_sid)
                        return True
                    except Exception as e:
                        logger.warning(
                            "Watch notification self-post wake failed for "
                            "session %s: %s",
                            raw_sid, e,
                        )
                        return False
                logger.warning(
                    "Dropping watch notification for raw session %s: no "
                    "api_server adapter to self-post through",
                    raw_sid,
                )
                return None
            logger.warning(
                "Dropping watch notification with no routing metadata for process %s",
                evt.get("session_id", "unknown"),
            )
            return None
        platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        # Resolve through the shared native transport resolver.
        adapter = None
        try:
            _platform_enum = Platform(platform_name)
        except (ValueError, KeyError):
            _platform_enum = None
        if _platform_enum is not None:
            try:
                _transport = resolve_delivery_transport(
                    _platform_enum, self.config, self.adapters,
                )
            except Exception:
                _transport = None
            if _transport is not None:
                adapter = _transport.adapter
        if adapter is None:
            # Legacy literal scan — still correct for native adapters, and
            # keeps minimal runner stubs (tests) and exotic platform strings
            # working when the resolver can't run.
            for p, a in self.adapters.items():
                if p.value == platform_name:
                    adapter = a
                    break
        if not adapter:
            return None
        from gateway.wake import adapter_supports_push as _wake_push_ok
        if not _wake_push_ok(adapter):
            # Non-push adapter (api_server) resolved WITH routing metadata:
            # its chat_id is the raw session id (see _bind_api_server_session,
            # which binds chat_id = session_id). handle_message would run the
            # wake under a build_session_key()-derived key that never matches
            # the raw X-Hermes-Session-Id session — self-post instead.
            from gateway.wake import deliver_wake
            raw_sid = str(evt.get("origin_session_id") or "").strip() or str(source.chat_id or "")
            try:
                logger.info(
                    "Watch pattern notification — waking api_server session "
                    "%s via self-post",
                    raw_sid,
                )
                await deliver_wake(adapter, text=synth_text, session_id=raw_sid)
                return True
            except Exception as e:
                logger.warning(
                    "Watch notification self-post wake failed for session "
                    "%s: %s",
                    raw_sid, e,
                )
                return False
        try:
            metadata = {}
            parent_session_id = str(evt.get("parent_session_id") or "").strip()
            if parent_session_id:
                metadata["gateway_session_id"] = parent_session_id
            synth_event = MessageEvent(
                text=synth_text,
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
                message_id=str(evt.get("message_id") or "").strip() or None,
                metadata=metadata,
            )
            logger.info(
                "Watch pattern notification — injecting for %s chat=%s thread=%s",
                platform_name,
                source.chat_id,
                source.thread_id,
            )
            # Relay-plane egress priming (defect #4, staging 2026-08-09): a
            # synthetic turn injected right after a restart reaches a relay
            # adapter whose per-chat routing caches are cold (they warm only
            # on inbound), so its replies egress without tenant
            # discriminators and the connector's fail-closed guard declines
            # them. Prime the caches from this event's session-store origin.
            _prime = getattr(adapter, "prime_routing_cache", None)
            if callable(_prime):
                _prime(synth_event)
            await adapter.handle_message(synth_event)
            return True
        except Exception as e:
            logger.error("Watch notification injection error: %s", e)
            return False

    @staticmethod
    def _completion_delivery_identity(evt: dict) -> Optional[tuple[str, str, object]]:
        """Return a producer-stable identity when one is available.

        Delegation UUIDs identify one producer completion. Process session IDs
        are normally unique too, but include the persisted spawn epoch so an
        explicitly reused ID represents a distinct process incarnation. Legacy
        process events without ``started_at`` are delivered without deduplication
        rather than risking suppression of a real completion.
        """
        evt_type = str(evt.get("type") or "")
        if evt_type == "async_delegation":
            producer_id = str(evt.get("delegation_id") or "")
            return (evt_type, producer_id, "") if producer_id else None
        if evt_type == "completion":
            producer_id = str(evt.get("session_id") or "")
            started_at = evt.get("started_at")
            if producer_id and started_at is not None:
                return (evt_type, producer_id, started_at)
        return None

    async def _classify_completion_target(self, parent_session_id: str) -> str:
        """Classify an async-completion delivery target before adapter acceptance.

        Returns one of:

        - ``"deliver"`` — the spawning session is live, or ended by a
          compression rotation with a verified live continuation. The inner
          #55578 resolver (:meth:`_resolve_async_delegation_session`) still
          owns the actual route retarget; this pre-flight only proves the
          completion is deliverable so the durable ack stays honest.
        - ``"terminal"`` — the spawning session is gone for good (unknown, or
          ended at an explicit user boundary such as /new). Delivery can never
          succeed; the durable row should be terminally dropped rather than
          falsely acknowledged as delivered or replayed forever as pending.
        - ``"retry"`` — transient uncertainty (session DB unavailable, lookup
          error, or a compression rotation caught mid-flight before its
          continuation exists). The claim should be released so a later
          consumer can retry; the attempt cap bounds the churn.
        """
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return "retry"
        try:
            parent = await session_db.get_session(parent_session_id)
        except Exception:
            logger.debug(
                "Async-completion pre-flight parent lookup failed for %s",
                parent_session_id, exc_info=True,
            )
            return "retry"
        if parent is None:
            return "terminal"
        if not parent.get("ended_at"):
            return "deliver"
        end_reason = str(parent.get("end_reason") or "")
        if end_reason != "compression":
            # An ended parent is only unreachable when the USER closed the
            # thread of work (explicit boundary: /new -> session_reset /
            # new_session, user_exit, session_switch). Idle/timeout ends are
            # normal lifecycle ends — the platform chat
            # remains routable, and the #55578 resolver retargets the
            # completion to the chat's current session. Dropping those loses
            # finished work (staging incident 2026-08-09: completed
            # delegation batch never delivered because the parent had
            # idle-ended). The boundary set is shared with the resolver
            # (_USER_BOUNDARY_END_REASONS) so this verdict and the pipeline's
            # routing decision cannot drift apart.
            if end_reason in _USER_BOUNDARY_END_REASONS:
                return "terminal"
            return "deliver"
        try:
            tip_session_id = await session_db.get_compression_tip(parent_session_id)
            if not tip_session_id or tip_session_id == parent_session_id:
                # Rotation caught mid-flight: parent is compression-ended but
                # its continuation isn't visible yet. Retry, don't drop.
                return "retry"
            tip = await session_db.get_session(tip_session_id)
        except Exception:
            logger.debug(
                "Async-completion pre-flight tip lookup failed for %s",
                parent_session_id, exc_info=True,
            )
            return "retry"
        if tip is None or tip.get("ended_at"):
            return "retry"
        return "deliver"

    async def _deliver_completion_notification(
        self, synth_text: str, evt: dict,
    ) -> Optional[bool]:
        """Deliver once per live gateway, or return False for a retry.

        ``True`` means this caller reached adapter acceptance, ``False`` means
        injection failed and the claim was released for retry, and ``None``
        means either another same-lifecycle caller owns/delivered the producer
        event or the event has no gateway route. No cross-process exactly-once
        guarantee is claimed.
        """
        identity = self._completion_delivery_identity(evt)
        durable_claim_id = ""
        durable_delegation_id = ""
        if evt.get("type") == "async_delegation":
            durable_delegation_id = str(evt.get("delegation_id") or "")
            if durable_delegation_id:
                try:
                    from tools.async_delegation import claim_completion_delivery

                    durable_claim_id = f"gateway:{id(self)}:{__import__('uuid').uuid4().hex}"
                    if not claim_completion_delivery(
                        durable_delegation_id, durable_claim_id,
                    ):
                        return None
                except Exception as exc:
                    logger.warning(
                        "Could not claim durable async completion %s: %s",
                        durable_delegation_id, exc,
                    )
                    return False
            parent_session_id = str(evt.get("parent_session_id") or "").strip()
            if parent_session_id:
                # Pre-flight (#65838-class): adapter acceptance is NOT proof of
                # delivery — the inner #55578 resolver can still fail closed
                # inside the message pipeline AFTER the adapter accepted, which
                # would falsely acknowledge the durable row as delivered.
                # Verify the target here, before acceptance, and give drops an
                # honest durable disposition.
                verdict = await self._classify_completion_target(parent_session_id)
                if verdict == "terminal":
                    logger.warning(
                        "Async delegation %s targets permanently-gone session %s; "
                        "terminally dropping delivery (result remains in the "
                        "delegation records).",
                        durable_delegation_id or "<legacy>", parent_session_id,
                    )
                    if durable_claim_id:
                        try:
                            from tools.async_delegation import drop_completion_delivery

                            drop_completion_delivery(
                                durable_delegation_id, durable_claim_id,
                            )
                        except Exception:
                            logger.debug(
                                "Could not drop durable completion claim",
                                exc_info=True,
                            )
                    return None
                if verdict == "retry":
                    if durable_claim_id:
                        try:
                            from tools.async_delegation import release_completion_delivery

                            release_completion_delivery(
                                durable_delegation_id, durable_claim_id,
                            )
                        except Exception:
                            logger.debug(
                                "Could not release durable completion claim",
                                exc_info=True,
                            )
                    return False
        elif evt.get("type") == "completion":
            # Background-process completions carry only session_key (chat/
            # thread routing), so after /new the notification from the OLD
            # session would land in the chat's NEW session. Stamped events
            # (spawn-time parent_session_id from terminal_tool) get the same
            # session-boundary pre-flight as async delegations — one policy
            # owner (_classify_completion_target), never a forked predicate.
            # Legacy/unstamped events keep today's behavior and deliver.
            parent_session_id = str(evt.get("parent_session_id") or "").strip()
            if parent_session_id:
                verdict = await self._classify_completion_target(parent_session_id)
                if verdict == "terminal":
                    logger.warning(
                        "Background process %s completion targets "
                        "permanently-gone session %s (user boundary such as "
                        "/new); dropping notification (output remains "
                        "available via process(action='log')).",
                        evt.get("session_id") or "<unknown>", parent_session_id,
                    )
                    return None
                if verdict == "retry":
                    # Transient uncertainty (session DB unavailable or a
                    # compression rotation mid-flight): signal the watcher to
                    # re-poll and try again rather than dropping or
                    # misrouting the result.
                    return False
        if identity is not None:
            with self._completion_delivery_lock:
                if (
                    identity in self._completion_deliveries_inflight
                    or identity in self._completion_deliveries_delivered
                ):
                    return None
                self._completion_deliveries_inflight.add(identity)

        accepted = False
        try:
            injection_result = await self._inject_watch_notification(synth_text, evt)
            if injection_result is not True:
                return injection_result
            accepted = True

            if identity is not None:
                with self._completion_delivery_lock:
                    self._completion_deliveries_inflight.discard(identity)
                    self._completion_deliveries_delivered[identity] = None
                    while (
                        len(self._completion_deliveries_delivered)
                        > self._completion_delivery_retention
                    ):
                        self._completion_deliveries_delivered.popitem(last=False)

            # If the durable async-delegation producer branch is present, its
            # SQLite row remains the authoritative replay state. Acknowledge it
            # after adapter acceptance; this gateway keeps no parallel ledger.
            if durable_claim_id:
                try:
                    from tools.async_delegation import complete_completion_delivery

                    complete_completion_delivery(
                        durable_delegation_id, durable_claim_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not acknowledge durable async completion %s: %s",
                        durable_delegation_id, exc,
                    )
            return True
        finally:
            if identity is not None and not accepted:
                with self._completion_delivery_lock:
                    self._completion_deliveries_inflight.discard(identity)
            if durable_claim_id and not accepted:
                try:
                    from tools.async_delegation import release_completion_delivery

                    release_completion_delivery(
                        durable_delegation_id, durable_claim_id,
                    )
                except Exception:
                    logger.debug("Could not release durable completion claim", exc_info=True)

    @staticmethod
    def _completion_notification_batch_key(evt: dict) -> tuple[str, ...]:
        """Return a routing-complete key for short-window process fan-in."""
        return tuple(str(evt.get(field) or "") for field in (
            "session_key",
            "platform",
            "chat_type",
            "chat_id",
            "thread_id",
            "user_id",
        ))

    @staticmethod
    def _format_coalesced_process_completions(entries: list[tuple[str, dict, asyncio.Future]]) -> str:
        """Build one bounded synthetic event from several redacted completions."""
        lines = [
            f"[IMPORTANT: {len(entries)} background processes completed for this session.",
            "Treat these results as one completion batch and send at most one "
            "consolidated user-facing response.",
        ]
        shown = entries[:10]
        for _text, evt, _future in shown:
            session_id = str(evt.get("session_id") or "unknown")
            exit_code = evt.get("exit_code")
            reason = str(evt.get("completion_reason") or "exited")
            # Completion-event output is normally passed through the terminal
            # redactor at the producer seam, but that redactor is deliberately
            # configurable.  This synthetic turn is gateway user-facing input,
            # so keep the unconditional gateway floor here as defence in depth.
            # Redact before slicing: truncating first can leave a credential
            # fragment that no longer matches the authoritative patterns.
            output = _redact_gateway_user_facing_secrets(
                str(evt.get("output") or "")
            ).strip()
            if len(output) > 800:
                output = f"[… truncated …]\n{output[-800:]}"
            lines.append(
                f"\n- {session_id}: exit_code={exit_code}, reason={reason}"
            )
            if output:
                lines.append(output)
        omitted = len(entries) - len(shown)
        if omitted:
            lines.append(
                f"\n- … and {omitted} more completion(s); inspect them with "
                "the process tool if they affect the conclusion."
            )
        lines.append(
            "If a result does not change the current conclusion, absorb it silently.]"
        )
        return "\n".join(lines)

    def _record_coalesced_completion_siblings(self, events: list[dict]) -> None:
        """Extend a successful primary delivery claim to its batched siblings."""
        with self._completion_delivery_lock:
            for evt in events:
                identity = self._completion_delivery_identity(evt)
                if identity is None:
                    continue
                self._completion_deliveries_inflight.discard(identity)
                self._completion_deliveries_delivered[identity] = None
            while (
                len(self._completion_deliveries_delivered)
                > self._completion_delivery_retention
            ):
                self._completion_deliveries_delivered.popitem(last=False)

    async def _flush_process_completion_batch(self, key: tuple[str, ...]) -> None:
        """Deliver one short-window completion batch and resolve its waiters."""
        current_task = asyncio.current_task()
        entries: list[tuple[str, dict, asyncio.Future]] = []
        delivered: Optional[bool] = False
        try:
            await asyncio.sleep(self._completion_notification_batch_window)
            entries = self._completion_notification_batches.pop(key, [])
            # Detach before adapter delivery.  A completion that arrives while
            # this batch is in flight must be able to schedule the next flush.
            if self._completion_notification_batch_tasks.get(key) is current_task:
                self._completion_notification_batch_tasks.pop(key, None)
            if not entries:
                return
            if len(entries) == 1:
                synth_text = entries[0][0]
            else:
                synth_text = self._format_coalesced_process_completions(entries)

            # A duplicate primary can legitimately return None from the
            # lifecycle dedupe seam.  Try the next batch identity so a
            # fresh sibling is never discarded with that duplicate.
            delivered = None
            for _text, candidate_evt, _future in entries:
                delivered = await self._deliver_completion_notification(
                    synth_text, candidate_evt,
                )
                if delivered is not None:
                    break
            if delivered is True and len(entries) > 1:
                self._record_coalesced_completion_siblings(
                    [evt for _text, evt, _future in entries]
                )
        except asyncio.CancelledError:
            # Shutdown may cancel us either during the fan-in window or while
            # adapter delivery is blocked.  Recover entries that have not yet
            # detached and resolve every waiter as retryable before adapters
            # are torn down.
            delivered = False
            if not entries:
                entries = self._completion_notification_batches.pop(key, [])
            raise
        except Exception:
            logger.exception("Coalesced process completion delivery failed")
            delivered = False
        finally:
            # Never strand watcher futures if formatting, delivery, or task
            # cancellation interrupts a batch.  False follows the existing
            # watcher retry path; None remains the ordinary dedupe result.
            for _text, _evt, future in entries:
                if not future.done():
                    future.set_result(delivered)
            # Do not remove a newer flush task that reused the same route key.
            if self._completion_notification_batch_tasks.get(key) is current_task:
                self._completion_notification_batch_tasks.pop(key, None)

    async def _cancel_process_completion_batch_tasks(self) -> None:
        """Settle pending completion batches before adapter teardown."""
        self._completion_notification_batches_stopping = True
        tasks = {
            task
            for task in getattr(
                self, "_completion_notification_batch_flush_tasks", set()
            )
            if not task.done()
        }
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Defensive cleanup for an orphaned queue with no live flush task.
        batches = getattr(self, "_completion_notification_batches", {})
        for entries in batches.values():
            for _text, _evt, future in entries:
                if not future.done():
                    future.set_result(False)
        batches.clear()
        getattr(self, "_completion_notification_batch_tasks", {}).clear()
        getattr(self, "_completion_notification_batch_flush_tasks", set()).clear()

    async def _enqueue_process_completion_notification(
        self, synth_text: str, evt: dict,
    ) -> Optional[bool]:
        """Fan in concurrent process completions that share one conversation."""
        # Some unit tests construct GatewayRunner with object.__new__.  Keep the
        # batching seam lazy so those focused lifecycle tests remain valid.
        if not hasattr(self, "_completion_notification_batches"):
            self._completion_notification_batches = {}
        if not hasattr(self, "_completion_notification_batch_tasks"):
            self._completion_notification_batch_tasks = {}
        if not hasattr(self, "_completion_notification_batch_flush_tasks"):
            self._completion_notification_batch_flush_tasks = set()
        if not hasattr(self, "_completion_notification_batch_window"):
            self._completion_notification_batch_window = 0.1
        if not hasattr(self, "_completion_notification_batches_stopping"):
            self._completion_notification_batches_stopping = False

        if self._completion_notification_batches_stopping:
            return False

        key = self._completion_notification_batch_key(evt)
        future = asyncio.get_running_loop().create_future()
        self._completion_notification_batches.setdefault(key, []).append(
            (synth_text, evt, future)
        )
        if key not in self._completion_notification_batch_tasks:
            task = asyncio.create_task(
                self._flush_process_completion_batch(key)
            )
            self._completion_notification_batch_tasks[key] = task
            # Keep the flush alive and include it in the gateway's normal
            # lifecycle accounting.  Focused tests that construct a runner via
            # object.__new__ lazily receive the same ownership set.
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = set()
            self._background_tasks.add(task)
            self._completion_notification_batch_flush_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(
                self._completion_notification_batch_flush_tasks.discard
            )
        return await future

    def _enrich_async_delegation_routing(self, evt: dict) -> None:
        """Fill platform/chat_id/thread_id/chat_type on an async-delegation event.

        Async-delegation completion events only carry ``session_key`` (the
        daemon worker has no access to the per-message routing metadata the
        terminal background watcher captures at spawn time). Parse the
        session_key into the routing fields ``_build_process_event_source``
        expects. Best-effort: a CLI-origin event (empty session_key) is left
        as-is and simply won't route on the gateway.
        """
        if evt.get("platform"):
            return  # already enriched
        parsed = _parse_session_key(evt.get("session_key", "") or "")
        if not parsed:
            return
        evt["platform"] = parsed.get("platform", "")
        evt["chat_type"] = parsed.get("chat_type", "")
        evt["chat_id"] = parsed.get("chat_id", "")
        if parsed.get("thread_id"):
            evt["thread_id"] = parsed["thread_id"]

    @staticmethod
    def _async_delegation_group_key(evt: dict) -> tuple[str, ...]:
        """Return the same-session routing key for async completion coalescing.

        Two events coalesce only when every routing dimension matches — the
        originating session key, the parent session the result re-enters, and
        the full gateway route. Events for different sessions never coalesce.
        """
        return tuple(str(evt.get(field) or "") for field in (
            "session_key",
            "parent_session_id",
            "platform",
            "chat_type",
            "chat_id",
            "thread_id",
            "user_id",
        ))

    @staticmethod
    def _format_coalesced_async_delegations(blocks: list[str]) -> str:
        """Join per-delegation formatted blocks into one consolidated turn."""
        header = (
            f"[IMPORTANT: {len(blocks)} background subagent delegations "
            "completed for this session. Treat these results as one "
            "completion batch and send at most one consolidated user-facing "
            "response. If a result does not change the current conclusion, "
            "absorb it silently.]"
        )
        return "\n\n".join([header, *blocks])

    async def _deliver_async_delegation_group(
        self, group: list[dict],
    ) -> Optional[bool]:
        """Deliver a same-session batch of async completions as ONE turn.

        A single-event group rides the existing per-event path unchanged. For
        a multi-event group the primary event is delivered through
        ``_deliver_completion_notification`` (which owns its durable claim,
        the lifecycle dedupe, and the target preflight), carrying a
        consolidated text that also contains every sibling result whose
        durable row THIS runner successfully claimed up front. Only after
        adapter acceptance are the sibling claims acknowledged — the durable
        ledger never acks work that was not delivered, and a sibling claimed
        by another consumer is excluded from the consolidated text entirely
        so its content cannot be double-delivered.

        Returns ``True`` after adapter acceptance, ``False`` when the caller
        should requeue the group for retry, and ``None`` when nothing in the
        group is deliverable by this runner (siblings that still need a retry
        are requeued here before returning).
        """
        from tools.process_registry import process_registry as _pr

        deliverable: list[tuple[dict, str]] = []
        for evt in group:
            synth_text = _format_gateway_process_notification(evt)
            if not synth_text:
                continue
            identity = self._completion_delivery_identity(evt)
            if identity is not None:
                with self._completion_delivery_lock:
                    if (
                        identity in self._completion_deliveries_inflight
                        or identity in self._completion_deliveries_delivered
                    ):
                        continue
            deliverable.append((evt, synth_text))

        if not deliverable:
            return None
        if len(deliverable) == 1:
            evt, synth_text = deliverable[0]
            return await self._deliver_completion_notification(synth_text, evt)

        from tools.async_delegation import (
            claim_event_delivery,
            complete_event_delivery,
            release_event_delivery,
        )

        primary_evt, primary_text = deliverable[0]
        blocks = [primary_text]
        siblings: list[tuple[dict, str]] = []
        for evt, synth_text in deliverable[1:]:
            claim_id = claim_event_delivery(evt, f"gateway-batch:{id(self)}")
            if claim_id is None:
                # Another consumer owns this row's delivery; keep its result
                # out of our consolidated text so it is never double-injected.
                continue
            siblings.append((evt, claim_id))
            blocks.append(synth_text)

        if not siblings:
            return await self._deliver_completion_notification(
                primary_text, primary_evt,
            )

        consolidated = self._format_coalesced_async_delegations(blocks)
        delivered: Optional[bool] = False
        try:
            delivered = await self._deliver_completion_notification(
                consolidated, primary_evt,
            )
        finally:
            if delivered is True:
                for evt, claim_id in siblings:
                    try:
                        complete_event_delivery(evt, claim_id)
                    except Exception:
                        logger.debug(
                            "Could not acknowledge coalesced durable completion",
                            exc_info=True,
                        )
                self._record_coalesced_completion_siblings(
                    [evt for evt, _claim_id in siblings]
                )
            else:
                # Not delivered — release every sibling claim so a retry (or
                # another consumer) can claim it, honestly leaving the durable
                # rows pending.
                for evt, claim_id in siblings:
                    try:
                        release_event_delivery(evt, claim_id)
                    except Exception:
                        logger.debug(
                            "Could not release coalesced durable claim",
                            exc_info=True,
                        )
                if delivered is None:
                    # The primary was dropped/owned elsewhere but the siblings
                    # still need delivery — requeue just them for the next tick.
                    for evt, _claim_id in siblings:
                        _pr.completion_queue.put(evt)
        return delivered

    async def _async_delegation_watcher(self, interval: float = 2.0) -> None:
        """Drain async-delegation completions and inject them as new turns.

        Background subagents (``delegate_task(background=true)``) run on the
        async-delegation daemon executor — they have no per-process watcher
        task, so their completion events would only be seen by the post-turn
        queue drain. This watcher covers the IDLE case: when a background
        subagent finishes while no agent turn is running, its result still
        re-enters the originating session promptly.

        Mirrors the CLI's idle ``process_loop`` drain. Stays silent when the
        queue has nothing for us; ignores non-async event types (those are
        handled by ``_run_process_watcher`` / the post-turn drain).
        """
        await asyncio.sleep(3)  # let platforms finish connecting
        from tools.process_registry import process_registry as _pr
        while self._running:
            try:
                # Peek the queue for async-delegation events. We must NOT
                # consume watch/completion events here (other drains own them),
                # so requeue anything that isn't ours.
                requeue = []
                async_events = []
                while not _pr.completion_queue.empty():
                    try:
                        evt = _pr.completion_queue.get_nowait()
                    except Exception:
                        break
                    if evt.get("type") == "async_delegation":
                        async_events.append(evt)
                    else:
                        requeue.append(evt)
                for evt in requeue:
                    _pr.completion_queue.put(evt)
                # A same-tick drain often carries several completions for the
                # SAME originating session (a fan-out of background subagents
                # finishing together).  Delivering each one individually floods
                # the session with N synthetic turns (#70300) — group by full
                # gateway route + parent session and inject one consolidated
                # turn per group.  Events for different sessions never coalesce.
                groups: dict[tuple[str, ...], list[dict]] = {}
                group_order: list[tuple[str, ...]] = []
                for evt in async_events:
                    self._enrich_async_delegation_routing(evt)
                    key = self._async_delegation_group_key(evt)
                    if key not in groups:
                        groups[key] = []
                        group_order.append(key)
                    groups[key].append(evt)
                for key in group_order:
                    group = groups[key]
                    try:
                        delivered = await self._deliver_async_delegation_group(group)
                        if delivered is False:
                            for evt in group:
                                _pr.completion_queue.put(evt)
                    except Exception as e:
                        for evt in group:
                            _pr.completion_queue.put(evt)
                        logger.error("Async delegation injection error: %s", e)
            except Exception as e:
                logger.debug("Async delegation watcher error: %s", e)
            await asyncio.sleep(interval)

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``concise`` — one-line status message on completion (default);
            failures append a short output tail
          - ``all``    — running-output updates + final raw-output message
          - ``result`` — final raw-output completion message only
          - ``error``  — final raw-output message only when exit code != 0
          - ``off``    — no messages at all
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        user_id = watcher.get("user_id", "")
        user_name = watcher.get("user_name", "")
        message_id = str(watcher.get("message_id") or "").strip() or None
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = self._load_background_notifications_mode()

        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)

        if notify_mode == "off" and not agent_notify:
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # --- Agent-triggered completion: inject synthetic message ---
                # Skip if the agent already consumed the result via wait/log.
                # poll() is read-only and intentionally does NOT mark consumed
                # (#10156) — a status check must not suppress this delivery turn.
                from tools.process_registry import format_process_notification, process_registry as _pr_check
                if agent_notify and not _pr_check.is_completion_consumed(session_id):
                    from agent.redact import redact_terminal_output
                    from tools.ansi_strip import strip_ansi
                    _command = getattr(session, "command", "") or ""
                    _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
                    _raw = redact_terminal_output(_raw, _command)
                    _command = _redact_gateway_user_facing_secrets(_command)
                    # Truncate at line boundaries so notifications never start
                    # mid-line (fixes #23284). Keep the last ~2000 chars but
                    # snap to the nearest preceding newline, then prepend a
                    # truncation marker when output was cut.
                    _LIMIT = 2000
                    if len(_raw) > _LIMIT:
                        _tail = _raw[-_LIMIT:]
                        _nl = _tail.find("\n")
                        _tail = _tail[_nl + 1:] if _nl != -1 else _tail
                        _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
                    else:
                        _out = _raw
                    _out = _redact_gateway_user_facing_secrets(_out)
                    completion_evt = {
                        "type": "completion",
                        "session_id": session_id,
                        "session_key": session_key,
                        "platform": platform_name,
                        "chat_type": watcher.get("chat_type", ""),
                        "chat_id": chat_id,
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "user_name": user_name,
                        "message_id": message_id,
                        "started_at": getattr(session, "started_at", None),
                        "command": _command,
                        "exit_code": session.exit_code,
                        "completion_reason": getattr(session, "completion_reason", "exited"),
                        "termination_source": getattr(session, "termination_source", ""),
                        "output": _out,
                        # Spawning conversation's session-db id (stamped at
                        # spawn time in terminal_tool). Lets the delivery
                        # pre-flight drop this completion when the user closed
                        # that session (/new) before the process finished.
                        "parent_session_id": (
                            watcher.get("parent_session_id")
                            or getattr(session, "parent_session_id", "")
                            or ""
                        ),
                    }
                    synth_text = format_process_notification(completion_evt)
                    if not synth_text:
                        break
                    delivered = await self._enqueue_process_completion_notification(
                        synth_text, completion_evt,
                    )
                    if delivered is False:
                        # The process remains terminal; retry after failed
                        # adapter injection instead of suppressing the result.
                        continue
                    break

                # --- Normal text-only notification ---
                # Skip when the agent already consumed this completion via
                # wait/log (#65379): process(wait) returned the exit code and
                # output inline, so the raw "[Background process ... finished
                # with exit code ...]" message would be a duplicate delivery
                # of the same completion. The agent_notify branch above
                # already honors _completion_consumed; without this check its
                # skip FALLS THROUGH to this block and re-delivers the output
                # the agent is actively summarizing. poll() is read-only and
                # intentionally does not mark consumed (#10156), so a status
                # check never suppresses this message.
                if _pr_check.is_completion_consumed(session_id):
                    logger.debug(
                        "Process watcher: completion for %s already consumed "
                        "via wait/log — skipping raw notification (#65379)",
                        session_id,
                    )
                    break
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in {"concise", "all", "result"}
                    or (notify_mode == "error" and session.exit_code not in {0, None})
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    if new_output:
                        from agent.redact import redact_terminal_output
                        new_output = redact_terminal_output(
                            new_output, getattr(session, "command", "") or ""
                        )
                        # redact_terminal_output() is unforced, so it returns raw
                        # text when security.redact_secrets is off.  This send
                        # goes straight to the platform adapter, so it needs the
                        # same unconditional floor as the agent-notify path.
                        new_output = _redact_gateway_user_facing_secrets(new_output)
                    if notify_mode == "concise":
                        _cmd_disp = _redact_gateway_user_facing_secrets(
                            getattr(session, "command", "") or ""
                        )
                        _started = getattr(session, "started_at", None)
                        _dur = None
                        if isinstance(_started, (int, float)):
                            _dur = max(0.0, time.time() - _started)
                        message_text = _format_concise_process_notification(
                            session_id,
                            _cmd_disp,
                            session.exit_code,
                            new_output,
                            duration_seconds=_dur,
                        )
                    else:
                        message_text = (
                            f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                            f"Here's the final output:\n{new_output}]"
                        )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(
                                chat_id,
                                message_text,
                                metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                            )
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output available -- deliver status update (only in "all" mode)
                # Skip periodic updates for agent_notify watchers (they only care about completion)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                if new_output:
                    from agent.redact import redact_terminal_output
                    new_output = redact_terminal_output(
                        new_output, getattr(session, "command", "") or ""
                    )
                    new_output = _redact_gateway_user_facing_secrets(new_output)
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(
                            chat_id,
                            message_text,
                            metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                        )
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)
