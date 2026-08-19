"""Messaging-platform connection, credential, and reconnect runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.async_utils import consume_detached_task_result
from gateway.config import (
    Platform,
    PlatformConfig,
    platform_binds_port as _platform_binds_port,
)
from gateway.history import _float_env
from gateway.runtime_config import _load_gateway_runtime_config
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionSource
from hermes_constants import get_hermes_home

logger = logging.getLogger("gateway.run")

_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0
_TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT = 180.0
_TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT = 45.0
_RECONNECT_BACKOFF_CAP = 300
_RECONNECT_ATTENTION_AFTER_SECONDS = _float_env(
    "HERMES_RECONNECT_ATTENTION_AFTER_SECONDS", 7200
)
_DOCKER_VOLUME_SPEC_RE = re.compile(
    r"^(?P<host>.+):(?P<container>/[^:]+?)(?::(?P<options>[^:]+))?$"
)
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/outputs"}
_UNSET = object()

def _platform_has_bot_credential(platform: "Platform", platform_config: "PlatformConfig") -> bool:
    """Return True when a token-authenticated platform has a usable bot credential.

    Platforms that do not use ``PlatformConfig.token`` always return True so we
    never skip them here (Signal session paths, port-binding HTTP adapters, etc.).
    """
    from gateway.config import PLATFORM_TOKEN_ENV_NAMES

    if platform not in PLATFORM_TOKEN_ENV_NAMES:
        return True
    token = getattr(platform_config, "token", None) or ""
    if isinstance(token, str) and token.strip():
        return True
    # Some adapters also accept api_key as the primary credential.
    api_key = getattr(platform_config, "api_key", None) or ""
    if isinstance(api_key, str) and api_key.strip():
        return True
    return False

async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto ``self.adapters``.

    The reconnect watcher in ``GatewayRunner._platform_reconnect_watcher``
    constructs a fresh adapter on every retry attempt. When the connect
    call fails — for any of the three reasons (non-retryable error,
    retryable error, exception during connect) — the adapter is dropped
    without ever being installed, so nothing else will call its
    ``disconnect()``. Any resources the adapter opened in ``__init__``
    (for example a database-backed adapter) stay open until
    garbage collection sweeps the unreachable object, which Python's
    cyclic GC does not do promptly for asyncio-bound objects with
    native handles. The cumulative leak is 2 fds × every retry at the
    300s backoff cap ≈ 12 fds/hour, and the default 2560-fd ulimit
    is exhausted in ~12h of continuous failure, after which every
    open() call on the gateway raises ``OSError: [Errno 24] Too many
    open files`` and the gateway becomes a zombie (#37011).

    This helper centralises the dispose-with-suppression so the three
    failure paths in the reconnect watcher can all call it without
    each one having to know that ``disconnect()`` may itself raise
    on a half-constructed adapter.

    ``adapter`` may be ``None``: the reconnect watcher initialises
    ``adapter = None`` before the ``try`` so the ``except Exception``
    arm can dispose a half-constructed object, and also early-returns
    here when ``_create_adapter()`` returned ``None``.
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters can raise from
        # disconnect() on objects that never finished initializing.
        # We must not let that escape and abort the watcher loop.
        #
        # On Python 3.8+, ``asyncio.CancelledError`` inherits from
        # ``BaseException`` (not ``Exception``), so this ``except
        # Exception`` does not swallow task cancellation. We don't
        # re-raise explicitly because the watcher loop intentionally
        # treats dispose failures as best-effort: a failed ``disconnect``
        # call should not take down the reconnect watcher that
        # itself is what's keeping the gateway alive during a partial
        # outage.
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__),
            exc_info=True,
        )

def _reconnect_backoff(attempt: int) -> int:
    """Exponential reconnect backoff: 30s, 60s, 120s, ... capped at 5 min."""
    return min(30 * (2 ** (attempt - 1)), _RECONNECT_BACKOFF_CAP)

def _reconnect_needs_attention(info: dict, now: float) -> bool:
    """Return True when a reconnect-queue entry has been continuously queued
    long enough to warrant a NEEDS_ATTENTION signal.

    ``queued_at`` is (re)stamped whenever the platform (re)enters the queue,
    so a platform that reconnects successfully and later fails again starts a
    fresh clock — only *continuous* failure escalates. Entries queued before
    this field existed (in-flight upgrade) are treated as newly queued.
    """
    if _RECONNECT_ATTENTION_AFTER_SECONDS <= 0:
        return False  # escalation disabled
    queued_at = info.get("queued_at")
    if queued_at is None:
        info["queued_at"] = now
        return False
    return (now - queued_at) >= _RECONNECT_ATTENTION_AFTER_SECONDS

class PlatformRuntime:

    def _warn_if_docker_media_delivery_is_risky(self) -> None:
        """Warn when Docker-backed gateways lack an explicit export mount.

        MEDIA delivery happens in the gateway process, so paths emitted by the model
        must be readable from the host. A plain container-local path like
        `/workspace/report.txt` or `/output/report.txt` often exists only inside
        Docker, so users commonly need a dedicated export mount such as
        `host-dir:/output`.
        """
        if os.getenv("TERMINAL_ENV", "").strip().lower() != "docker":
            return

        connected = self.config.get_connected_platforms()
        messaging_platforms = [p for p in connected if p != Platform.LOCAL]
        if not messaging_platforms:
            return

        raw_volumes = os.getenv("TERMINAL_DOCKER_VOLUMES", "").strip()
        volumes: List[str] = []
        if raw_volumes:
            try:
                parsed = json.loads(raw_volumes)
                if isinstance(parsed, list):
                    volumes = [str(v) for v in parsed if isinstance(v, str)]
            except Exception:
                logger.debug("Could not parse TERMINAL_DOCKER_VOLUMES for gateway media warning", exc_info=True)

        has_explicit_output_mount = False
        for spec in volumes:
            match = _DOCKER_VOLUME_SPEC_RE.match(spec)
            if not match:
                continue
            container_path = match.group("container")
            if container_path in _DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS:
                has_explicit_output_mount = True
                break

        if has_explicit_output_mount:
            return

        logger.warning(
            "Docker backend is enabled for the messaging gateway but no explicit host-visible "
            "output mount (for example '/home/user/.hermes/cache/documents:/output') is configured. "
            "This is fine if the model already emits host-visible paths, but MEDIA file delivery can fail "
            "for container-local paths like '/workspace/...' or '/output/...'."
        )

    def _has_setup_skill(self) -> bool:
        """Check if the hermes-agent-setup skill is installed."""
        try:
            from tools.skill_manager_tool import _find_skill
            return _find_skill("hermes-agent-setup") is not None
        except Exception:
            return False

    async def _await_adapter_cleanup_with_timeout(
        self, awaitable: Awaitable[Any], timeout: float
    ) -> bool:
        """Wait for adapter cleanup without letting cancellation swallowing hang us.

        ``asyncio.wait_for`` cancels an overdue child but then waits for it to
        exit. An adapter close path that catches ``CancelledError`` can therefore
        block recovery forever. Keep ownership of the old task through its done
        callback, but release the runner at the deadline.
        """
        if timeout <= 0:
            await awaitable
            return True

        task = asyncio.ensure_future(awaitable)
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(consume_detached_task_result)
            raise
        if task in done:
            await task
            return True

        task.cancel()
        task.add_done_callback(consume_detached_task_result)
        return False

    async def _safe_adapter_disconnect(self, adapter, platform) -> None:
        """Call adapter.disconnect() defensively, swallowing any error.

        Used when adapter.connect() failed or raised — the adapter may
        have allocated partial resources (aiohttp.ClientSession, poll
        tasks, child subprocesses) that would otherwise leak and surface
        as "Unclosed client session" warnings at process exit.

        Must tolerate partial-init state and never raise, since callers
        use it inside error-handling blocks.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        try:
            completed = await self._await_adapter_cleanup_with_timeout(
                adapter.disconnect(), timeout
            )
            if not completed:
                logger.warning(
                    "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
                    timeout,
                    platform.value if platform is not None else "adapter",
                )
        except Exception as e:
            logger.debug(
                "Defensive %s disconnect after failed connect raised: %s",
                platform.value if platform is not None else "adapter",
                e,
            )

    async def _bounded_adapter_teardown(
        self, adapter, platform, *, profile: Optional[str] = None
    ) -> None:
        """Tear down one adapter on the shutdown path with bounded awaits.

        Both ``cancel_background_tasks()`` and ``disconnect()`` can block
        indefinitely when a platform's network state is half-dead (e.g. a
        wedged Feishu/Lark WebSocket thread waiting on I/O). An unbounded
        await here stalls the entire shutdown sequence past systemd's
        ``TimeoutStopSec``; the resulting SIGKILL skips ``atexit`` PID-file
        cleanup, so the next start dies with "PID file race lost" (#14128).

        Each await uses the existing per-adapter timeout budget
        (``HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT``). On timeout the old
        task is cancelled and detached, then teardown forces forward progress;
        the loop never hangs even if an adapter swallows cancellation. Never
        raises.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        suffix = f" (profile: {profile})" if profile else ""
        started_at = time.monotonic()
        try:
            cancelled = await self._await_adapter_cleanup_with_timeout(
                adapter.cancel_background_tasks(), timeout
            )
            if not cancelled:
                logger.warning(
                    "✗ %s background-task cancel timed out after %.1fs - forcing continue%s",
                    platform.value, timeout, suffix,
                )
        except Exception as e:
            logger.debug("✗ %s background-task cancel error%s: %s", platform.value, suffix, e)
        try:
            disconnected = await self._await_adapter_cleanup_with_timeout(
                adapter.disconnect(), timeout
            )
            if disconnected:
                logger.info(
                    "✓ %s disconnected (%.2fs)%s",
                    platform.value, time.monotonic() - started_at, suffix,
                )
            else:
                logger.warning(
                    "✗ %s disconnect timed out after %.1fs - forcing continue%s",
                    platform.value, timeout, suffix,
                )
        except Exception as e:
            logger.error(
                "✗ %s disconnect error after %.2fs%s: %s",
                platform.value, time.monotonic() - started_at, suffix, e,
            )

    def _adapter_disconnect_timeout_secs(self) -> float:
        """Return the per-adapter disconnect timeout used during shutdown."""
        raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT

    def _platform_connect_timeout_secs(self, platform=None, *, initial: bool = False) -> float:
        """Return the per-platform connect timeout used during startup/retry.

        ``initial=True`` marks the cold-start connect awaited before the
        gateway reaches ``running``. Telegram's full connect budget (180s,
        raised for #67498 so cold polling can prove getUpdates readiness) is
        deliberately NOT spent there: an unreachable Telegram would hold the
        whole gateway out of the ``running`` state for the full budget
        (#85993). The cold-start wait is capped and the platform is handed to
        the reconnect watcher, which retries with the full budget (and
        ``is_reconnect=True``, preserving the offline update queue — #46621).
        """
        raw = os.getenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        if platform == Platform.TELEGRAM:
            if initial:
                return _TELEGRAM_INITIAL_CONNECT_TIMEOUT_SECS_DEFAULT
            return _TELEGRAM_CONNECT_TIMEOUT_SECS_DEFAULT
        return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT

    async def _connect_adapter_with_timeout(
        self, adapter, platform, *, is_reconnect: bool = False, initial: bool = False
    ) -> bool:
        """Connect an adapter without allowing one platform to block others.

        ``is_reconnect`` is forwarded to ``adapter.connect()`` so platform
        adapters can distinguish a cold first boot (drop any stale
        server-side queue) from a watcher reconnect after a prolonged outage
        (preserve the queue so messages sent during the outage are delivered
        rather than silently dropped — #46621).

        ``initial`` selects the capped cold-start budget for platforms whose
        full connect budget is too long to spend before the gateway reaches
        ``running`` (#85993 — Telegram's 180s).
        """
        timeout = self._platform_connect_timeout_secs(platform, initial=initial)
        if timeout <= 0:
            return await adapter.connect(is_reconnect=is_reconnect)
        # Use the detach-on-timeout pattern instead of plain asyncio.wait_for:
        # asyncio.wait_for cancels the overdue task but then waits for it to
        # exit. An adapter connect() that catches CancelledError can therefore
        # block recovery forever (the watcher never reaches the next retry).
        # Keep ownership of the old task through its done callback, but
        # release the runner at the deadline (#70344).
        task = asyncio.ensure_future(
            adapter.connect(is_reconnect=is_reconnect)
        )
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(consume_detached_task_result)
            raise
        if task in done:
            result = await task
            return bool(result)
        task.cancel()
        task.add_done_callback(consume_detached_task_result)
        raise TimeoutError(
            f"{platform.value} connect timed out after {timeout:g}s"
        )

    async def _connect_initial_adapter_with_timeout(self, adapter, platform) -> bool:
        """Connect one cold-start adapter with tightly scoped replace intent.

        The capability is visible only while this initial connect is awaited.
        Reconnects call ``_connect_adapter_with_timeout`` directly and adapters
        also default to deny, so a later network recovery can never evict a
        healthy token holder.
        """
        adapter._platform_lock_takeover_allowed = bool(
            self._platform_lock_takeover_on_start
        )
        try:
            return await self._connect_adapter_with_timeout(
                adapter, platform, initial=True
            )
        finally:
            adapter._platform_lock_takeover_allowed = False

    async def _handle_reaction_event(self, ctx: Dict[str, Any]) -> None:
        """Fan a normalised platform reaction event out to the HookRegistry.

        Adapters call this via ``set_reaction_handler`` for every
        platform-native reaction event they surface. The adapter-supplied
        ``event_name`` ("reaction:added" / "reaction:removed") becomes the
        hook event so user hooks subscribe with the same name scheme as the
        existing ``agent:*`` family. Errors never block the adapter's event
        loop — the hook contract is non-blocking.
        """
        event_name = str(ctx.get("event_name") or "reaction:added")
        try:
            await self.hooks.emit(event_name, ctx)
        except Exception:
            logger.debug("[Gateway] reaction hook emit failed", exc_info=True)

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup.

        If the error is retryable (e.g. network blip, DNS failure), queue the
        platform for background reconnection instead of giving up permanently.

        The notification arrives on the failing adapter's own polling task,
        and the disconnect inside the handler can cancel that task mid-flight:
        disconnect()'s current-task guard misses it because
        _safe_adapter_disconnect runs the close in a wrapper task. A cancelled
        handler dies between the fatal log and the reconnect queue, silently
        stranding the platform (observed 2026-07-21: telegram popped from
        adapters but never queued after a travel network outage). Run the real
        work in a detached task that adapter teardown cannot cancel.
        """
        tasks = getattr(self, "_fatal_handler_tasks", None)
        if tasks is None:
            tasks = self._fatal_handler_tasks = set()
        task = asyncio.create_task(self._handle_adapter_fatal_error_detached(adapter))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        # Await so callers that expect completion still get it — but through
        # shield(): Task.cancel() on the caller also cancels the future it is
        # awaiting (_fut_waiter), so a plain `await task` would tunnel the
        # cancellation straight into the "detached" task. shield() absorbs
        # it: the caller sees CancelledError, the handler runs to completion.
        await asyncio.shield(task)

    def _queue_retryable_fatal_platform(self, adapter: BasePlatformAdapter) -> bool:
        """Queue a retryable fatal adapter for background reconnection.

        Returns True when the platform was newly queued. Idempotent if already
        queued. Must not await: callers invoke this *before* any disconnect
        await so a wedged close cannot strand the platform (#80598).
        """
        if not adapter.fatal_error_retryable:
            return False
        platform_config = self.config.platforms.get(adapter.platform)
        if not platform_config or adapter.platform in self._failed_platforms:
            return False
        self._failed_platforms[adapter.platform] = {
            "config": platform_config,
            "attempts": 0,
            "next_retry": time.monotonic(),
            "queued_at": time.monotonic(),
            "credential_claim": self._adapter_credential_claim(
                adapter.platform, adapter
            ),
        }
        logger.info(
            "%s queued for background reconnection",
            adapter.platform.value,
        )
        # Ensure the reconnect watcher is alive — if it died (e.g. from
        # exhausting its restart budget), respawn it so queued platforms
        # are not permanently stranded (#70344).
        self._ensure_reconnect_watcher_running()
        return True

    async def _handle_adapter_fatal_error_detached(
        self, adapter: BasePlatformAdapter
    ) -> None:
        """Run the fatal handler; if the platform still ends up stranded
        (not reconnected, not queued, not intentionally disabled), exit the
        gateway with failure so the service manager restarts it instead of
        leaving a silent partial outage."""
        try:
            # Outer hard deadline (#80598): even with queue-before-disconnect,
            # a hang anywhere in the impl (status write side effects, detach
            # races, etc.) must not leave this task wedged forever — the
            # stranded check in ``finally`` only runs when we return.
            timeout = self._adapter_disconnect_timeout_secs()
            if timeout <= 0:
                await self._handle_adapter_fatal_error_impl(adapter)
            else:
                # Disconnect budget plus a small overhead for queue/status
                # bookkeeping. Keep the additive proportional so tests that
                # shrink the disconnect timeout still finish promptly.
                outer = timeout + min(2.0, max(0.05, timeout))
                completed = await self._await_adapter_cleanup_with_timeout(
                    self._handle_adapter_fatal_error_impl(adapter),
                    outer,
                )
                if not completed:
                    logger.error(
                        "Fatal-error handling for %s timed out after %.1fs; "
                        "ensuring reconnect queue is populated",
                        adapter.platform.value,
                        outer,
                    )
                    self._queue_retryable_fatal_platform(adapter)
        except asyncio.CancelledError:
            # Best-effort queue before re-raising: a cancelled fatal handler
            # must not strand a retryable platform (#80598).
            try:
                self._queue_retryable_fatal_platform(adapter)
            except Exception:
                logger.debug(
                    "Failed to queue %s after fatal-handler cancellation",
                    adapter.platform.value,
                    exc_info=True,
                )
            raise
        except Exception:
            logger.exception(
                "Fatal-error handling for %s raised unexpectedly",
                adapter.platform.value,
            )
            # Best-effort queue so an unexpected raise mid-handler cannot
            # leave a retryable platform permanently deaf (#80598).
            try:
                self._queue_retryable_fatal_platform(adapter)
            except Exception:
                logger.debug(
                    "Failed to queue %s after fatal-handler exception",
                    adapter.platform.value,
                    exc_info=True,
                )
        finally:
            platform = adapter.platform
            shutdown_event = getattr(self, "_shutdown_event", None)
            stranded = (
                adapter.fatal_error_retryable
                and platform not in self.adapters
                and platform not in getattr(self, "_failed_platforms", {})
                and not (shutdown_event is not None and shutdown_event.is_set())
            )
            if stranded:
                logger.error(
                    "%s adapter was lost without entering the reconnection "
                    "queue; exiting gateway so the service manager restarts it.",
                    platform.value,
                )
                self._exit_reason = (
                    f"{platform.value} adapter lost without reconnection queue"
                )
                self._exit_with_failure = True
                await self.stop()

    async def _handle_adapter_fatal_error_impl(self, adapter: BasePlatformAdapter) -> None:
        # Snapshot the current owner of this platform slot before doing
        # anything else. If it's neither this adapter nor empty, a different
        # adapter has already taken over (e.g. this is a delayed notification
        # from a background retry chain that raced with, and lost to, a
        # reconnect that already succeeded). Acting on a stale notification
        # would overwrite an already-healthy platform's runtime status and
        # incorrectly re-queue it for reconnection, so bail out before any of
        # that happens.
        existing = self.adapters.get(adapter.platform)
        if existing is not None and existing is not adapter:
            logger.debug(
                "Ignoring stale fatal error from a superseded %s adapter instance: %s",
                adapter.platform.value,
                adapter.fatal_error_code or "unknown",
            )
            return

        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )
        if adapter.fatal_error_retryable:
            platform_state = "retrying"
        else:
            platform_state = "fatal"
        self._update_platform_runtime_status(
            adapter.platform.value,
            platform_state=platform_state,
            error_code=adapter.fatal_error_code,
            error_message=adapter.fatal_error_message,
        )

        if existing is adapter:
            # Claim this adapter for teardown before awaiting disconnect() —
            # a second fatal-error notification for the same adapter (e.g.
            # from a concurrent recovery path) would otherwise still see
            # itself as "existing" during the await below and disconnect()
            # the same object twice.
            self.adapters.pop(adapter.platform, None)
            self.delivery_router.adapters = self.adapters

        # Queue retryable failures BEFORE any disconnect await (#80598).
        # A half-dead transport can wedge native close() (or swallow
        # CancelledError inside it) so the previous "disconnect then queue"
        # order left platforms permanently deaf inside a live process even
        # after the network recovered. Populate the queue first so the
        # reconnect watcher always has work; teardown is best-effort after.
        self._queue_retryable_fatal_platform(adapter)

        if existing is adapter:
            # A half-closed transport can wedge an adapter's native close()
            # indefinitely. Reuse the shutdown-path timeout so this runtime
            # fatal handler always returns to the stay-alive / stranded path.
            await self._safe_adapter_disconnect(adapter, adapter.platform)

        if not self.adapters and not self._failed_platforms:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            if adapter.fatal_error_retryable:
                self._exit_with_failure = True
                logger.error("No connected messaging platforms remain. Shutting down gateway for service restart.")
            else:
                logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()
        elif not self.adapters and self._failed_platforms:
            # All platforms are down and queued for background reconnection.
            # Keep the gateway alive so:
            #   • cron jobs still run
            #   • the reconnect watcher can recover platforms when the
            #     underlying problem clears (proxy comes back, user runs
            #     `hermes gateway setup telegram`, etc.)
            # We used to exit-with-failure here to trigger systemd restart,
            # but that converted a transient outage into a restart loop and
            # killed in-process state every time. The reconnect watcher
            # already handles long-running recovery — let it do its job.
            logger.warning(
                "No connected messaging platforms remain, but %d platform(s) "
                "queued for reconnection — gateway staying alive, watcher will "
                "retry in background.",
                len(self._failed_platforms),
            )

    def _update_platform_runtime_status(
        self,
        platform: str,
        *,
        platform_state: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        needs_attention: Optional[bool] = None,
        retrying_since: Any = _UNSET,
    ) -> None:
        try:
            from gateway.status import write_runtime_status
            extra: Dict[str, Any] = {}
            if needs_attention is not None:
                extra["needs_attention"] = needs_attention
            if retrying_since is not _UNSET:
                extra["retrying_since"] = retrying_since
            write_runtime_status(
                platform=platform,
                platform_state=platform_state,
                error_code=error_code,
                error_message=error_message,
                **extra,
            )
        except Exception:
            pass

    def _pause_failed_platform(self, platform, *, reason: str = "") -> None:
        """Mark a queued platform as paused — keep it in ``_failed_platforms``
        but stop the reconnect watcher from hammering it.

        Used by ``/platform pause <name>`` for manual operator intervention.
        Paused platforms are surfaced in ``/platform list`` and resumed with
        ``/platform resume <name>``.  Note: the reconnect watcher does NOT
        auto-pause — retryable (network/DNS) failures keep retrying at the
        backoff cap indefinitely so a transient outage self-heals without
        manual intervention.
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return
        if info.get("paused"):
            return
        info["paused"] = True
        info["pause_reason"] = reason or "auto-paused after repeated failures"
        # Push next_retry far enough out that even if "paused" is missed
        # by a stale code path, the watcher won't fire on it.
        info["next_retry"] = float("inf")
        try:
            self._update_platform_runtime_status(
                platform.value,
                platform_state="paused",
                error_code=None,
                error_message=info["pause_reason"],
            )
        except Exception:
            pass
        logger.warning(
            "%s paused after %d consecutive failures (%s) — "
            "fix the underlying issue then run `/platform resume %s` "
            "to retry, or `hermes gateway restart` to restart the gateway.",
            platform.value, info.get("attempts", 0),
            info["pause_reason"], platform.value,
        )

    def _resume_paused_platform(self, platform) -> bool:
        """Unpause a platform — reset its attempt counter and schedule an
        immediate retry.  Returns True if the platform was paused and is
        now queued; False if it wasn't paused (or wasn't in the queue).
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return False
        if not info.get("paused"):
            return False
        info["paused"] = False
        info.pop("pause_reason", None)
        info["attempts"] = 0
        info["next_retry"] = time.monotonic()  # retry on next watcher tick
        try:
            self._update_platform_runtime_status(
                platform.value,
                platform_state="retrying",
                error_code=None,
                error_message=None,
            )
        except Exception:
            pass
        logger.info("%s resumed — retrying on next watcher tick", platform.value)
        return True

    def _active_profile_name(self) -> str:
        """Return the profile name this gateway represents."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return "default"

    def _ensure_reconnect_watcher_running(self) -> None:
        """Ensure the platform reconnect watcher background task is alive.

        If the tracked reconnect watcher task has died (e.g. from exhausting
        its restart budget, or a terminal exception that _spawn_supervised
        could not recover), respawns it so platforms queued for reconnection
        are not permanently stranded. Called after queueing a retryable fatal
        error in _handle_adapter_fatal_error (#70344).
        """
        if not getattr(self, "_running", False):
            return
        task = getattr(self, "_reconnect_watcher_task", None)
        if task is not None and not task.done():
            return  # already alive
        logger.warning(
            "Reconnect watcher task is dead (done=%s) — respawning",
            task.done() if task is not None else "N/A",
        )
        self._reconnect_watcher_task = self._spawn_supervised(
            self._platform_reconnect_watcher,
            "platform_reconnect_watcher",
            on_spawn=lambda t: setattr(self, "_reconnect_watcher_task", t),
        )

    async def _platform_reconnect_watcher(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Uses exponential backoff: 30s → 60s → 120s → 240s → 300s (cap).
        Retryable failures (network/DNS blips) keep retrying at the backoff
        cap indefinitely — they self-heal once connectivity returns, so a
        transient outage never requires manual intervention. Non-retryable
        failures (bad auth, etc.) drop out of the queue immediately. The
        circuit breaker (``_pause_failed_platform`` / ``/platform pause``)
        remains available for manual operator control via ``/platform list``
        and ``/platform resume <name>``, but is no longer triggered
        automatically — auto-pausing a recovered platform was the cause of
        bots silently staying dead after a transient DNS failure.
        """
        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                # Nothing to reconnect — sleep and check again
                for _ in range(30):
                    if not self._running:
                        return
                    if self._failed_platforms:
                        break
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                info = self._failed_platforms.get(platform)
                if info is None:
                    # Removed concurrently (e.g. a manual /platform resume,
                    # or a reconnect that succeeded via a different path)
                    # between the snapshot above and this lookup. Not an
                    # error -- just nothing to do for it this pass.
                    continue
                # Skip paused platforms entirely — they need explicit
                # /platform resume to come back.
                if info.get("paused"):
                    continue
                # Long-lived retry-loop escalation (OOF-156): once a platform
                # has been continuously queued past the attention threshold,
                # flag it NEEDS_ATTENTION in runtime status so owners and
                # fleet monitoring see "this is not a blip" — a dead token,
                # revoked intent, or crash-looping sidecar otherwise presents
                # as ordinary "retrying" forever. Retries continue unchanged:
                # this is a signal, NOT a circuit breaker (auto-pause was
                # deliberately removed — see this docstring's history).
                if not info.get("attention_flagged") and _reconnect_needs_attention(info, now):
                    info["attention_flagged"] = True
                    queued_for = now - info.get("queued_at", now)
                    retrying_since_iso = (
                        datetime.now(timezone.utc) - timedelta(seconds=queued_for)
                    ).isoformat()
                    logger.warning(
                        "%s has been failing/reconnecting continuously for "
                        "%.1f hours (%d attempts) — flagging NEEDS_ATTENTION. "
                        "Retries continue, but this usually means a permanent "
                        "problem (revoked credentials, missing intents, broken "
                        "sidecar). Check `hermes status` / `/platform list`.",
                        platform.value,
                        queued_for / 3600.0,
                        info.get("attempts", 0),
                    )
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        needs_attention=True,
                        retrying_since=retrying_since_iso,
                    )
                if now < info["next_retry"]:
                    continue  # not time yet

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                # Empty-token primary configs can never reconnect; drop them so
                # multiplex setups where a secondary profile owns the bot do
                # not spin forever (#64674).
                if not _platform_has_bot_credential(platform, platform_config):
                    logger.warning(
                        "Reconnect %s: no bot credential on queued config, "
                        "removing from retry queue",
                        platform.value,
                    )
                    del self._failed_platforms[platform]
                    continue
                logger.info(
                    "Reconnecting %s (attempt %d)...",
                    platform.value, attempt,
                )

                adapter = None
                try:
                    adapter = self._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self._primary_message_handler())
                    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
                    adapter.set_session_store(self.session_store)
                    adapter.set_busy_session_handler(self._handle_active_session_busy_message)
                    _set_reaction = getattr(adapter, "set_reaction_handler", None)
                    if callable(_set_reaction):
                        _set_reaction(self._handle_reaction_event)
                    adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
                    adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
                    adapter.set_platform_event_handler(self._primary_platform_event_handler())
                    adapter._busy_input_mode = self._busy_input_mode

                    # Reconnect after an outage: preserve the platform's
                    # server-side update queue so messages sent while the bot
                    # was offline are delivered rather than dropped (#46621).
                    success = await self._connect_adapter_with_timeout(
                        adapter, platform, is_reconnect=True
                    )
                    if success:
                        self.adapters[platform] = adapter
                        self.delivery_router.adapters = self.adapters
                        del self._failed_platforms[platform]
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="connected",
                            error_code=None,
                            error_message=None,
                            needs_attention=False,
                            retrying_since=None,
                        )
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            await build_channel_directory(self.adapters)
                        except Exception:
                            pass

                        # A platform that was offline at gateway startup never
                        # got its restart-interrupted sessions auto-resumed —
                        # the startup pass skips sessions whose adapter isn't
                        # connected yet. Now that it's back, retry the
                        # auto-resume scoped to this platform so recovery
                        # doesn't silently wait for a manual user message.
                        try:
                            self._schedule_resume_pending_sessions(platform=platform)
                        except Exception:
                            logger.debug(
                                "resume-pending reschedule after %s reconnect failed",
                                platform.value,
                                exc_info=True,
                            )
                    # Check if the failure is non-retryable
                    elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        logger.warning(
                            "Reconnect %s: non-retryable error (%s), removing from retry queue",
                            platform.value, adapter.fatal_error_message,
                        )
                        # The adapter is about to be dropped from the queue
                        # without ever being installed on self.adapters, so
                        # nothing else will call disconnect() on it. We must
                        # dispose it here, otherwise the resource owners it
                        # constructed in __init__ can leak descriptors. The
                        # gateway hits the 2560-fd limit after ~12h of
                        # failed reconnects at the 300s backoff cap (#37011).
                        await _dispose_unused_adapter(adapter)
                        del self._failed_platforms[platform]
                    else:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message or "failed to reconnect",
                        )
                        backoff = _reconnect_backoff(attempt)
                        info["attempts"] = attempt
                        info["next_retry"] = time.monotonic() + backoff
                        logger.info(
                            "Reconnect %s failed, next retry in %ds",
                            platform.value, backoff,
                        )
                        # Same fd-leak concern as the non-retryable branch
                        # above: the adapter failed to connect and is being
                        # thrown away. Without an explicit dispose call, the
                        # resources it opened in __init__ stay open until
                        # the next GC pass — and aiohttp/SQLite handles
                        # don't get GC'd promptly, so 2 fds/retry leak at
                        # 300s backoff cap = ~12 fds/hour (#37011).
                        await _dispose_unused_adapter(adapter)
                        # Retryable failures (network/DNS blips) keep retrying
                        # at the backoff cap indefinitely — they self-heal once
                        # connectivity returns. We do NOT auto-pause them: a
                        # transient outage must never require manual `/platform
                        # resume` to recover. Non-retryable failures (bad auth,
                        # etc.) already drop out of the queue via the
                        # `not fatal_error_retryable` branch above, so anything
                        # reaching here is by definition retryable.
                except Exception as e:
                    if adapter is not None:
                        # An exception escaping the connect call path
                        # (DNS timeout, aiohttp server.start() crash, etc.)
                        # leaves the adapter in the same unowned state as
                        # the two branches above. Dispose so __init__
                        # resources don't accumulate while the watcher
                        # keeps retrying.
                        await _dispose_unused_adapter(adapter)
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message=str(e),
                    )
                    backoff = _reconnect_backoff(attempt)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )
                    # A raised exception during reconnect (connect timeout, DNS
                    # resolution failure, etc.) is inherently transient — keep
                    # retrying at the backoff cap rather than auto-pausing.

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self._running:
                    return
                await asyncio.sleep(1)

    def _primary_message_handler(self):
        """Return the single profile's message handler."""
        return self._handle_message

    async def _handle_gateway_platform_event(self, event: dict, source) -> None:
        """Authorize and publish one normalized adapter event to plugin hooks."""
        try:
            from hermes_cli.lifecycle import has_hook, invoke_hook

            if not has_hook("gateway_platform_event"):
                return
            if not self._is_user_authorized(source):
                return
            invoke_hook("gateway_platform_event", **event)
        except Exception:
            # Observer failures must never break the adapter's update loop.
            logger.debug("gateway_platform_event hook dispatch failed", exc_info=True)

    def _primary_platform_event_handler(self):
        return self._handle_gateway_platform_event

    @staticmethod
    def _adapter_credential_claim(
        platform: Platform, adapter: Any
    ) -> Optional[tuple]:
        """Return the exclusive credential resource claimed by an adapter."""
        fingerprint = PlatformRuntime._adapter_credential_fingerprint(adapter)
        if fingerprint is None:
            return None
        return (platform, fingerprint)

    @staticmethod
    def _adapter_credential_fingerprint(adapter: Any) -> Optional[str]:
        """Return a stable, log-safe fingerprint of an adapter's credential.

        Used only to detect two profiles claiming the same platform credential.
        Returns a salted hash (never the credential itself) of the adapter's
        primary credential, or None when no credential is discoverable (in
        which case we don't attempt conflict detection for it).
        """
        token = None
        for attr in (
            "token",
            "bot_token",
            "_token",
            "api_token",
            "_bot_token",
        ):
            val = getattr(adapter, attr, None)
            if isinstance(val, str) and val.strip():
                token = val.strip()
                break
        # Adapters may store the token on their config sub-object.
        if not token:
            cfg = getattr(adapter, "config", None)
            if cfg is not None:
                for attr in ("token", "bot_token"):
                    val = getattr(cfg, attr, None)
                    if isinstance(val, str) and val.strip():
                        token = val.strip()
                        break
        if not token:
            config = getattr(adapter, "config", None)
            val = getattr(config, "token", None)
            if isinstance(val, str) and val.strip():
                token = val.strip()
        if not token:
            return None
        import hashlib
        return hashlib.sha256(("hermes-mux:" + token).encode("utf-8")).hexdigest()[:16]

    def _create_adapter(
        self,
        platform: Platform,
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create a retained gateway transport adapter."""
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        adapter = None
        if platform == Platform.TELEGRAM:
            from gateway.platforms.telegram.adapter import (
                _build_adapter,
                check_telegram_requirements,
            )
            if not check_telegram_requirements():
                logger.warning("Telegram: python-telegram-bot is not installed")
                return None
            adapter = _build_adapter(config)
        elif platform == Platform.MATTERMOST:
            from gateway.platforms.mattermost import (
                MattermostAdapter,
                check_mattermost_requirements,
                validate_mattermost_config,
            )
            if not check_mattermost_requirements():
                logger.warning("Mattermost: aiohttp is not installed")
                return None
            if not validate_mattermost_config(config):
                logger.warning("Mattermost: URL or token is not configured")
                return None
            adapter = MattermostAdapter(config)
        if adapter is not None:
            adapter.gateway_runner = self
        return adapter

    def _make_adapter_auth_check(
        self,
        platform: Platform,
        profile_name: Optional[str] = None,
    ) -> Callable[[str, Optional[str], Optional[str]], bool]:
        """Build a platform-bound auth callback for adapter use.

        Adapters that fetch external thread context call this through
        ``BasePlatformAdapter._is_sender_authorized`` to mark non-allowlisted
        senders as unverified in LLM context, mitigating indirect prompt
        injection from third parties in shared threads/channels.

        The returned callback delegates to :meth:`_is_user_authorized` so the
        full auth chain — platform allowlists, group allowlists, pairing
        store, allow-all flags — stays the single source of truth.

        ``profile_name`` binds the callback to the secondary adapter's own
        multiplex profile, so its ``SessionSource`` resolves that profile's
        secret scope instead of falling back to the active profile.
        """
        def check(
            user_id: str,
            chat_type: Optional[str] = None,
            chat_id: Optional[str] = None,
        ) -> bool:
            if not user_id:
                return False
            source = SessionSource(
                platform=platform,
                chat_id=chat_id or "",
                chat_type=chat_type or "group",
                user_id=user_id,
                profile=profile_name,
            )
            return self._is_user_authorized(source)
        return check
