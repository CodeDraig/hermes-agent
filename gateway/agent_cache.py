"""Ownership and typed entries for retained per-session gateway agents."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import agent.lifecycle as lifecycle
from gateway.session_state import AGENT_PENDING

logger = logging.getLogger("gateway.run")


@dataclass
class AgentCacheEntry:
    agent: Any
    signature: str
    message_count: int | None
    session_id: str | None


class GatewayAgentCache:
    """Own retained agents, their coherence snapshots, and eviction policy."""

    def __init__(
        self,
        *,
        sessions: Any,
        session_store: Any,
        session_db: Callable[[], Any],
        load_config: Callable[[], dict],
    ) -> None:
        self.entries: OrderedDict[str, AgentCacheEntry] = OrderedDict()
        self.lock = threading.Lock()
        self.bounds: Any = None
        self._sessions = sessions
        self._session_store = session_store
        self._session_db = session_db
        self._load_config = load_config

    async def refresh_message_count(
        self, session_key: str, session_id: str | None
    ) -> None:
        db = self._session_db()
        if db is None or not session_id:
            return
        try:
            row = await db.get_session(session_id)
            live = row.get("message_count", 0) if row else None
        except Exception:
            return
        if live is None:
            return
        with self.lock:
            cached = self.entries.get(session_key)
            if not cached or cached.agent is AGENT_PENDING:
                return
            if cached.session_id != session_id:
                return
            cached.message_count = live

    def evict(self, session_key: str) -> None:
        state = self._sessions.peek(session_key)
        if state is not None:
            state.conversation.ephemeral_pin = None
        with self.lock:
            evicted = self.entries.pop(session_key, None)
        agent = evicted.agent if evicted else None
        if agent is None or agent is AGENT_PENDING:
            return
        running_ids = {
            id(active)
            for _, active in self._sessions.running_items()
            if active is not None and active is not AGENT_PENDING
        }
        if id(agent) in running_ids:
            return
        try:
            threading.Thread(
                target=self.release_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-evict-{str(session_key)[:24]}",
            ).start()
        except Exception:
            self.release_soft(agent)

    @staticmethod
    def init_for_turn(agent: Any, interrupt_depth: int) -> None:
        if interrupt_depth == 0:
            from agent.session_activity import ActivityProvenance

            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
            agent._last_activity_provenance = ActivityProvenance.UNKNOWN
            if hasattr(agent, "_last_flushed_db_idx"):
                agent._last_flushed_db_idx = 0
        agent._api_call_count = 0

    def _commit_memory_before_soft_evict(self, agent: Any, key: str) -> None:
        if agent is None or not hasattr(agent, "commit_memory_session"):
            return
        if getattr(agent, "_memory_manager", None) is None:
            return
        try:
            self._session_store._ensure_loaded()
            entry = self._session_store._entries.get(key)
            if entry is None:
                return
            if not self._session_store.is_session_finalizable(entry):
                return
            if self._session_store._is_session_expired(entry):
                return
            messages = getattr(agent, "_session_messages", None)
            lifecycle.commit_memory_session(
                agent, messages if isinstance(messages, list) else None
            )
            logger.debug(
                "Committed memory before soft-evicting finalizable session=%s",
                key,
            )
        except Exception as exc:
            logger.debug("Pre-evict memory commit failed for %s: %s", key, exc)

    def _commit_then_release_soft(self, agent: Any, key: str) -> None:
        self._commit_memory_before_soft_evict(agent, key)
        self.release_soft(agent)

    @staticmethod
    def release_soft(agent: Any) -> None:
        if agent is None:
            return
        try:
            lifecycle.release_clients(agent)
        except Exception:
            pass
        if hasattr(agent, "_session_messages"):
            agent._session_messages = []
        if hasattr(agent, "_db_flush_scan_prefix"):
            agent._db_flush_scan_prefix = None

    def _resolved_bounds(self):
        if self.bounds is None:
            from gateway.agent_cache_pressure import resolve_agent_cache_bounds

            try:
                self.bounds = resolve_agent_cache_bounds(self._load_config())
            except Exception as exc:
                logger.debug("Agent cache bounds config read failed: %s", exc)
                self.bounds = resolve_agent_cache_bounds({})
        return self.bounds

    def cap(self) -> int:
        configured = self._resolved_bounds().max_size
        return configured if configured else 128

    def idle_ttl(self) -> float:
        configured = self._resolved_bounds().idle_ttl_secs
        return configured if configured else 3600.0

    def sweep_under_pressure(self) -> int:
        from gateway.agent_cache_pressure import (
            plan_pressure_evictions,
            read_anon_rss_mb,
            transcript_persistence_caught_up,
        )

        bounds = self._resolved_bounds()
        if not bounds.memory_high_mb or not self.entries:
            return 0
        rss_mb = read_anon_rss_mb()
        if rss_mb is None or rss_mb < bounds.memory_high_mb:
            return 0
        running_ids = {
            id(agent)
            for _, agent in self._sessions.running_items()
            if agent is not None and agent is not AGENT_PENDING
        }

        def is_evictable(_key: str, agent: Any) -> bool:
            return (
                agent is not None
                and agent is not AGENT_PENDING
                and id(agent) not in running_ids
                and transcript_persistence_caught_up(agent)
            )

        with self.lock:
            ordered = [(key, entry.agent) for key, entry in self.entries.items()]
            plan = plan_pressure_evictions(
                ordered,
                is_evictable=is_evictable,
                max_evictions=bounds.max_evictions_per_pass,
                protect_recent=bounds.protect_recent,
            )
            for key, _ in plan:
                self.entries.pop(key, None)
        if not plan:
            mid_turn = sum(
                1 for _, agent in ordered if agent is not None and id(agent) in running_ids
            )
            unflushed = sum(
                1
                for _, agent in ordered
                if agent is not None
                and agent is not AGENT_PENDING
                and id(agent) not in running_ids
                and not transcript_persistence_caught_up(agent)
            )
            logger.warning(
                "Agent cache pressure: anon RSS %dMB over budget %dMB but no "
                "evictable session (%d cached, %d mid-turn, %d blocked on "
                "un-flushed persistence)%s",
                rss_mb,
                bounds.memory_high_mb,
                len(ordered),
                mid_turn,
                unflushed,
                (
                    " — transcripts are not reaching the session DB; the "
                    "memory valve cannot shed sessions until they persist."
                    if unflushed and not mid_turn
                    else " — memory will keep climbing until those turns finish."
                ),
            )
            return 0
        count = len(plan)
        logger.warning(
            "Agent cache pressure: anon RSS %dMB over budget %dMB — evicting "
            "%d LRU session(s): %s",
            rss_mb,
            bounds.memory_high_mb,
            count,
            ", ".join(key for key, _ in plan),
        )
        try:
            threading.Thread(
                target=self._release_pressure_batch,
                args=(plan,),
                daemon=True,
                name="agent-cache-pressure",
            ).start()
        except Exception:
            self._release_pressure_batch(plan)
        return count

    def _release_pressure_batch(self, plan: list[tuple[str, Any]]) -> None:
        while plan:
            key, agent = plan.pop(0)
            try:
                self._commit_then_release_soft(agent, key)
            except Exception as exc:
                logger.debug("Pressure release failed for %s: %s", key, exc)
            del agent
        try:
            from hermes_cli.mem_trim import trim_memory

            trim_memory(force=True, reason="agent_cache_pressure")
        except Exception:
            pass

    def enforce_cap(self) -> None:
        running_ids = {
            id(agent)
            for _, agent in self._sessions.running_items()
            if agent is not None and agent is not AGENT_PENDING
        }
        cap = self.cap()
        excess = max(0, len(self.entries) - cap)
        plan: list[tuple[str, Any]] = []
        for key in list(self.entries)[:excess]:
            entry = self.entries.get(key)
            agent = entry.agent if entry else None
            if agent is not None and id(agent) in running_ids:
                continue
            plan.append((key, agent))
        for key, _ in plan:
            self.entries.pop(key, None)
        remaining = len(self.entries) - cap
        if remaining > 0:
            logger.warning(
                "Agent cache over cap (%d > %d); %d excess slot(s) held by "
                "mid-turn agents — will re-check on next insert.",
                len(self.entries),
                cap,
                remaining,
            )
        for key, agent in plan:
            logger.info(
                "Agent cache at cap; evicting LRU session=%s (cache_size=%d)",
                key,
                len(self.entries),
            )
            if agent is not None:
                threading.Thread(
                    target=self._commit_then_release_soft,
                    args=(agent, key),
                    daemon=True,
                    name=f"agent-cache-evict-{key[:24]}",
                ).start()

    def sweep_idle(self) -> int:
        now = time.time()
        idle_ttl = self.idle_ttl()
        running_ids = {
            id(agent)
            for _, agent in self._sessions.running_items()
            if agent is not None and agent is not AGENT_PENDING
        }
        to_evict: list[tuple[str, Any]] = []
        with self.lock:
            for key, entry in list(self.entries.items()):
                agent = entry.agent
                if agent is None or id(agent) in running_ids:
                    continue
                last_activity = getattr(agent, "_last_activity_ts", None)
                if last_activity is None or now - last_activity <= idle_ttl:
                    continue
                session_entry = None
                try:
                    self._session_store._ensure_loaded()
                    session_entry = self._session_store._entries.get(key)
                except Exception:
                    pass
                if (
                    session_entry is not None
                    and self._session_store.is_session_finalizable(session_entry)
                    and not self._session_store._is_session_expired(session_entry)
                ):
                    continue
                to_evict.append((key, agent))
            for key, _ in to_evict:
                self.entries.pop(key, None)
        for key, agent in to_evict:
            logger.info(
                "Agent cache idle-TTL evict: session=%s (idle=%.0fs)",
                key,
                now - getattr(agent, "_last_activity_ts", now),
            )
            threading.Thread(
                target=self.release_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-cache-idle-{key[:24]}",
            ).start()
        return len(to_evict)
