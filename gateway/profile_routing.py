"""Profile-based routing for the gateway with hierarchical matching.

Allows a single Hermes instance to route specific workspaces/channels/threads
to different profiles — each with their own model, tools, memory, and persona.

Matching priority (most specific first):
  1. platform + chat_id + thread_id (exact thread)  — specificity 14
  2. platform + chat_id (channel route)             — specificity 6
  3. platform + scope_id (workspace/server route)   — specificity 2
  4. No match                                       → default profile

Parent-chain matching:
For threaded transports, ``parent_chat_id`` carries the direct parent.
Routes keyed on a channel match both direct messages and messages in
any thread/post whose parent is that channel.

Configuration (config.yaml):

    gateway:
      profile_routes:
        - name: workspace-default
          platform: mattermost
          scope_id: "YOUR_WORKSPACE_ID"
          profile: workspace-profile

        - name: special-channel
          platform: mattermost
          scope_id: "YOUR_WORKSPACE_ID"
          chat_id: "YOUR_CHANNEL_ID"
          profile: channel-profile

        - name: thread-route
          platform: mattermost
          chat_id: "YOUR_CHANNEL_ID"
          thread_id: "YOUR_THREAD_ID"
          profile: thread-profile
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

from contextlib import contextmanager
from pathlib import Path

from gateway.config import GatewayConfig, load_gateway_config
from hermes_constants import get_hermes_home

runtime_logger = logging.getLogger("gateway.run")


class ProfileRouteRejected(RuntimeError):
    """An explicit route matched a profile this gateway does not serve."""


@dataclass(frozen=True)
class ProfileRoute:
    """A single routing rule that maps a platform scope to a profile."""

    name: str
    platform: str
    profile: str
    scope_id: Optional[str] = None
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    enabled: bool = True

    @property
    def specificity(self) -> int:
        """Higher value = more specific match."""
        s = 0
        if self.scope_id:
            s += 2
        if self.chat_id:
            s += 4
        if self.thread_id:
            s += 8
        return s

    def matches(
        self,
        platform: str,
        scope_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_chat_id: Optional[str] = None,
    ) -> bool:
        """Return True if this route matches the given source fields.

        All configured discriminators are matched conjunctively (AND): every
        discriminator that the route declares must hold. ``chat_id`` supports
        hierarchical matching for threads:
        - Direct channel match: chat_id == route.chat_id
        - Thread in channel: parent_chat_id == route.chat_id
        A route declaring both ``scope_id`` and ``chat_id`` requires both to
        match (a chat match alone does not satisfy a scope constraint).
        """
        if not self.enabled:
            return False
        if self.platform != platform:
            return False
        if self.thread_id and self.thread_id != thread_id:
            return False
        if self.chat_id and self.chat_id != chat_id and self.chat_id != parent_chat_id:
            return False
        if self.scope_id and self.scope_id != scope_id:
            return False
        return True


def parse_profile_routes(raw: Optional[List[Dict[str, Any]]]) -> List[ProfileRoute]:
    """Parse profile_routes from config.yaml into ProfileRoute objects.

    Returns routes sorted by specificity (most specific first).
    """
    if not raw:
        return []
    routes: List[ProfileRoute] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "")
        platform = entry.get("platform", "")
        profile = entry.get("profile", "")
        if not platform or not profile:
            logger.warning(
                "Skipping profile route %s: missing platform or profile",
                name,
            )
            continue
        # Validate profile name to prevent path traversal. Lazy import avoids a
        # circular dependency at module load time.
        try:
            from hermes_cli.profiles import (
                normalize_profile_name,
                validate_profile_name,
            )
            profile = normalize_profile_name(profile)
            validate_profile_name(profile)
        except (ValueError, ImportError):
            logger.warning("Skipping profile route %s: invalid profile name %r", name, profile)
            continue
        routes.append(
            ProfileRoute(
                name=name,
                platform=platform,
                profile=profile,
                scope_id=entry.get("scope_id"),
                chat_id=entry.get("chat_id"),
                thread_id=entry.get("thread_id"),
                enabled=entry.get("enabled", True),
            )
        )
    # Sort: most specific first so the first match wins.
    routes.sort(key=lambda r: r.specificity, reverse=True)
    logger.debug("Loaded %d profile routes (most-specific-first)", len(routes))
    return routes


def match_profile_route(
    routes: List[ProfileRoute],
    platform: str,
    scope_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    parent_chat_id: Optional[str] = None,
) -> Optional[ProfileRoute]:
    """Return the best-matching route, or None for no match."""
    for route in routes:
        if route.matches(platform, scope_id=scope_id, chat_id=chat_id, thread_id=thread_id, parent_chat_id=parent_chat_id):
            return route
    return None

class MultiplexConfigError(RuntimeError):
    """A profile multiplexer config is invalid.

    Distinct from a transient adapter-connect failure: a config error means the
    operator must fix config.yaml. Fatal configuration errors propagate to the
    startup guard instead of being treated as retryable adapter noise.
    """

class SecondaryPortBindingConfigError(MultiplexConfigError):
    """A secondary profile conflicts with the multiplexer's shared listener."""

def _multiplex_profile_homes(config: object) -> list[tuple[str, "Path"]]:
    """Return the authoritative profile set for one multiplex gateway config."""
    from hermes_cli.profiles import profiles_to_serve

    return list(
        profiles_to_serve(
            multiplex=True,
            profile_allowlist=getattr(config, "multiplex_profile_allowlist", None),
        )
    )

@contextmanager
def _profile_runtime_scope(profile_home: "Path"):
    """Scope config/skills/memory AND credentials to a profile for one turn.

    Combines the two seams the multiplexer needs:
      1. ``set_hermes_home_override`` — redirects ``get_hermes_home()`` (config,
         skills, memory, SOUL, sessions) to the profile's home. Contextvar, so
         it propagates into the agent worker thread via ``copy_context()``.
      2. ``set_secret_scope`` — installs the profile's ``.env`` secrets as the
         authoritative credential source, so ``get_secret`` reads this profile's
         keys and never the process-global ``os.environ`` (which in a
         multiplexer may hold another profile's values).

    Only used on the multiplexed inbound path. Single-profile gateways never
    enter this scope, so their behavior is unchanged. Loading the profile's
    ``.env`` here does NOT mutate ``os.environ`` — ``build_profile_secret_scope``
    returns an isolated dict — which is what keeps subprocesses (MCP, kanban)
    from inheriting cross-profile secrets.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import (
        build_profile_secret_scope,
        set_secret_scope,
        reset_secret_scope,
    )
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(str(profile_home))
    hydrate_profile_secret_sources(Path(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

def load_gateway_config_for_runner() -> "GatewayConfig":
    """Load gateway config for the process-level GatewayRunner.

    When ``gateway.multiplex_profiles`` is off, this is identical to
    ``load_gateway_config()`` (legacy single-profile path).

    When multiplexing is on, reload under the default/active profile's
    ``_profile_runtime_scope`` so platform tokens in that profile's ``.env``
    resolve through the secret scope — the same path secondary profiles use
    in ``_start_one_profile_adapters``. Without this, primary startup calls
    ``load_gateway_config()`` unscoped: ``_getenv`` falls through to
    ``os.environ``, which often has no ``TELEGRAM_BOT_TOKEN`` once the token
    lives only under ``profiles/<name>/.env`` (#64674).

    Single-profile gateways never set ``multiplex_profiles``, so they keep the
    unscoped load and are unaffected.
    """
    cfg = load_gateway_config()
    if not getattr(cfg, "multiplex_profiles", False):
        return cfg
    try:
        home = get_hermes_home()
    except Exception:
        return cfg
    try:
        with _profile_runtime_scope(Path(home)):
            return load_gateway_config()
    except Exception:
        runtime_logger.debug(
            "multiplex default-scope config reload failed; using unscoped load",
            exc_info=True,
        )
        return cfg
