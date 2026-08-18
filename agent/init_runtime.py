"""Runtime-owned phases of AIAgent initialization."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

from agent.model_metadata import fetch_model_metadata
from hermes_cli.route_identity import normalize_route_base_url
from hermes_cli.timeouts import get_provider_request_timeout
from utils import base_url_host_matches

logger = logging.getLogger("run_agent")

# OpenRouter metadata is warmed once per process. A gateway constructs an agent
# for each message, so this guard prevents one worker thread per inbound message.
openrouter_prewarm_done = threading.Event()


def routermint_headers() -> dict[str, str]:
    """Return the User-Agent RouterMint needs to avoid Cloudflare blocks."""
    from hermes_cli import __version__ as hermes_version

    return {"User-Agent": f"HermesAgent/{hermes_version}"}


def qwen_portal_headers() -> dict[str, str]:
    """Return the default HTTP headers required by Qwen Portal."""
    import platform

    qwen_code_version = "0.14.1"
    user_agent = f"QwenCode/{qwen_code_version} ({platform.system().lower()}; {platform.machine()})"
    return {
        "User-Agent": user_agent,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": user_agent,
        "X-DashScope-AuthType": "qwen-oauth",
    }


def _moa_reference_output_allowed(agent: Any) -> bool:
    """Keep MoA display events off only the machine-readable ``-Q`` surface."""
    return not (
        getattr(agent, "platform", None) == "cli"
        and getattr(agent, "tool_progress_mode", "all") == "off"
    )

def _relay_moa_reference_event(agent: Any, event: str, **kwargs: Any) -> None:
    """Relay MoA display events while preserving the ``-Q`` stdout contract."""
    if not _moa_reference_output_allowed(agent):
        return
    cb = getattr(agent, "tool_progress_callback", None)
    if cb is None:
        return
    try:
        if event == "moa.reference":
            cb(
                "moa.reference",
                str(kwargs.get("label") or ""),
                str(kwargs.get("text") or ""),
                None,
                moa_index=kwargs.get("index"),
                moa_count=kwargs.get("count"),
            )
        elif event == "moa.aggregating":
            cb(
                "moa.aggregating",
                str(kwargs.get("aggregator") or ""),
                None,
                None,
                moa_ref_count=kwargs.get("ref_count"),
            )
    except Exception:
        pass

def _normalize_route_base_url(base_url: Any) -> str:
    """Canonicalize an endpoint URL for model-route identity comparisons."""
    return normalize_route_base_url(base_url)

def _provider_default_routes(provider: str) -> set[str]:
    """Return known exact default routes for a canonical provider id."""
    routes: set[str] = set()
    try:
        from hermes_cli.providers import HERMES_OVERLAYS, get_provider

        overlay = HERMES_OVERLAYS.get(provider)
        provider_def = get_provider(provider, allow_network=False)
        for value in (
            getattr(overlay, "base_url_override", ""),
            getattr(provider_def, "base_url", ""),
        ):
            route = _normalize_route_base_url(value)
            if route:
                routes.add(route)
    except Exception:
        pass

    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider)
        route = _normalize_route_base_url(
            getattr(profile, "base_url", "")
        )
        if route:
            routes.add(route)
    except Exception:
        pass

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.models import normalize_provider as normalize_model_provider
        from hermes_cli.providers import normalize_provider as normalize_registry_provider

        for provider_id, config in PROVIDER_REGISTRY.items():
            canonical_id = normalize_registry_provider(
                normalize_model_provider(provider_id)
            )
            if canonical_id != provider:
                continue
            route = _normalize_route_base_url(
                getattr(config, "inference_base_url", "")
            )
            if route:
                routes.add(route)
    except Exception:
        pass

    if provider == "gemini":
        routes.update(
            f"{route.rstrip('/')}/openai"
            for route in list(routes)
        )
    return routes

def _context_route_mismatch(
    configured_base_url: Any,
    active_base_url: Any,
    configured_provider: Any,
    active_provider: Any,
    *,
    already_normalized: bool = False,
) -> bool:
    """Return whether a context pin's configured route differs from runtime."""
    if already_normalized:
        configured_route = str(configured_base_url or "")
        active_route = str(active_base_url or "")
    else:
        configured_route = _normalize_route_base_url(configured_base_url)
        active_route = _normalize_route_base_url(active_base_url)
    if configured_route:
        return configured_route != active_route

    configured_provider = str(configured_provider or "").strip()
    active_provider = str(active_provider or "").strip()
    if not configured_provider:
        return False
    try:
        from hermes_cli.models import normalize_provider as normalize_model_provider

        configured_provider = normalize_model_provider(configured_provider)
        active_provider = normalize_model_provider(active_provider)
    except Exception:
        configured_provider = configured_provider.lower()
        active_provider = active_provider.lower()
    try:
        from hermes_cli.providers import normalize_provider as normalize_registry_provider

        configured_provider = normalize_registry_provider(configured_provider)
        active_provider = normalize_registry_provider(active_provider)
    except Exception:
        pass

    if active_route:
        configured_routes = _provider_default_routes(configured_provider)
        if configured_routes:
            return active_route not in configured_routes
        # Named/custom providers have no catalog default routes. An empty
        # configured URL with a matching provider identity is still the same
        # route — agent_init fills base_url from custom_providers before this
        # check, but gateway display/hygiene paths historically compared the
        # raw empty model.base_url and falsely dropped model.context_length,
        # falling through to family defaults (e.g. qwen → 131072) on Discord
        # session-reset banners while /status still showed the config pin.
        if active_provider and configured_provider == active_provider:
            return False
        return True
    return bool(
        configured_provider
        and active_provider
        and configured_provider != active_provider
    )

def _normalize_custom_provider_name(value: Any) -> str:
    """Mirror runtime normalization for a requested custom-provider identity."""
    return str(value or "").strip().lower().replace(" ", "-")

def _custom_provider_runtime_ids(value: Any) -> set[str]:
    """Return raw/menu identities that runtime accepts for a configured name."""
    normalized = _normalize_custom_provider_name(value)
    if not normalized:
        return set()
    return {normalized, f"custom:{normalized}"}

def _normalized_custom_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")

def _custom_provider_model_matches(agent_model: str, entry: Dict[str, Any]) -> bool:
    agent_model_norm = str(agent_model or "").strip().lower()
    # Multi-model entries use the current ``models`` mapping; the agent's
    # model matching any catalog key counts.
    # Without this, a provider whose `model`/`default_model` differs from the
    # session model silently fails to match and per-provider request settings
    # (extra_body, e.g. OpenAI service_tier) are dropped — billing the whole
    # session at the wrong tier (July 2026 sweeper incident: flex config
    # ignored, ~2.3x overbilling).
    models = entry.get("models")
    catalog: List[str] = []
    if isinstance(models, dict):
        catalog = [str(k).strip().lower() for k in models.keys()]
    if catalog and agent_model_norm in catalog:
        return True
    provider_model = str(entry.get("model", "") or "").strip().lower()
    if not provider_model and not catalog:
        return True
    return provider_model == agent_model_norm

def _custom_provider_extra_body_for_agent(
    *,
    provider: str,
    model: str,
    base_url: str,
    custom_providers: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    provider_norm = (provider or "").strip().lower()
    if provider_norm == "custom":
        provider_key_filter = ""
    elif provider_norm.startswith("custom:"):
        provider_key_filter = provider_norm.split(":", 1)[1].strip()
    else:
        return None

    target_url = _normalized_custom_base_url(base_url)
    if not target_url:
        return None

    fallback: Optional[Dict[str, Any]] = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        if provider_key_filter:
            entry_keys = {
                str(entry.get("provider_key", "") or "").strip().lower(),
                str(entry.get("name", "") or "").strip().lower(),
            }
            if provider_key_filter not in entry_keys:
                continue
        if _normalized_custom_base_url(entry.get("base_url")) != target_url:
            continue
        extra_body = entry.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body:
            continue
        provider_model = str(entry.get("model", "") or "").strip()
        if provider_model:
            if _custom_provider_model_matches(model, entry):
                return dict(extra_body)
        elif fallback is None:
            fallback = dict(extra_body)

    return fallback

def _merge_custom_provider_extra_body(agent, custom_providers: List[Dict[str, Any]]) -> None:
    extra_body = _custom_provider_extra_body_for_agent(
        provider=agent.provider,
        model=agent.model,
        base_url=agent.base_url,
        custom_providers=custom_providers,
    )
    if not extra_body:
        return

    overrides = dict(getattr(agent, "request_overrides", {}) or {})
    merged_extra_body = dict(extra_body)
    existing_extra_body = overrides.get("extra_body")
    if isinstance(existing_extra_body, dict):
        merged_extra_body.update(existing_extra_body)
    overrides["extra_body"] = merged_extra_body
    agent.request_overrides = overrides


def initialize_provider_route(agent, *, api_mode, provider_name, credential_pool):
    if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "codex_app_server"}:
        agent.api_mode = api_mode
    elif agent.provider == "openai-codex":
        agent.api_mode = "codex_responses"
    elif agent.provider in {"xai", "xai-oauth"}:
        agent.api_mode = "codex_responses"
    elif (provider_name is None) and (
        agent._base_url_hostname == "chatgpt.com"
        and "/backend-api/codex" in agent._base_url_lower
    ):
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
    elif (provider_name is None) and agent._base_url_hostname == "api.x.ai":
        agent.api_mode = "codex_responses"
        agent.provider = "xai"
    elif agent.provider == "anthropic" or (provider_name is None and agent._base_url_hostname == "api.anthropic.com"):
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
    elif agent._base_url_lower.rstrip("/").endswith("/anthropic"):
        # Third-party Anthropic-compatible endpoints (e.g. MiniMax, DashScope)
        # use a URL convention ending in /anthropic. Auto-detect these so the
        # Anthropic Messages API adapter is used instead of chat completions.
        agent.api_mode = "anthropic_messages"
    elif agent.provider == "bedrock" or (
        agent._base_url_hostname.startswith("bedrock-runtime.")
        and base_url_host_matches(agent._base_url_lower, "amazonaws.com")
    ):
        # AWS Bedrock — auto-detect from provider name or base URL
        # (bedrock-runtime.<region>.amazonaws.com).
        agent.api_mode = "bedrock_converse"
    elif agent.provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal is dual-wire: anthropic/* → Messages, everything else →
        # chat_completions. Callers that already pass api_mode win above;
        # this covers direct AIAgent construction without a resolved runtime.
        from hermes_cli.providers import nous_api_mode

        agent.api_mode = nous_api_mode(agent.model)
    else:
        agent.api_mode = "chat_completions"

    # Credential-pool validation runs AFTER provider auto-detection so
    # a pool scoped to e.g. "anthropic" is not rejected when the agent
    # was constructed with provider=None and an anthropic.com URL.
    # Regression from #63048 which placed this check before the
    # URL-based auto-detection block above (fixed #63425).
    if credential_pool is not None:
        try:
            from agent.credential_pool import credential_pool_matches_provider

            if not credential_pool_matches_provider(
                credential_pool,
                agent.provider,
                base_url=agent.base_url,
            ):
                agent._credential_pool = None
        except Exception:
            agent._credential_pool = None

    # Eagerly warm the transport cache so import errors surface at init,
    # not mid-conversation.  Also validates the api_mode is registered.
    try:
        agent._get_transport()
    except Exception:
        pass  # Non-fatal — transport may not exist for all modes yet

    try:
        from hermes_cli.model_normalize import (
            _AGGREGATOR_PROVIDERS,
            normalize_model_for_provider,
        )

        if agent.provider not in _AGGREGATOR_PROVIDERS:
            agent.model = normalize_model_for_provider(agent.model, agent.provider)
    except Exception:
        pass

    # GPT-5.x models usually require the Responses API path, but some
    # providers have exceptions (for example Copilot's gpt-5-mini still
    # uses chat completions). Also auto-upgrade for direct OpenAI URLs
    # (api.openai.com) since all newer tool-calling models prefer
    # Responses there. ACP runtimes are excluded: CopilotACPClient
    # handles its own routing and does not implement the Responses API
    # surface.
    # When api_mode was explicitly provided, respect it — the user
    # knows what their endpoint supports (#10473).
    # Exception: Azure OpenAI serves gpt-5.x on /chat/completions and
    # does NOT support the Responses API — skip the upgrade for Azure
    # (openai.azure.com), even though it looks OpenAI-compatible.
    if (
        api_mode is None
        and agent.api_mode == "chat_completions"
        and agent.provider != "copilot-acp"
        and not str(agent.base_url or "").lower().startswith("acp://copilot")
        and not str(agent.base_url or "").lower().startswith("acp+tcp://")
        and not agent._is_azure_openai_url()
        and (
            agent._is_direct_openai_url()
            or agent._provider_model_requires_responses_api(
                agent.model,
                provider=agent.provider,
            )
        )
    ):
        agent.api_mode = "codex_responses"
        # Invalidate the eager-warmed transport cache — api_mode changed
        # from chat_completions to codex_responses after the warm at __init__.
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()

    # Pre-warm OpenRouter model metadata cache in a background thread.
    # fetch_model_metadata() is cached for 1 hour; this avoids a blocking
    # HTTP request on the first API response when pricing is estimated.
    # Use a process-level Event so this thread is only spawned once — a new
    # AIAgent is created for every gateway request, so without the guard
    # each message leaks one OS thread and the process eventually exhausts
    # the system thread limit (RuntimeError: can't start new thread).
    if (agent.provider == "openrouter" or agent._is_openrouter_url()) and \
            not openrouter_prewarm_done.is_set():
        openrouter_prewarm_done.set()
        threading.Thread(
            target=fetch_model_metadata,
            daemon=True,
            name="openrouter-prewarm",
        ).start()


def initialize_provider_client(agent, *, api_key, base_url, fallback_providers):
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False

    # Resolve per-provider / per-model request timeout once up front so
    # every client construction path below (Anthropic native, OpenAI-wire,
    # router-based implicit auth) can apply it consistently.  Bedrock
    # Claude uses its own timeout path and is not covered here.
    _provider_timeout = get_provider_request_timeout(agent.provider, agent.model)

    if agent.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
        # Bedrock + Claude → use AnthropicBedrock SDK for full feature parity
        # (prompt caching, thinking budgets, adaptive thinking).
        _is_bedrock_anthropic = agent.provider == "bedrock"
        if _is_bedrock_anthropic:
            from agent.anthropic_adapter import build_anthropic_bedrock_client
            _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
            _br_region = _region_match.group(1) if _region_match else "us-east-1"
            agent._bedrock_region = _br_region
            agent._anthropic_client = build_anthropic_bedrock_client(_br_region)
            agent._anthropic_api_key = "aws-sdk"
            agent._anthropic_base_url = base_url
            agent._is_anthropic_oauth = False
            agent.api_key = "aws-sdk"
            agent.client = None
            agent._client_kwargs = {}
            if not agent.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock + AnthropicBedrock SDK, {_br_region})")
        else:
            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own API key.
            # Falling back would send Anthropic credentials to third-party endpoints (Fixes #1739, #minimax-401).
            _is_native_anthropic = agent.provider == "anthropic"
            effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")

            # MiniMax OAuth issues short-lived (~15-min) access tokens. The
            # Anthropic SDK caches ``api_key`` as a static string at client
            # construction time, so a session that resolves the bearer once
            # at startup will keep sending the same token until MiniMax
            # returns 401 mid-session. Swap the static string for a callable
            # token provider — ``build_anthropic_client`` recognizes the
            # callable and installs an httpx event hook that mints a fresh
            # bearer per outbound request (re-reading auth.json so a refresh
            # persisted by another process is visible immediately).
            # The cached refresh path is a no-op when the token still has
            # ``MINIMAX_OAUTH_REFRESH_SKEW_SECONDS`` of life left, so steady-
            # state cost is one file read + one timestamp compare per request.
            if agent.provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
                try:
                    from hermes_cli.auth import build_minimax_oauth_token_provider
                    effective_key = build_minimax_oauth_token_provider()
                except Exception as _mm_exc:  # noqa: BLE001 — never block startup on this
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "MiniMax OAuth: failed to install per-request token provider "
                        "(%s); falling back to static bearer that will expire ~15min in.",
                        _mm_exc,
                    )

            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = base_url
            # Only mark the session as OAuth-authenticated when the token
            # genuinely belongs to native Anthropic.  Third-party providers
            # (MiniMax, Kimi, GLM, LiteLLM proxies) that accept the
            # Anthropic protocol must never trip OAuth code paths — doing
            # so injects Claude-Code identity headers and system prompts
            # that cause 401/403 on their endpoints.  Guards #1739 and
            # the third-party identity-injection bug.
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            agent._is_anthropic_oauth = _is_oat(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
            agent._anthropic_client = build_anthropic_client(effective_key, base_url, timeout=_provider_timeout)
            # No OpenAI client needed for Anthropic mode
            agent.client = None
            agent._client_kwargs = {}
            if not agent.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {agent.model} (Anthropic native)")
                # ``effective_key`` may be a callable Entra ID bearer
                # provider for Azure Foundry anthropic_messages mode.
                # The Anthropic adapter installs an httpx event hook
                # that mints a fresh JWT per request — we never
                # invoke or inspect the callable in the banner.
                from agent.azure_identity_adapter import is_token_provider

                if is_token_provider(effective_key):
                    print("🔑 Using credentials: Microsoft Entra ID")
                elif isinstance(effective_key, str) and len(effective_key) > 12:
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
    elif agent.provider == "moa":
        from agent.moa_loop import build_moa_facade
        agent.api_mode = "chat_completions"

        # build_moa_facade wires the reference relay that routes
        # reference-model outputs to the agent's tool_progress_callback so
        # every surface that already consumes it (CLI spinner/scrollback, TUI,
        # desktop, gateway) can show each reference's answer as a labelled
        # block before the aggregator acts. The facade emits "moa.reference",
        # "moa.progress", "moa.phase", and "moa.aggregating" events, forwarded
        # through the same callback the tool lifecycle uses. Best-effort and
        # cache-safe — display-only events, they never touch the message
        # history. The factory is shared with the fallback-restore/recovery
        # paths so a restored facade keeps emitting these events (#53802).
        agent.client = build_moa_facade(agent, agent.model)
        agent._client_kwargs = {}
        agent.api_key = api_key or "moa-virtual-provider"
        agent.base_url = "moa://local"
        if not agent.quiet_mode:
            print(f"🤖 AI Agent initialized with MoA preset: {agent.model}")
    elif agent.api_mode == "bedrock_converse":
        # AWS Bedrock — uses boto3 directly, no OpenAI client needed.
        # Region is extracted from the base_url or defaults to us-east-1.
        _region_match = re.search(r"bedrock-runtime\.([a-z0-9-]+)\.", base_url or "")
        agent._bedrock_region = _region_match.group(1) if _region_match else "us-east-1"
        # Guardrail config — read from config.yaml at init time.
        agent._bedrock_guardrail_config = None
        try:
            from hermes_cli.config import load_config_readonly as _load_br_cfg
            _gr = _load_br_cfg().get("bedrock", {}).get("guardrail", {})
            if _gr.get("guardrail_identifier") and _gr.get("guardrail_version"):
                agent._bedrock_guardrail_config = {
                    "guardrailIdentifier": _gr["guardrail_identifier"],
                    "guardrailVersion": _gr["guardrail_version"],
                }
                if _gr.get("stream_processing_mode"):
                    agent._bedrock_guardrail_config["streamProcessingMode"] = _gr["stream_processing_mode"]
                if _gr.get("trace"):
                    agent._bedrock_guardrail_config["trace"] = _gr["trace"]
        except Exception:
            pass
        agent.client = None
        agent._client_kwargs = {}
        if not agent.quiet_mode:
            _gr_label = " + Guardrails" if agent._bedrock_guardrail_config else ""
            print(f"🤖 AI Agent initialized with model: {agent.model} (AWS Bedrock, {agent._bedrock_region}{_gr_label})")
    else:
        if api_key and base_url:
            # Explicit credentials from CLI/gateway — construct directly.
            # The runtime provider resolver already handled auth for us.
            # Extract query params (e.g. Azure api-version) from base_url
            # and pass via default_query to prevent loss during SDK URL
            # joining (httpx drops query string when joining paths).
            _parsed_url = urlparse(base_url)
            if _parsed_url.query:
                _clean_url = urlunparse(_parsed_url._replace(query=""))
                _query_params = {
                    k: v[0] for k, v in parse_qs(_parsed_url.query).items()
                }
                client_kwargs = {
                    "api_key": api_key,
                    "base_url": _clean_url,
                    "default_query": _query_params,
                }
            else:
                client_kwargs = {"api_key": api_key, "base_url": base_url}
            if _provider_timeout is not None:
                client_kwargs["timeout"] = _provider_timeout
            if agent.provider == "copilot-acp":
                client_kwargs["command"] = agent.acp_command
                client_kwargs["args"] = agent.acp_args
            effective_base = base_url
            if base_url_host_matches(effective_base, "openrouter.ai"):
                from agent.auxiliary_client import build_or_headers
                client_kwargs["default_headers"] = build_or_headers()
            elif base_url_host_matches(effective_base, "integrate.api.nvidia.com"):
                from agent.auxiliary_client import build_nvidia_nim_headers
                client_kwargs["default_headers"] = build_nvidia_nim_headers(effective_base)
            elif base_url_host_matches(effective_base, "api.routermint.com"):
                client_kwargs["default_headers"] = routermint_headers()
            elif base_url_host_matches(effective_base, "githubcopilot.com"):
                from hermes_cli.models import copilot_default_headers

                client_kwargs["default_headers"] = copilot_default_headers()
            elif base_url_host_matches(effective_base, "api.kimi.com"):
                client_kwargs["default_headers"] = {
                    "User-Agent": "claude-code/0.1.0",
                }
            elif base_url_host_matches(effective_base, "portal.qwen.ai"):
                client_kwargs["default_headers"] = qwen_portal_headers()
            elif base_url_host_matches(effective_base, "chatgpt.com"):
                from agent.auxiliary_client import _codex_cloudflare_headers
                client_kwargs["default_headers"] = _codex_cloudflare_headers(api_key)
            elif base_url_host_matches(effective_base, "x.ai"):
                from tools.xai_http import hermes_xai_default_headers

                client_kwargs["default_headers"] = hermes_xai_default_headers()
            elif "default_headers" not in client_kwargs:
                # Fall back to profile.default_headers for providers that
                # declare custom headers (e.g. Vercel AI Gateway attribution,
                # Kimi User-Agent on non-kimi.com endpoints).
                try:
                    from providers import get_provider_profile as _gpf
                    _ph = _gpf(agent.provider)
                    if _ph and _ph.default_headers:
                        client_kwargs["default_headers"] = dict(_ph.default_headers)
                except Exception:
                    pass
        else:
            # No explicit creds — use the centralized provider router
            from agent.auxiliary_client import resolve_provider_client
            _routed_client, _ = resolve_provider_client(
                agent.provider or "auto", model=agent.model, raw_codex=True)
            if _routed_client is not None:
                client_kwargs = {
                    "api_key": _routed_client.api_key,
                    "base_url": str(_routed_client.base_url),
                }
                if _provider_timeout is not None:
                    client_kwargs["timeout"] = _provider_timeout
                # Preserve provider-specific headers the router set.  The
                # OpenAI SDK stores caller-provided default_headers in
                # _custom_headers; older/mocked clients may expose
                # _default_headers instead.
                _routed_headers = getattr(_routed_client, "_custom_headers", None)
                if not _routed_headers:
                    _routed_headers = getattr(_routed_client, "default_headers", None)
                if not _routed_headers:
                    _routed_headers = getattr(_routed_client, "_default_headers", None)
                if _routed_headers:
                    client_kwargs["default_headers"] = dict(_routed_headers)
            else:
                # When the user explicitly chose a non-OpenRouter provider
                # but no credentials were found, fail fast with a clear
                # message instead of silently routing through OpenRouter.
                _explicit = (agent.provider or "").strip().lower()
                if _explicit and _explicit not in {"auto", "openrouter", "custom"}:
                    # Look up the actual env var name from the provider
                    # config — some providers use non-standard names
                    # (e.g. alibaba → DASHSCOPE_API_KEY, not ALIBABA_API_KEY).
                    _env_hint = f"{_explicit.upper()}_API_KEY"
                    try:
                        from hermes_cli.auth import PROVIDER_REGISTRY
                        _pcfg = PROVIDER_REGISTRY.get(_explicit)
                        if _pcfg and _pcfg.api_key_env_vars:
                            _env_hint = _pcfg.api_key_env_vars[0]
                    except Exception:
                        pass
                    # --- Init-time fallback (#17929) ---
                    _fb_entries = []
                    _fb_entries = [
                        f for f in (fallback_providers or [])
                        if isinstance(f, dict) and f.get("provider") and f.get("model")
                    ]
                    _fb_resolved = False
                    for _fb in _fb_entries:
                        try:
                            from hermes_cli.fallback_config import resolve_entry_api_key
                            _fb_explicit_key = resolve_entry_api_key(_fb)
                            _fb_client, _fb_model = resolve_provider_client(
                                _fb["provider"], model=_fb["model"], raw_codex=True,
                                explicit_base_url=_fb.get("base_url"),
                                explicit_api_key=_fb_explicit_key,
                            )
                        except Exception as _fb_exc:
                            logger.debug(
                                "Init-time fallback entry %s failed: %s",
                                _fb.get("provider"), _fb_exc,
                            )
                            continue
                        if _fb_client is not None:
                            agent.provider = _fb["provider"]
                            agent.model = _fb_model or _fb["model"]
                            agent._fallback_activated = True
                            client_kwargs = {
                                "api_key": _fb_client.api_key,
                                "base_url": str(_fb_client.base_url),
                            }
                            if _provider_timeout is not None:
                                client_kwargs["timeout"] = _provider_timeout
                            _fb_headers = getattr(_fb_client, "_custom_headers", None)
                            if not _fb_headers:
                                _fb_headers = getattr(_fb_client, "default_headers", None)
                            if not _fb_headers:
                                _fb_headers = getattr(_fb_client, "_default_headers", None)
                            if _fb_headers:
                                client_kwargs["default_headers"] = dict(_fb_headers)
                            _fb_resolved = True
                            break
                    if not _fb_resolved:
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_env_hint} environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                if not getattr(agent, "_fallback_activated", False):
                    # No provider configured — reject with a clear message.
                    raise RuntimeError(
                        "No LLM provider configured. Run `hermes model` to "
                        "select a provider, or run `hermes setup` for first-time "
                        "configuration."
                    )

        agent._client_kwargs = client_kwargs  # stored for rebuilding after interrupt

        # Enable fine-grained tool streaming for Claude on OpenRouter.
        # Without this, Anthropic buffers the entire tool call and goes
        # silent for minutes while thinking — OpenRouter's upstream proxy
        # times out during the silence.  The beta header makes Anthropic
        # stream tool call arguments token-by-token, keeping the
        # connection alive.
        _effective_base = str(client_kwargs.get("base_url", "")).lower()
        if base_url_host_matches(_effective_base, "openrouter.ai") and "claude" in (agent.model or "").lower():
            headers = client_kwargs.get("default_headers") or {}
            existing_beta = headers.get("x-anthropic-beta", "")
            _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
            if _FINE_GRAINED not in existing_beta:
                if existing_beta:
                    headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                else:
                    headers["x-anthropic-beta"] = _FINE_GRAINED
                client_kwargs["default_headers"] = headers

        # User-configured request headers (model.default_headers in
        # config.yaml) override provider/SDK defaults. Lets custom
        # OpenAI-compatible endpoints behind a gateway/WAF that rejects the
        # OpenAI SDK's identifying headers swap in a plain User-Agent. (#40033)
        # client_kwargs is the same dict object as agent._client_kwargs, so
        # this mutation is reflected in the client built just below.
        agent._apply_user_default_headers()

        try:
            from hermes_cli.config import (
                apply_custom_provider_extra_headers_to_client_kwargs,
                apply_custom_provider_tls_to_client_kwargs,
                get_compatible_custom_providers,
                load_config,
            )

            _cp_config = load_config()
            _cp_entries = get_compatible_custom_providers(_cp_config)
            _cp_base_url = str(client_kwargs.get("base_url") or agent.base_url or "")
            apply_custom_provider_tls_to_client_kwargs(
                client_kwargs,
                _cp_base_url,
                _cp_entries,
            )
            # Per-provider extra HTTP headers (providers.<name>.extra_headers /
            # custom_providers[].extra_headers) — proxies, gateways, custom
            # auth. Applied last so the most specific config level wins.
            # SECURITY: values may carry credentials — never log them.
            apply_custom_provider_extra_headers_to_client_kwargs(
                client_kwargs,
                _cp_base_url,
                _cp_entries,
            )
        except Exception:
            logger.debug("custom-provider TLS resolution skipped", exc_info=True)

        agent.api_key = client_kwargs.get("api_key", "")
        agent.base_url = client_kwargs.get("base_url", agent.base_url)
        try:
            from agent.ssl_guard import verify_ca_bundle_with_fallback

            verify_ca_bundle_with_fallback()
            agent.client = agent._create_openai_client(client_kwargs, reason="agent_init", shared=True)
            if not agent.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {agent.model}")
                if base_url:
                    print(f"🔗 Using custom base URL: {base_url}")
                # ``api_key`` may be a callable Entra ID bearer
                # provider (Azure Foundry). The OpenAI SDK mints a
                # fresh JWT per request internally — the banner
                # never invokes or inspects the callable.
                from agent.azure_identity_adapter import is_token_provider

                key_used = client_kwargs.get("api_key", "none")
                if is_token_provider(key_used):
                    print("🔑 Using credentials: Microsoft Entra ID")
                elif isinstance(key_used, str) and key_used and key_used != "dummy-key" and len(key_used) > 12:
                    print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                else:
                    print("⚠️  Warning: API key appears invalid or missing")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    # Keep a stable identity for the pool entry that supplied this runtime.
    # OAuth refreshes can replace the runtime token before a failed request is
    # recovered, so the mutable API-key value alone cannot reliably attribute
    # the failure to its source entry.
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    sync_credential_pool_entry_id(agent)

    # Provider fallback chain — ordered list of backup providers tried
    # when the primary is exhausted (rate-limit, overload, connection
    # failure). The constructor contract is a normalized ordered list.
    agent._fallback_chain = [
        f for f in (fallback_providers or [])
        if isinstance(f, dict) and f.get("provider") and f.get("model")
    ]
    agent._fallback_index = 0
    agent._fallback_activated = getattr(agent, "_fallback_activated", False)
    if agent._fallback_chain and not agent.quiet_mode:
        if len(agent._fallback_chain) == 1:
            fb = agent._fallback_chain[0]
            print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
        else:
            print(f"🔄 Fallback chain ({len(agent._fallback_chain)} providers): " +
                  " → ".join(f"{f['model']} ({f['provider']})" for f in agent._fallback_chain))
