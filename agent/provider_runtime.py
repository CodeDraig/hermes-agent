"""Responsibility-owned agent provider runtime behavior."""
import asyncio
import base64
import copy
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hermes_constants import get_hermes_home
from agent.credential_pool import (
    STATUS_EXHAUSTED,
    credential_pool_matches_provider,
    pool_may_recover_from_rate_limit,
)
from agent.error_classifier import FailoverReason
from agent.model_metadata import is_local_endpoint
from agent.process_bootstrap import (
    OpenAI,
    _get_proxy_for_base_url,
)
from agent.tool_dispatch_helpers import (
    _is_multimodal_tool_result,
    _multimodal_text_summary,
)
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)
from utils import base_url_host_matches, base_url_hostname, env_float, model_forces_max_completion_tokens


logger = logging.getLogger(__name__)

_MAX_AUTH_REFRESH_ATTEMPTS = 2
_TRANSIENT_TRANSPORT_ERRORS = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "APIConnectionError",
        "APITimeoutError",
    }
)
VALID_CACHE_TTLS = ("5m", "1h")

try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent.parent / ".env"
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")

_REQUEST_CLIENT_REUSE_REASONS = frozenset({
    "request_complete",
    "stream_request_complete",
})

_MAX_DATA_URL_BASE64_BYTES = 20 * 1024 * 1024

def set_base_url(agent, value: str) -> None:
    """Set the endpoint and its cached normalized forms together."""
    agent.base_url = value
    agent._base_url_lower = value.lower() if value else ""
    agent._base_url_hostname = base_url_hostname(value)

def _effective_lmstudio_context_length(
    config_context_length: Optional[int],
    runtime_context_length: Any,
) -> Optional[int]:
    """Return a safe context budget from explicit intent and verified runtime."""
    explicit = (
        config_context_length
        if isinstance(config_context_length, int)
        and not isinstance(config_context_length, bool)
        and config_context_length > 0
        else None
    )
    runtime_value = getattr(runtime_context_length, "context_length", runtime_context_length)
    runtime = (
        runtime_value
        if isinstance(runtime_value, int)
        and not isinstance(runtime_value, bool)
        and runtime_value > 0
        else None
    )
    if bool(getattr(runtime_context_length, "rejected", False)) or (
        bool(getattr(runtime_context_length, "load_attempted", False))
        and runtime is None
    ):
        return None
    if runtime is not None and explicit is not None:
        return min(runtime, explicit)
    return runtime if runtime is not None else explicit

def _lmstudio_load_was_unverified(load_result: Any) -> bool:
    """Return true when a management load was rejected or unverifiable."""
    return bool(getattr(load_result, "rejected", False)) or (
        bool(getattr(load_result, "load_attempted", False))
        and getattr(load_result, "context_length", None) is None
    )

def _ensure_lmstudio_runtime_loaded(
    self,
    config_context_length: Optional[int] = None,
) -> Any:
    """Preload LM Studio unless configured to rely on JIT loading."""
    if (self.provider or "").strip().lower() != "lmstudio":
        return None
    if (getattr(self, "lmstudio_load_mode", "explicit") or "explicit").strip().lower() == "jit":
        logger.debug("LM Studio explicit preload skipped: lmstudio_load_mode=jit")
        return None

    from hermes_cli.models import ensure_lmstudio_model_loaded

    if config_context_length is None:
        config_context_length = getattr(self, "_config_context_length", None)
    return ensure_lmstudio_model_loaded(
        self.model,
        self.base_url,
        getattr(self, "api_key", ""),
        config_context_length,
        return_load_result=True,
    )


def _is_direct_openai_url(self, base_url: str = None) -> bool:
    """Return True when a base URL targets OpenAI's native API."""
    if base_url is not None:
        hostname = base_url_hostname(base_url)
    else:
        hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
            getattr(self, "_base_url_lower", "")
        )
    return hostname == "api.openai.com"

def _is_azure_openai_url(self, base_url: str = None) -> bool:
    """Return True when a base URL targets Azure OpenAI.

    Azure OpenAI exposes an OpenAI-compatible endpoint at
    ``{resource}.openai.azure.com/openai/v1`` that accepts the
    standard ``openai`` Python client.  Unlike api.openai.com it
    does NOT support the Responses API — gpt-5.x models are served
    on the regular ``/chat/completions`` path — so routing decisions
    must treat Azure separately from direct OpenAI.
    """
    if base_url is not None:
        url = str(base_url).lower()
    else:
        url = getattr(self, "_base_url_lower", "") or ""
    return base_url_host_matches(url, "openai.azure.com")

def _is_github_copilot_url(self, base_url: str = None) -> bool:
    """Return True when a base URL targets GitHub Copilot's OpenAI-compatible API."""
    if base_url is not None:
        hostname = base_url_hostname(base_url)
    else:
        hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
            getattr(self, "_base_url_lower", "")
        )
    if not hostname:
        return False
    return hostname == "api.githubcopilot.com" or hostname.endswith(".githubcopilot.com")

def _resolved_api_call_timeout(self) -> float:
    """Resolve the effective per-call request timeout in seconds.

    Priority:
      1. ``providers.<id>.models.<model>.timeout_seconds`` (per-model override)
      2. ``providers.<id>.request_timeout_seconds`` (provider-wide)
      3. ``HERMES_API_TIMEOUT`` env var (legacy escape hatch)
      4. 1800.0s default

    Used by OpenAI-wire chat completions (streaming and non-streaming) so
    the per-provider config knob wins over the 1800s default.  Without this
    helper, the hardcoded ``HERMES_API_TIMEOUT`` fallback would always be
    passed as a per-call ``timeout=`` kwarg, overriding the client-level
    timeout the AgentState.__init__ path configured.
    """
    cfg = get_provider_request_timeout(self.provider, self.model)
    if cfg is not None:
        return cfg
    return env_float("HERMES_API_TIMEOUT", 1800.0)

def _resolved_api_call_stale_timeout_base(self) -> tuple[float, bool]:
    """Resolve the base non-stream stale timeout and whether it is implicit.

    Priority:
      1. ``providers.<id>.models.<model>.stale_timeout_seconds``
      2. ``providers.<id>.stale_timeout_seconds``
      3. ``HERMES_API_CALL_STALE_TIMEOUT`` env var
      4. 90.0s default (time-to-first-byte for non-streaming / Codex
         internal-streaming requests; lowered from 300s in May 2026 so
         fallback providers kick in faster when upstream providers
         stall).  The detector still scales up for large contexts in
         ``_compute_non_stream_stale_timeout``.

    Returns ``(timeout_seconds, uses_implicit_default)`` so the caller can
    preserve legacy behaviors that only apply when the user has *not*
    explicitly configured a stale timeout, such as auto-disabling the
    detector for local endpoints.
    """
    cfg = get_provider_stale_timeout(self.provider, self.model)
    if cfg is not None:
        return cfg, False

    env_timeout = os.getenv("HERMES_API_CALL_STALE_TIMEOUT")
    if env_timeout is not None:
        return float(env_timeout), False

    # Reasoning-model floor: auto-mitigation for known reasoning models
    # (Nemotron 3 Ultra, OpenAI o1/o3, Anthropic Opus 4.x thinking,
    # DeepSeek R1, Qwen QwQ, xAI Grok reasoning, etc.) whose cloud
    # gateways idle-kill before the model's thinking phase ends.
    # uses_implicit_default is False here so the local-endpoint
    # short-circuit in _compute_non_stream_stale_timeout does not
    # disable stale detection for users running reasoning models on a
    # local NIM endpoint.
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    reasoning_floor = get_reasoning_stale_timeout_floor(self.model)
    if reasoning_floor is not None:
        return reasoning_floor, False

    return 90.0, True

def _compute_non_stream_stale_timeout(self, api_payload: Any) -> float:
    """Compute the effective non-stream stale timeout for this request.

    Accepts either the full ``api_kwargs`` dict (Chat Completions or
    Responses API) or a legacy ``messages`` list.  Context-size scaling
    applies the same way to both shapes via
    :func:`agent.chat_completion_helpers.estimate_request_context_tokens`.
    """
    stale_base, uses_implicit_default = _resolved_api_call_stale_timeout_base(self)
    base_url = getattr(self, "_base_url", None) or self.base_url or ""
    if uses_implicit_default and base_url and is_local_endpoint(base_url):
        return float("inf")

    from agent.chat_completion_helpers import estimate_request_context_tokens
    est_tokens = estimate_request_context_tokens(api_payload)
    if est_tokens > 100_000:
        return max(stale_base, 240.0)
    if est_tokens > 50_000:
        return max(stale_base, 150.0)
    return stale_base

def _codex_silent_hang_hint(self, model: Optional[str] = None) -> Optional[str]:
    """Return an actionable hint when this request matches a known
    Codex silent-reject configuration, else ``None``.

    The ChatGPT Codex backend (``chatgpt.com/backend-api/codex``) has
    historically silently dropped certain model requests: the connection
    is accepted but no stream events are emitted and no error is raised.
    The stale-call detector ends the hang, but a generic "timed out"
    message gives the user no path forward.

    This helper substitutes an actionable hint into the stale-timeout
    warning when the request matches a known silent-reject pattern.

    Currently flagged: ``gpt-5.5`` family on the Codex backend.  See
    hermes-agent #21444 for the symptom history.  The upstream backend
    behavior has historically come and gone with ChatGPT entitlement
    changes — the heuristic stays in place as future-proofing even when
    the symptom is dormant.

    Does NOT fix the backend issue.  Only converts an opaque stale-timeout
    into actionable text so users learn the workaround in seconds rather
    than digging through logs.
    """
    if self.api_mode != "codex_responses":
        return None
    is_codex_backend = (
        self.provider == "openai-codex"
        or (
            getattr(self, "_base_url_hostname", "") == "chatgpt.com"
            and "/backend-api/codex" in (getattr(self, "_base_url_lower", "") or "")
        )
    )
    if not is_codex_backend:
        return None
    eff_model = (model if model is not None else self.model) or ""
    model_lower = eff_model.lower()
    # Match the gpt-5.5 family — bare ``gpt-5.5``, ``gpt-5.5-codex``,
    # vendor-prefixed variants like ``openai/gpt-5.5``, and any future
    # ``gpt-5.5-*`` SKU.  Anchor at a word boundary on either side so
    # unrelated tokens like ``gpt-5.50`` do not match.
    if not re.search(r"(?:^|[/\-_])gpt-5\.5(?:$|[\-_])", model_lower):
        return None
    return (
        f"Codex backend appears to be silently rejecting {eff_model!r} "
        "on chatgpt.com/backend-api/codex (no stream events, no error). "
        "This is a known backend-side pattern that has affected ChatGPT "
        "Plus accounts intermittently. "
        "Workaround: try `gpt-5.4` on the same OAuth profile, or `gpt-5.3-codex`, "
        "or switch to a different model/provider in your fallback chain. "
        "Some ChatGPT Codex accounts do not support `gpt-5.4-codex`. "
        "See hermes-agent#21444 for symptom history."
    )

def _is_openrouter_url(self) -> bool:
    """Return True when the base URL targets OpenRouter."""
    return base_url_host_matches(self._base_url_lower, "openrouter.ai")

def _is_copilot_url(self) -> bool:
    """Return True when the base URL targets GitHub Copilot or GitHub Models."""
    return (
        base_url_host_matches(self._base_url_lower, "api.githubcopilot.com")
        or base_url_host_matches(self._base_url_lower, "models.github.ai")
    )

def _is_copilot_provider(self) -> bool:
    """True when the active provider is GitHub Copilot, however spelled.

    ``self.provider`` is not always the normalized slug: ``/model`` and
    profile configs can leave the alias ``github-copilot`` (or ``github``)
    in place — a single session log can show both ``provider=copilot`` and
    ``provider=github-copilot`` for the same account. A bare
    ``provider == "copilot"`` gate silently skips credential recovery for
    the alias spellings, so this is the single owner of the check; the
    Copilot base URL is accepted as a fallback signal.
    """
    if (self.provider or "").strip().lower() in {"copilot", "github-copilot", "github"}:
        return True
    return _is_copilot_url(self)

def _is_codex_backend(self) -> bool:
    """Return True for the ChatGPT OAuth Codex Responses backend."""
    return (
        getattr(self, "api_mode", None) == "codex_responses"
        and getattr(self, "_base_url_hostname", "") == "chatgpt.com"
        and "/backend-api/codex"
        in (getattr(self, "_base_url_lower", "") or "")
    )



def _model_requires_responses_api(model: str) -> bool:
    """Return True for models that require the Responses API path.

    GPT-5.x models are rejected on /v1/chat/completions by both
    OpenAI and OpenRouter (error: ``unsupported_api_for_model``).
    Detect these so the correct api_mode is set regardless of
    which provider is serving the model.
    """
    m = model.lower()
    # Strip vendor prefix (e.g. "openai/gpt-5.4" → "gpt-5.4")
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m.startswith("gpt-5")

def _provider_model_requires_responses_api(
    model: str,
    *,
    provider: Optional[str] = None,
) -> bool:
    """Return True when this provider/model pair should use Responses API."""
    normalized_provider = (provider or "").strip().lower()
    # Nous serves GPT-5.x models via its OpenAI-compatible chat
    # completions endpoint; its /v1/responses endpoint returns 404.
    if normalized_provider == "nous":
        return False
    if normalized_provider == "custom":
        # Generic custom endpoints are conservative by default. They may
        # relay GPT-5 models without full Responses semantics, so only
        # direct OpenAI/xAI URL detection should auto-upgrade them.
        return False
    if normalized_provider == "copilot":
        try:
            from hermes_cli.models import _should_use_copilot_responses_api
            return _should_use_copilot_responses_api(model)
        except Exception:
            # Fall back to the generic GPT-5 rule if Copilot-specific
            # logic is unavailable for any reason.
            pass
    return _model_requires_responses_api(model)

def _max_tokens_param(self, value: int) -> dict:
    """Return the correct max tokens kwarg for the current provider.

    OpenAI's newer models (gpt-4o, gpt-4.1, gpt-5+, o-series) require
    'max_completion_tokens'. Azure OpenAI and GitHub Copilot also require
    'max_completion_tokens' for those families served via their
    OpenAI-compatible endpoints. OpenRouter, local models, and older
    OpenAI models use 'max_tokens'.

    The check is URL-first (api.openai.com / Azure / Copilot all use the
    new kwarg), then falls back to a model-name check so third-party
    OpenAI-compatible endpoints fronting those models are recognised —
    URL-only detection misses that case and silently sends the wrong
    kwarg, which the upstream model rejects with a 400.
    """
    if (
        _is_direct_openai_url(self)
        or _is_azure_openai_url(self)
        or _is_github_copilot_url(self)
        or model_forces_max_completion_tokens(self.model)
    ):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}

def _requested_output_cap_from_api_kwargs(api_kwargs: Any) -> Optional[int]:
    """Extract the outgoing response token cap from a prepared request."""
    if not isinstance(api_kwargs, dict):
        return None
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        raw = api_kwargs.get(key)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None

def _has_content_after_think_block(self, content: str) -> bool:
    """
    Check if content has actual text after any reasoning/thinking blocks.

    This detects cases where the model only outputs reasoning but no actual
    response, which indicates an incomplete generation that should be retried.
    Must stay in sync with _strip_think_blocks() tag variants.

    Args:
        content: The assistant message content to check

    Returns:
        True if there's meaningful content after think blocks, False otherwise
    """
    if not content:
        return False

    # Remove all reasoning tag variants (must match _strip_think_blocks)
    from agent import message_protocol

    cleaned = message_protocol.strip_think_blocks(self, content)

    # Check if there's any non-whitespace content remaining
    return bool(cleaned.strip())


def _has_natural_response_ending(content: str) -> bool:
    """Heuristic: does visible assistant text look intentionally finished?"""
    if not content:
        return False
    stripped = content.rstrip()
    if not stripped:
        return False
    if stripped.endswith("```"):
        return True
    if stripped.endswith('^'):
        return True
    last = stripped[-1]
    if last in '.!?:)"\']}。！？：）】」』》^':
        return True
    # Emoji ranges (Misc Symbols, Dingbats, Emoticons, Supplemental, etc.)
    if ord(last) >= 0x1F300:
        return True
    return False

def _is_ollama_glm_backend(self) -> bool:
    """Detect Ollama-hosted GLM models affected by stop misreports.

    Ollama can misreport truncated output as finish_reason='stop'.
    Detection relies on explicit Ollama signatures:
    - Port 11434 (Ollama default)
    - "ollama" in the base URL (e.g. ollama.local, /ollama/ path)
    - provider explicitly set to "ollama"

    Crucially it does NOT match arbitrary local/private endpoints
    (LiteLLM/sglang/vLLM/LM Studio proxies, Tailscale boxes), which
    report finish_reason correctly and were the source of #13971's
    false-positive truncation continuations.
    """
    model_lower = (self.model or "").lower()
    provider_lower = (self.provider or "").lower()
    if "glm" not in model_lower and provider_lower != "zai":
        return False
    if "ollama" in self._base_url_lower or ":11434" in self._base_url_lower:
        return True
    return provider_lower == "ollama"

def _should_treat_stop_as_truncated(
    self,
    finish_reason: str,
    assistant_message,
    messages: Optional[list] = None,
) -> bool:
    """Detect conservative stop->length misreports for Ollama-hosted GLM models."""
    if finish_reason != "stop" or self.api_mode != "chat_completions":
        return False
    if not _is_ollama_glm_backend(self):
        return False
    if not any(
        isinstance(msg, dict) and msg.get("role") == "tool"
        for msg in (messages or [])
    ):
        return False
    if assistant_message is None or getattr(assistant_message, "tool_calls", None):
        return False

    content = getattr(assistant_message, "content", None)
    if not isinstance(content, str):
        return False

    from agent import message_protocol

    visible_text = message_protocol.strip_think_blocks(self, content).strip()
    if not visible_text:
        return False
    if len(visible_text) < 20 or not re.search(r"\s", visible_text):
        return False

    return not _has_natural_response_ending(visible_text)



def _thread_identity(self) -> str:
    thread = threading.current_thread()
    return f"{thread.name}:{thread.ident}"

def _client_log_context(self) -> str:
    provider = getattr(self, "provider", "unknown")
    base_url = getattr(self, "base_url", "unknown")
    model = getattr(self, "model", "unknown")
    return (
        f"thread={_thread_identity(self)} provider={provider} "
        f"base_url={base_url} model={model}"
    )

def _openai_client_lock(self) -> threading.RLock:
    lock = getattr(self, "_client_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._client_lock = lock
    return lock

def _is_openai_client_closed(client: Any) -> bool:
    """Check if an OpenAI client is closed.

    Handles both property and method forms of is_closed:
    - httpx.Client.is_closed is a bool property
    - openai.OpenAI.is_closed is a method returning bool

    Prior bug: getattr(client, "is_closed", False) returned the bound method,
    which is always truthy, causing unnecessary client recreation on every call.
    """
    from unittest.mock import Mock

    if isinstance(client, Mock):
        return False

    is_closed_attr = getattr(client, "is_closed", None)
    if is_closed_attr is not None:
        # Handle method (openai SDK) vs property (httpx)
        if callable(is_closed_attr):
            if is_closed_attr():
                return True
        elif bool(is_closed_attr):
            return True

    http_client = getattr(client, "_client", None)
    if http_client is not None:
        return bool(getattr(http_client, "is_closed", False))
    return False

def _build_keepalive_http_client(base_url: str = "", *, verify: Any = True) -> Any:
    """Build an httpx.Client with proactive idle-connection reaping.

    Previously this method injected a custom ``httpx.HTTPTransport``
    with ``socket_options`` (``SO_KEEPALIVE``, ``TCP_KEEPIDLE``, …) to
    prevent CLOSE-WAIT accumulation on long-lived connections (#10324).

    That approach broke streaming for providers behind reverse proxies
    (OpenResty, Cloudflare, etc.) because the custom socket options
    conflict with the proxy's chunked-transfer handling (#54049,
    #12952).  It also stripped ``TCP_NODELAY``, stalling TLS handshakes
    and SSE encoding.

    The fix moves connection lifecycle management from the socket layer
    to the HTTP pool layer: ``keepalive_expiry=20.0`` tells httpx to
    close idle pooled connections *before* a reverse proxy's typical
    30–60 s timeout drops them, preventing CLOSE-WAIT accumulation
    without modifying socket options.  The default httpx transport
    preserves OS TCP defaults (including ``TCP_NODELAY``).

    ``verify`` carries per-provider ``ssl_ca_cert`` / ``ssl_verify`` and
    ``HERMES_CA_BUNDLE`` settings.  It is passed on the client AND on
    the plain no-proxy mounts (a mounted transport owns the SSL context
    for its scheme).
    """
    try:
        import httpx as _httpx

        # Explicitly read proxy settings so requests route through
        # HTTP_PROXY / HTTPS_PROXY / NO_PROXY correctly.
        _proxy = _get_proxy_for_base_url(base_url)

        # Proactive pool reaping: close idle connections at 20 s,
        # before reverse proxies (30–60 s typical) send FIN and
        # cause CLOSE-WAIT accumulation.
        _limits = _httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=20.0,
        )

        # Timeouts: generous read=None for SSE streaming endpoints.
        _timeout = _httpx.Timeout(
            connect=15.0,
            read=None,
            write=15.0,
            pool=10.0,
        )

        # When _proxy is None (NO_PROXY bypass or no proxy configured),
        # mount plain transports to prevent httpx from reading env proxy
        # vars and creating an HTTPProxy mount that would bypass our
        # NO_PROXY resolution.
        _mounts = {}
        if _proxy is None:
            _mounts = {
                "http://": _httpx.HTTPTransport(verify=verify),
                "https://": _httpx.HTTPTransport(verify=verify),
            }
        return _httpx.Client(
            limits=_limits,
            timeout=_timeout,
            proxy=_proxy,
            mounts=_mounts or None,
            verify=verify,
        )
    except Exception:
        return None




def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
    if client is None:
        return
    # Force-close TCP sockets first to prevent CLOSE-WAIT accumulation,
    # then do the graceful SDK-level close.
    force_closed = _force_close_tcp_sockets(client)
    try:
        client.close()
        logger.info(
            "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
            reason,
            shared,
            force_closed,
            _client_log_context(self),
        )
    except Exception as exc:
        logger.debug(
            "OpenAI client close failed (%s, shared=%s) %s error=%s",
            reason,
            shared,
            _client_log_context(self),
            exc,
        )

def _retire_shared_openai_client(self, client: Any, *, reason: str) -> None:
    """Ownership-safe retirement of a replaced shared OpenAI client.


    #70773 / #67142 / #29507: ``client.close()`` releases the pool's raw
    FDs from the *calling* thread. The shared primary client has no single
    owning thread — worker threads from stale-killed attempts may still be
    unwinding their SSL BIOs, and the codex-direct / MoA paths stream on
    the shared client itself. If we release an FD while another thread's
    SSL layer still caches the raw integer fd, the kernel can recycle it
    into an unrelated ``open()`` (e.g. ``kanban.db``) and the unwinding
    TLS flush then writes an application-data record into that file — the
    SQLite-header corruption documented in #29507/#70773.

    Only an owner may release FDs, and a replaced shared client has none.
    So nobody calls ``lifecycle.close()``: we ``shutdown()`` the pooled sockets
    (FD-safe from any thread; unblocks in-flight readers with EOF/EPIPE)
    and defer the actual FD release to garbage collection. Refcounting
    guarantees the underlying sockets are only collected once every
    thread that borrowed the client has unwound — GC *is* the ownership
    handshake. In the common case (no borrower) the refcount hits zero on
    this line and the FDs are released immediately anyway.
    """
    if client is None:
        return
    try:
        shutdown_count = _force_close_tcp_sockets(client)
        logger.info(
            "Shared OpenAI client retired (%s, tcp_shutdown=%d, "
            "fd_release=deferred_to_gc) %s",
            reason,
            shutdown_count,
            _client_log_context(self),
        )
    except Exception as exc:
        logger.debug(
            "Shared OpenAI client retire failed (%s) %s error=%s",
            reason,
            _client_log_context(self),
            exc,
        )

def _build_primary_client_for_active_provider(self, *, reason: str) -> Any:
    """Build the shared client shape required by the active provider.

    MoA is a virtual provider whose ``client`` is an in-process facade,
    not an OpenAI SDK client. Generic rebuild paths (credential rotation,
    timeout application, and dead-connection cleanup) still pass through
    this helper, so they must preserve that provider/client invariant.
    """
    if (getattr(self, "provider", "") or "").strip().lower() == "moa":
        from agent.moa_loop import build_moa_facade

        return build_moa_facade(self, self.model)
    return _create_openai_client(self,
        self._client_kwargs,
        reason=reason,
        shared=True,
    )

def _replace_primary_openai_client(self, *, reason: str) -> bool:
    with _openai_client_lock(self):
        old_client = getattr(self, "client", None)
        try:
            new_client = _build_primary_client_for_active_provider(self,
                reason=reason,
            )
        except Exception as exc:
            logger.warning(
                "Failed to rebuild shared primary client (%s) %s error=%s",
                reason,
                _client_log_context(self),
                exc,
            )
            return False
        self.client = new_client
    # #70773: never hard-close the replaced shared client from here — the
    # caller may not be the thread whose request is still unwinding on the
    # old pool (credential rotation and dead-connection cleanup run on the
    # turn thread while stale-killed workers unwind; the codex-direct path
    # streams on the shared client itself). Retire it instead: sockets are
    # shut down (FD-safe), FD release deferred to GC.
    _retire_shared_openai_client(self, old_client, reason=f"replace:{reason}")
    return True

def _ensure_primary_openai_client(self, *, reason: str) -> Any:
    with _openai_client_lock(self):
        client = getattr(self, "client", None)
        if client is not None and not _is_openai_client_closed(client):
            return client
        old_client = client
        try:
            new_client = _create_openai_client(self,
                self._client_kwargs, reason=reason, shared=True
            )
        except Exception as exc:
            logger.warning(
                "Failed to recreate closed OpenAI client (%s) %s error=%s",
                reason,
                _client_log_context(self),
                exc,
            )
            raise RuntimeError("Failed to recreate closed OpenAI client") from exc
        self.client = new_client

    logger.warning(
        "Detected closed shared OpenAI client; recreated before use (%s) %s",
        reason,
        _client_log_context(self),
    )
    _close_openai_client(self, old_client, reason=f"replace:{reason}", shared=True)
    return new_client


def _api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
    """Return True when the outbound request still contains native image parts."""
    if not isinstance(api_kwargs, dict):
        return False
    candidates = []
    messages = api_kwargs.get("messages")
    if isinstance(messages, list):
        candidates.extend(messages)
    # Responses API payloads use `input`; after conversion, image parts can
    # still be present there instead of in `messages`.
    response_input = api_kwargs.get("input")
    if isinstance(response_input, list):
        candidates.extend(response_input)

    def _contains_image(value: Any) -> bool:
        if isinstance(value, dict):
            ptype = value.get("type")
            if ptype in {"image_url", "input_image"}:
                return True
            return any(_contains_image(v) for v in value.values())
        if isinstance(value, list):
            return any(_contains_image(v) for v in value)
        return False

    return any(_contains_image(item) for item in candidates)

def _copilot_headers_for_request(self, *, is_vision: bool) -> dict:
    from hermes_cli.copilot_auth import copilot_request_headers

    return copilot_request_headers(is_agent_turn=True, is_vision=is_vision)

def _request_client_cache_ref(self) -> dict:
    # Lazy init — tests build agents via AgentState.__new__ without __init__.
    cache = getattr(self, "_request_client_cache", None)
    if cache is None:
        cache = {"client": None, "kwargs": None, "poisoned": False, "in_use": False}
        self._request_client_cache = cache
    return cache

def _create_request_openai_client(self, *, reason: str, api_kwargs: Optional[dict] = None) -> Any:
    from unittest.mock import Mock

    primary_client = _ensure_primary_openai_client(self, reason=reason)
    if self.provider == "moa":
        return primary_client
    if isinstance(primary_client, Mock):
        return primary_client
    with _openai_client_lock(self):
        request_kwargs = dict(self._client_kwargs)
    # Per-request OpenAI-wire clients (used by both the non-streaming
    # chat-completions path and the streaming chat-completions path
    # in `_interruptible_api_call`) should not run the SDK's built-in
    # retry loop: the agent's outer loop owns retries with credential
    # rotation, provider fallback, and backoff that the SDK can't
    # see. Leaving SDK retries on (default 2) compounds with our outer
    # retries and lets a single hung provider request stretch to ~3x
    # the per-call timeout before our stale detector reports it.
    # Shared/primary clients and Anthropic / Bedrock paths are
    # unaffected (they don't go through here).
    request_kwargs["max_retries"] = 0
    if (
        base_url_host_matches(str(request_kwargs.get("base_url", "")), "githubcopilot.com")
        and _api_kwargs_have_image_parts(api_kwargs or {})
    ):
        request_kwargs["default_headers"] = _copilot_headers_for_request(self, is_vision=True)
    # Reuse the cached wire client while the effective kwargs are
    # unchanged: constructing openai.OpenAI + its httpx pool costs
    # ~19-35ms per LLM call (fresh TCP+TLS handshake), ~5x per turn.
    # The cache is a single checked-out slot: `in_use` prevents two
    # concurrent calls from sharing one pool's close/abort lifecycle
    # (a second concurrent call gets a fresh untracked client with
    # the old build-per-request behavior).
    stale = None
    with _openai_client_lock(self):
        cache = _request_client_cache_ref(self)
        cached = cache["client"]
        if cached is not None and not cache["in_use"]:
            if (
                not cache["poisoned"]
                and cache["kwargs"] == request_kwargs
                and not _is_openai_client_closed(cached)
            ):
                cache["in_use"] = True
                return cached
            # kwargs changed (credential rotation, provider failover),
            # poisoned by a cross-thread abort (#29507), or externally
            # closed — never reuse; discard and rebuild below.
            stale = cached
            cache["client"] = None
            cache["kwargs"] = None
            cache["poisoned"] = False
    if stale is not None:
        # Safe to close from this thread: in_use was False, so no
        # worker thread owns the pool's FDs (#29507 concerns clients
        # with an in-flight request on another thread).
        _close_openai_client(self, stale, reason=f"reuse_evict:{reason}", shared=False)
    client = _create_openai_client(self, request_kwargs, reason=reason, shared=False)
    with _openai_client_lock(self):
        cache = _request_client_cache_ref(self)
        if cache["client"] is None:
            cache["client"] = client
            # Snapshot nested dicts (default_headers): rotation sites
            # assign fresh inner dicts today, but an aliased inner
            # object would compare equal even after in-place mutation.
            cache["kwargs"] = {
                k: dict(v) if isinstance(v, dict) else v
                for k, v in request_kwargs.items()
            }
            cache["poisoned"] = False
            cache["in_use"] = True
        # else: a concurrent call holds the slot — hand this client
        # out untracked; _close_request_openai_client fully closes
        # untracked clients, preserving the per-request lifecycle.
    return client

def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
    with _openai_client_lock(self):
        cache = _request_client_cache_ref(self)
        if cache["client"] is client:
            if reason in _REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                # Clean finish on the owning thread — keep the wire client
                # (and its warm httpx pool) for the next sequential call.
                cache["in_use"] = False
                return
            # Failure / kill / abort outcome: drop the slot and fall
            # through to a real close. This runs on the owning worker
            # thread, which is where the FD release belongs (#29507).
            cache["client"] = None
            cache["kwargs"] = None
            cache["poisoned"] = False
            cache["in_use"] = False
    _close_openai_client(self, client, reason=reason, shared=False)

def _close_cached_request_openai_client(self, *, reason: str) -> None:
    """Teardown hook: really close the cached per-request wire client."""
    with _openai_client_lock(self):
        cache = getattr(self, "_request_client_cache", None)
        client = cache["client"] if cache else None
        in_use = bool(cache["in_use"]) if cache else False
        if cache is not None:
            cache["client"] = None
            cache["kwargs"] = None
            cache["poisoned"] = False
            cache["in_use"] = False
    if client is None:
        return
    if in_use:
        # A worker thread has this client checked out for an in-flight
        # request (workers can outlive turns — see interruptible_api_call).
        # client.close() here would release its FDs from a stranger thread,
        # the #29507 race teardown must not reintroduce. Abort the sockets
        # instead; the slot is already cleared, so the worker's own finally
        # sees an untracked client and does the real close on its thread.
        _abort_request_openai_client(self, client, reason=f"{reason}_in_flight")
        return
    _close_openai_client(self, client, reason=reason, shared=False)

def _abort_request_openai_client(self, client: Any, *, reason: str) -> None:
    """Cross-thread abort: shut sockets down without releasing FDs.

    Companion to :meth:`_close_request_openai_client` for stranger-thread
    callers (interrupt-check loop, stale-call detector). Calling
    ``client.close()`` from a thread that does not own the active httpx
    connection raced the still-live SSL BIO and corrupted unrelated file
    descriptors when the kernel recycled the just-freed TCP FD (#29507).

    Here we only ``shutdown(SHUT_RDWR)`` the pool's sockets. That unblocks
    the owning worker thread's pending ``recv``/``send`` with an EOF or
    ``EPIPE`` so it can unwind and close ``client`` from its own context
    — which is where the FD release belongs.
    """
    if client is None:
        return
    # A pool whose sockets were shut down from a stranger thread must
    # never be reused: poison the cache slot so the owner-thread close
    # discards it and the next create builds a fresh client.
    with _openai_client_lock(self):
        cache = _request_client_cache_ref(self)
        if cache["client"] is client:
            cache["poisoned"] = True
    try:
        shutdown_count = _force_close_tcp_sockets(client)
        # tcp_force_closed=0 means the stranger-thread abort found no
        # sockets to shut down — the worker stays blocked in recv and the
        # provider keeps the slot (#72975). Surface that as WARNING so it
        # cannot be mistaken for a successful abort in the logs.
        _log = logger.warning if shutdown_count == 0 else logger.info
        _log(
            "OpenAI client aborted (%s, shared=False, tcp_force_closed=%d, "
            "deferred_close=stranger_thread) %s%s",
            reason,
            shutdown_count,
            _client_log_context(self),
            (
                " — no sockets found; in-flight request may keep running "
                "until the provider finishes"
                if shutdown_count == 0
                else ""
            ),
        )
    except Exception as exc:
        logger.debug(
            "OpenAI client abort failed (%s, shared=False) %s error=%s",
            reason,
            _client_log_context(self),
            exc,
        )


def _request_anthropic_client_cache_ref(self) -> dict:
    # Lazy init — tests build agents via AgentState.__new__ without __init__.
    cache = getattr(self, "_request_anthropic_client_cache", None)
    if cache is None:
        cache = {"client": None, "key": None, "poisoned": False, "in_use": False}
        self._request_anthropic_client_cache = cache
    return cache

def _request_anthropic_client_key(self) -> tuple:
    """Cache key covering everything that forces a fresh client: credential
    rotation, base URL / region changes, timeout changes (model switch),
    and the 1M-context beta flag."""
    if getattr(self, "provider", None) == "bedrock":
        region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
        return ("bedrock", region)
    return (
        "direct",
        self._anthropic_api_key,
        getattr(self, "_anthropic_base_url", None),
        get_provider_request_timeout(self.provider, self.model),
        bool(getattr(self, "_oauth_1m_beta_disabled", False)),
    )

def _create_request_anthropic_client(self, *, reason: str) -> Any:
    """Build (or reuse) a request-local Anthropic client for one in-flight call.

    The shared ``_anthropic_client`` stays the long-lived primary, but the
    stale/interrupt watchdog runs on the poll thread and must never call
    ``lifecycle.close()`` on the client whose TLS socket a worker thread is still
    reading: releasing that FD from a stranger thread lets the kernel
    recycle it under a still-live SSL BIO, which then writes a TLS record
    into an unrelated SQLite header (#29507 / #67142). A per-request client
    lets the stranger thread ``shutdown()`` the socket while the owning
    worker performs the SDK-level close from its own context — the same
    ownership contract the OpenAI-wire path already uses.

    Also mirrors the OpenAI-wire path's single-slot cache
    (``_create_request_openai_client``): building ``anthropic.Anthropic``
    means a fresh httpx pool and TCP+TLS handshake per call, so the client
    is kept warm across sequential calls whose cache key (credentials,
    base URL/region, timeout, 1M-beta flag) hasn't changed. ``in_use``
    keeps a second concurrent call from sharing one pool's close/abort
    lifecycle — it gets a fresh untracked client instead.

    Mirrors ``_rebuild_anthropic_client`` construction (direct + Bedrock,
    1M-beta drop) but returns a fresh/cached client instead of swapping
    the shared one.
    """
    if self.api_mode == "anthropic_messages":
        _try_refresh_anthropic_client_credentials(self)
    key = _request_anthropic_client_key(self)

    stale = None
    with _openai_client_lock(self):
        cache = _request_anthropic_client_cache_ref(self)
        cached = cache["client"]
        if cached is not None and not cache["in_use"]:
            if (
                not cache["poisoned"]
                and cache["key"] == key
                and not _is_openai_client_closed(cached)
            ):
                cache["in_use"] = True
                return cached
            # Key changed (credential rotation, base URL/region, timeout,
            # 1M-beta flip), poisoned by a cross-thread abort, or
            # externally closed — never reuse; discard and rebuild below.
            stale = cached
            cache["client"] = None
            cache["key"] = None
            cache["poisoned"] = False
    if stale is not None:
        # Safe to close from this thread: in_use was False, so no worker
        # thread owns the pool's FDs (same #29507 reasoning as OpenAI).
        _close_request_anthropic_client(self, stale, reason=f"reuse_evict:{reason}")

    if key[0] == "bedrock":

        from agent.anthropic_adapter import build_anthropic_bedrock_client
        client = build_anthropic_bedrock_client(key[1])
    else:
        from agent.anthropic_adapter import build_anthropic_client
        client = build_anthropic_client(
            self._anthropic_api_key,
            getattr(self, "_anthropic_base_url", None),
            timeout=get_provider_request_timeout(self.provider, self.model),
            drop_context_1m_beta=key[4],
        )
    logger.debug(
        "Anthropic request client created (%s, shared=False) provider=%s model=%s",
        reason,
        getattr(self, "provider", None),
        getattr(self, "model", None),
    )
    with _openai_client_lock(self):
        cache = _request_anthropic_client_cache_ref(self)
        if cache["client"] is None:
            cache["client"] = client
            cache["key"] = key
            cache["poisoned"] = False
            cache["in_use"] = True
        # else: a concurrent call holds the slot — hand this client out
        # untracked; _close_request_anthropic_client fully closes
        # untracked clients, preserving the per-request lifecycle.
    return client

def _close_request_anthropic_client(self, client: Any, *, reason: str) -> None:
    """Owner-thread close of a request-local Anthropic client.

    On a clean finish (``reason`` in ``_REQUEST_CLIENT_REUSE_REASONS``)
    the pool is kept warm in the cache slot for the next sequential call,
    mirroring ``_close_request_openai_client``. Any other outcome
    (error / kill / abort / stale-slot eviction) force-closes the pool's
    TCP sockets first (CLOSE-WAIT hygiene, parity with
    ``_close_openai_client``), then does the graceful SDK close. Safe
    because the caller owns the connection.
    """
    if client is None:
        return
    with _openai_client_lock(self):
        cache = _request_anthropic_client_cache_ref(self)
        if cache["client"] is client:
            if reason in _REQUEST_CLIENT_REUSE_REASONS and not cache["poisoned"]:
                cache["in_use"] = False
                return
            cache["client"] = None
            cache["key"] = None
            cache["poisoned"] = False
            cache["in_use"] = False
    try:
        _force_close_tcp_sockets(client)
        client.close()
        logger.info(
            "Anthropic client closed (%s, shared=False) provider=%s model=%s",
            reason,
            getattr(self, "provider", None),
            getattr(self, "model", None),
        )
    except Exception as exc:
        logger.debug(
            "Anthropic client close failed (%s, shared=False) provider=%s model=%s error=%s",
            reason,
            getattr(self, "provider", None),
            getattr(self, "model", None),
            exc,
        )

def _close_cached_request_anthropic_client(self, *, reason: str) -> None:
    """Teardown hook: really close the cached per-request Anthropic client."""
    with _openai_client_lock(self):
        cache = getattr(self, "_request_anthropic_client_cache", None)
        client = cache["client"] if cache else None
        in_use = bool(cache["in_use"]) if cache else False
        if cache is not None:
            cache["client"] = None
            cache["key"] = None
            cache["poisoned"] = False
            cache["in_use"] = False
    if client is None:
        return
    if in_use:
        # A worker thread has this client checked out for an in-flight
        # request — same #29507 reasoning as the OpenAI teardown hook.
        _abort_request_anthropic_client(self, client, reason=f"{reason}_in_flight")
        return
    try:
        _force_close_tcp_sockets(client)
        client.close()
    except Exception:
        pass

def _abort_request_anthropic_client(self, client: Any, *, reason: str) -> None:
    """Cross-thread abort for request-local Anthropic clients.

    Stranger threads (the interrupt-check / stale-stream detector loop)
    must not call the SDK ``lifecycle.close()`` — that races the owning worker's live
    SSL BIO and can recycle a TLS FD into a SQLite header (#29507 /
    #67142). Only ``shutdown(SHUT_RDWR)`` the pool's sockets so the worker
    unblocks and releases the FD from its own thread.
    """
    if client is None:
        return
    # A pool whose sockets were shut down from a stranger thread must
    # never be reused: poison the cache slot so the owner-thread close
    # discards it and the next create builds a fresh client.
    with _openai_client_lock(self):
        cache = _request_anthropic_client_cache_ref(self)
        if cache["client"] is client:
            cache["poisoned"] = True
    try:
        shutdown_count = _force_close_tcp_sockets(client)
        # Same visibility contract as the OpenAI abort path (#72975):
        # zero sockets shut down means the abort did not unblock the
        # worker — log WARNING, not a success-shaped INFO.
        _log = logger.warning if shutdown_count == 0 else logger.info
        _log(
            "Anthropic client aborted (%s, shared=False, tcp_force_closed=%d, "
            "deferred_close=stranger_thread) provider=%s model=%s%s",
            reason,
            shutdown_count,
            getattr(self, "provider", None),
            getattr(self, "model", None),
            (
                " — no sockets found; in-flight request may keep running "
                "until the provider finishes"
                if shutdown_count == 0
                else ""
            ),
        )
    except Exception as exc:
        logger.debug(
            "Anthropic client abort failed (%s, shared=False) provider=%s model=%s error=%s",
            reason,
            getattr(self, "provider", None),
            getattr(self, "model", None),
            exc,
        )

def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
    """Forwarder — see ``agent.codex_runtime.run_codex_stream``."""
    from agent.codex_runtime import run_codex_stream
    return run_codex_stream(self, api_kwargs, client, on_first_delta)

def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
    """Forwarder — see ``agent.codex_runtime.run_codex_create_stream_fallback``."""
    from agent.codex_runtime import run_codex_create_stream_fallback
    return run_codex_create_stream_fallback(self, api_kwargs, client)

def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
    if self.api_mode != "codex_responses" or self.provider not in {"openai-codex", "xai-oauth"}:
        return False

    # Guard against silent account swap.
    #
    # When an agent is using a non-singleton credential — e.g. a manual
    # pool entry (``hermes auth add xai-oauth``) whose tokens belong to
    # a different account than the device_code singleton, or an agent
    # constructed with an explicit ``api_key=`` arg — force-refreshing
    # the singleton here and adopting its tokens silently re-routes the
    # rest of the conversation onto the singleton's account.  The
    # credential pool's reactive recovery (``_recover_with_credential_pool``)
    # is the right channel for that case; this path is the
    # singleton-only fallback used when the pool can't recover, and
    # MUST only fire when the agent really is on singleton tokens.
    try:
        if self.provider == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials

            singleton_now = resolve_codex_runtime_credentials(
                refresh_if_expiring=False,
            )
        else:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            singleton_now = resolve_xai_oauth_runtime_credentials(
                refresh_if_expiring=False,
            )
    except Exception as exc:
        logger.debug("%s singleton read failed: %s", self.provider, exc)
        return False

    singleton_key = str(singleton_now.get("api_key") or "").strip()
    active_key = str(self.api_key or "").strip()
    if singleton_key and active_key and singleton_key != active_key:
        logger.debug(
            "%s singleton tokens differ from the active api_key; "
            "skipping singleton force-refresh to avoid silent account swap. "
            "Reactive credential rotation should go through the pool.",
            self.provider,
        )
        return False

    try:
        if self.provider == "openai-codex":
            from hermes_cli.auth import resolve_codex_runtime_credentials

            old_key = str(self.api_key or "").strip()
            creds = resolve_codex_runtime_credentials(force_refresh=force)
        else:
            from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

            old_key = str(self.api_key or "").strip()
            creds = resolve_xai_oauth_runtime_credentials(force_refresh=force)
    except Exception as exc:
        logger.debug("%s credential refresh failed: %s", self.provider, exc)
        return False

    api_key = creds.get("api_key")
    base_url = creds.get("base_url")
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    # Defect 2 fix: return False when no NEW token was actually minted.
    # resolve_codex_runtime_credentials returns the same stale token
    # when the underlying refresh fails (failure is debug-only).
    # Comparing the access token (api_key) before/after detects this.
    new_key = api_key.strip()
    if old_key and new_key == old_key:
        logger.debug(
            "%s credential refresh returned the same token; "
            "refresh likely failed silently",
            self.provider,
        )
        return False

    self.api_key = api_key.strip()
    set_base_url(self, base_url.strip().rstrip("/"))
    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url

    if not _replace_primary_openai_client(self, reason=f"{self.provider}_credential_refresh"):
        return False

    return True

def _try_refresh_nous_client_credentials(
    self,
    *,
    force: bool = True,
) -> bool:
    if self.provider != "nous":
        return False
    # Portal serves anthropic/* on the native Messages route, so a session
    # can be holding either client kind when its short-lived invoke JWT
    # expires. Both need the refresh or the turn dies on a 401.
    if self.api_mode not in ("chat_completions", "anthropic_messages"):
        return False

    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials(
            timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
            force_refresh=force,
        )
    except Exception as exc:
        logger.debug("Nous credential refresh failed: %s", exc)
        return False

    api_key = creds.get("api_key")
    base_url = creds.get("base_url")
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    self.api_key = api_key.strip()
    set_base_url(self, base_url.strip().rstrip("/"))


    if self.api_mode == "anthropic_messages":
        self._anthropic_api_key = self.api_key
        self._anthropic_base_url = self.base_url
        _rebuild_anthropic_client(self)
        return True

    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url
    # Nous requests should not inherit OpenRouter-only attribution headers.
    self._client_kwargs.pop("default_headers", None)

    if not _replace_primary_openai_client(self, reason="nous_credential_refresh"):
        return False

    return True

def _try_refresh_env_client_credentials(self) -> bool:
    """Adopt ~/.hermes/.env credential/base-url edits at the turn boundary.

    A Settings save (desktop ``PUT /api/env``, ``hermes setup``) updates
    ``.env`` and the *saving* process's os.environ, but a live session
    worker keeps the base_url/api_key captured at agent init until it
    restarts — so an open chat silently keeps calling the old endpoint
    (#67821). Called at the start of each conversation turn, this
    re-resolves the provider's env-sourced credentials (load_env() is
    mtime-memoized, so an unchanged file costs one stat()) and rebuilds
    the client when the user edited them.

    Reacts only to env *edits* (resolved values changed since the last
    look), never to mere divergence from the agent's current values —
    credential-pool rotation and failover legitimately move the session
    off the env credential, and stomping those back every turn would
    flap. A config.yaml ``model.base_url`` (or a pool entry with a
    custom endpoint) also wins: edits are only adopted while the
    session's current base_url is still the registry default or the
    previously-seen env value.

    Covers api-key registry providers and named custom providers with a
    ``key_env`` (#67935) — the latter resolve to ``provider="custom"``
    with no registry entry, so they are matched through the runtime
    provider's config lookup instead.
    """
    if self.api_mode != "chat_completions":
        return False
    if getattr(self, "_fallback_activated", False):
        return False
    try:
        from agent.credential_pool import get_env_prefer_dotenv
        from hermes_cli.auth import PROVIDER_REGISTRY
    except ImportError:
        return False

    pconfig = PROVIDER_REGISTRY.get(self.provider)
    if (
        pconfig
        and getattr(pconfig, "auth_type", "") == "api_key"
        and getattr(pconfig, "api_key_env_vars", ())
    ):
        api_key = ""
        for env_var in pconfig.api_key_env_vars:
            api_key = get_env_prefer_dotenv(env_var).strip()
            if api_key:
                break
        if not api_key:
            return False

        env_url = ""
        if pconfig.base_url_env_var:
            env_url = get_env_prefer_dotenv(pconfig.base_url_env_var).strip().rstrip("/")
        default_base = (pconfig.inference_base_url or "").strip().rstrip("/")
        base_url = env_url or default_base
        if self.provider == "kimi-coding":
            from hermes_cli.auth import _resolve_kimi_base_url

            base_url = _resolve_kimi_base_url(
                api_key, pconfig.inference_base_url, env_url
            ).rstrip("/")
        elif self.provider == "zai":
            from hermes_cli.auth import _resolve_zai_base_url

            base_url = _resolve_zai_base_url(
                api_key, pconfig.inference_base_url, env_url
            ).rstrip("/")
    elif self.provider == "custom":
        # Named custom provider (#67935): identity lives in config
        # (``providers.<name>`` / ``custom_providers``), the credential in
        # the env var it names via ``key_env``. Re-resolve through the
        # same config lookup the runtime resolver uses; entries without
        # ``key_env`` (inline ``api_key``, pool-backed) have no
        # env-sourced credential to watch.
        try:
            from hermes_cli.runtime_provider import _get_named_custom_provider
        except ImportError:
            return False
        custom_provider = _get_named_custom_provider(
            getattr(self, "requested_provider", "") or ""
        )
        if not custom_provider:
            return False
        key_env = str(custom_provider.get("key_env") or "").strip()
        if not key_env:
            return False
        api_key = get_env_prefer_dotenv(key_env).strip()
        if not api_key:
            return False
        # Custom providers pin their endpoint in config, not env — the
        # config base_url is both the resolved and the "default" base, so
        # only key edits are ever adopted here.
        default_base = str(custom_provider.get("base_url") or "").strip().rstrip("/")
        base_url = default_base
    else:
        return False

    if not base_url:
        return False

    resolved = (base_url, api_key)
    prev = getattr(self, "_env_creds_seen", None)
    current_base = (self.base_url or "").strip().rstrip("/")

    if prev is None:
        # First look — no baseline to diff against. Adopt only the
        # boot-default case (worker spawned before the user saved an
        # override); anything else is unattributable on turn one.
        adopt = current_base == default_base and not (
            base_url == current_base and api_key == self.api_key
        )

        # #79156: if the session already holds a pool-rotated key, do
        # not treat that divergence as a boot-time env adoption. First
        # look would otherwise stomp the rotated key with the env value
        # while leaving ``_credential_pool_entry_id`` on the fallback.
        if (
            adopt
            and api_key != self.api_key
            and getattr(self, "_credential_pool", None) is not None
            and getattr(self, "_credential_pool_entry_id", None)
        ):
            adopt = False
    else:
        # Env unchanged → no-op; any drift from self.* is rotation/
        # failover or config precedence — leave it alone. An edit is
        # only adopted while the session still runs on the registry
        # default or the previously-seen env value.
        adopt = (
            resolved != prev
            and current_base in {default_base, prev[0]}
            and not (base_url == current_base and api_key == self.api_key)
        )

    if not adopt:
        self._env_creds_seen = resolved
        return False

    from hermes_cli.route_identity import normalize_route_base_url

    route_changed = normalize_route_base_url(self.base_url) != normalize_route_base_url(
        base_url
    )
    prior_api_key = self.api_key
    prior_base_url = self.base_url
    prior_client_kwargs = dict(self._client_kwargs)

    self.api_key = api_key
    set_base_url(self, base_url)
    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url
    # A base-url change moves the route: TLS material and default
    # headers derived from the old endpoint must be recomputed, exactly
    # as on credential-pool rotation.
    _reapply_route_client_config(self, route_changed=route_changed)

    if not _replace_primary_openai_client(self, reason="env_credential_refresh"):
        # Leave the baseline un-advanced so the unchanged edit is
        # retried next turn, and roll the agent back so its state keeps
        # matching the still-live old client.
        self.api_key = prior_api_key
        set_base_url(self, prior_base_url)
        self._client_kwargs.clear()
        self._client_kwargs.update(prior_client_kwargs)
        return False

    # Rebind the pool entry id to the key we just adopted. Leaving a
    # stale id after a key rewrite makes mark_exhausted_and_rotate
    # quarantine the wrong credential on the next 429 (#79156).
    try:
        sync_credential_pool_entry_id(self)
    except Exception:
        logger.debug(
            "sync_credential_pool_entry_id after env refresh failed",
            exc_info=True,
        )

    self._env_creds_seen = resolved
    logger.info(
        "Applied updated .env credentials for %s: endpoint %s",
        self.provider,
        self.base_url,
    )
    return True

def _try_refresh_vertex_client_credentials(self) -> bool:
    """Re-mint the Vertex OAuth2 access token and rebuild the OpenAI client.

    Vertex tokens live ~1 hour. On a long-lived agent (gateway session) a
    cached client's bearer token will expire mid-session, producing a 401.
    This re-resolves credentials via the adapter (which refreshes the
    underlying google-auth Credentials object when near expiry), swaps the
    new token into the client kwargs, and rebuilds the primary OpenAI
    client. Returns True when a usable token+base_url were obtained.
    """
    if self.api_mode != "chat_completions" or self.provider != "vertex":
        return False

    try:
        from agent.vertex_adapter import get_vertex_config

        token, base_url = get_vertex_config()
    except Exception as exc:
        logger.debug("Vertex credential refresh failed: %s", exc)
        return False

    if not isinstance(token, str) or not token.strip():
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    self.api_key = token.strip()
    set_base_url(self, base_url.strip().rstrip("/"))
    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url

    if not _replace_primary_openai_client(self, reason="vertex_credential_refresh"):
        return False

    logger.info("Vertex AI OAuth token refreshed")
    return True

def _try_refresh_copilot_client_credentials(self) -> bool:
    """Refresh Copilot credentials and rebuild the shared OpenAI client.

    The raw GitHub OAuth token (`gh auth token`) is usually stable, but the
    short-TTL *exchanged* IDE token minted from it is what Copilot actually
    authenticates — and it expires mid-session. A heavy/long turn whose
    request straddles that expiry gets a clean `401 IDE token expired:
    unauthorized: token expired`. Simply re-resolving the (unchanged) raw
    token and rebuilding the client leaves the SAME expired IDE token on the
    wire, so the retry 401s again and the turn aborts as non-retryable —
    only a gateway restart helped, because a cold process re-runs the
    exchange. Fix: force a fresh exchange (evict the cached exchanged JWT,
    then mint a new one) so the retry carries a valid IDE token. Mirrors the
    400 stale-credential recovery; the caller enforces the single-shot guard.
    """
    if not _is_copilot_provider(self):
        return False

    try:
        from hermes_cli.copilot_auth import (
            resolve_copilot_token,
            get_copilot_api_token,
            evict_cached_exchanged_token,
        )

        new_token, token_source = resolve_copilot_token()
    except Exception as exc:
        logger.debug("Copilot credential refresh failed: %s", exc)
        return False

    if not isinstance(new_token, str) or not new_token.strip():
        return False

    new_token = new_token.strip()

    # Force a fresh IDE-token exchange: the cached exchanged JWT is the thing
    # that expired ("401 IDE token expired"), so evict it and re-mint before
    # rebuilding the client. Fall back to the resolved (raw) token only if the
    # exchange itself is unavailable (network blip) — a client rebuild on the
    # raw token still clears stale client state and may recover on enterprise
    # seats where headers matter.
    try:
        evict_cached_exchanged_token(new_token)
        api_token, enterprise_base_url = get_copilot_api_token(new_token)
        if isinstance(api_token, str) and api_token.strip():
            new_token = api_token.strip()
            if enterprise_base_url:
                set_base_url(self, enterprise_base_url.rstrip("/"))
    except Exception as exc:
        logger.debug("Copilot 401 re-exchange failed, using resolved token: %s", exc)

    self.api_key = new_token
    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url
    _apply_client_headers_for_base_url(self, str(self.base_url or ""))

    if not _replace_primary_openai_client(self, reason="copilot_credential_refresh"):
        return False

    logger.info("Copilot credentials refreshed from %s", token_source)
    return True

def _try_recover_stale_copilot_credential(self) -> bool:
    """Force a fresh Copilot token exchange + client rebuild after a 400.

    Copilot surfaces a stale/degraded credential as a
    ``400 model_not_available_for_integrator`` /
    ``model_not_supported`` — NOT a clean 401 — so the normal 401 refresh
    path never fires. The most common trigger is a raw ``ghu_`` OAuth token
    that got seeded (and cached) when the startup token exchange degraded:
    the raw token routes the request to the restricted
    ``copilot-language-server`` integrator whose allowlist omits
    enterprise-only models (e.g. ``claude-opus-4.8``).

    Recovery = evict the poisoned cache entry, force a fresh exchange to
    mint the real ~437-char API token, re-apply the Copilot headers, and
    rebuild the shared client. Single-shot (guarded by the caller) so a
    genuinely unavailable model can't loop.
    """
    if not _is_copilot_provider(self):
        return False

    try:
        from hermes_cli.copilot_auth import (
            resolve_copilot_token,
            get_copilot_api_token,
            evict_cached_exchanged_token,
        )

        raw_token, token_source = resolve_copilot_token()
        if not isinstance(raw_token, str) or not raw_token.strip():
            return False
        raw_token = raw_token.strip()

        # Drop any cached (possibly degraded/raw) exchanged token so the
        # next exchange hits the network and mints a fresh one.
        evict_cached_exchanged_token(raw_token)

        api_token, enterprise_base_url = get_copilot_api_token(raw_token)
    except Exception as exc:
        logger.debug("Copilot stale-credential recovery failed: %s", exc)
        return False

    if not isinstance(api_token, str) or not api_token.strip():
        return False

    # If the exchange STILL degraded to the raw token, a rebuild won't help
    # — don't burn the single-shot retry on an identical request.
    if api_token == raw_token and not enterprise_base_url:

        logger.warning(
            "Copilot stale-credential recovery: exchange still degraded to "
            "raw token; skipping retry (network/exchange endpoint unavailable)."
        )
        return False

    self.api_key = api_token.strip()
    if enterprise_base_url:
        set_base_url(self, enterprise_base_url.rstrip("/"))
    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url
    _apply_client_headers_for_base_url(self, str(self.base_url or ""))

    if not _replace_primary_openai_client(self, reason="copilot_stale_credential_recovery"):
        return False

    logger.info("Copilot credentials re-exchanged after stale-credential 400 (source=%s)", token_source)
    return True

def _try_refresh_anthropic_client_credentials(self) -> bool:
    if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
        return False
    # Only refresh credentials for the native Anthropic provider.
    # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
    if self.provider != "anthropic":
        return False
    # Azure endpoints use static API keys — OAuth token rotation doesn't apply.
    # Refreshing would pick up ~/.claude/.credentials.json OAuth token and break auth.
    _base = getattr(self, "_anthropic_base_url", "") or ""
    if base_url_host_matches(_base, "azure.com"):
        return False

    try:
        from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

        new_token = resolve_anthropic_token()
    except Exception as exc:
        logger.debug("Anthropic credential refresh failed: %s", exc)
        return False

    if not isinstance(new_token, str) or not new_token.strip():
        return False
    new_token = new_token.strip()
    if new_token == self._anthropic_api_key:
        return False

    try:
        self._anthropic_client.close()
    except Exception:
        pass

    try:
        self._anthropic_client = build_anthropic_client(
            new_token,
            getattr(self, "_anthropic_base_url", None),
            timeout=get_provider_request_timeout(self.provider, self.model),
        )
    except Exception as exc:
        logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
        return False

    self._anthropic_api_key = new_token
    # Update OAuth flag — token type may have changed (API key ↔ OAuth).
    # Only treat as OAuth on native Anthropic; third-party endpoints using
    # the Anthropic protocol must not trip OAuth paths (#1739 & third-party
    # identity-injection guard).
    from agent.anthropic_adapter import _is_oauth_token
    self._is_anthropic_oauth = _is_oauth_token(new_token) if self.provider == "anthropic" else False
    return True

def _apply_client_headers_for_base_url(
    self,
    base_url: str,
    *,
    apply_user_headers: bool = True,
) -> None:
    from agent.auxiliary_client import (
        _AI_GATEWAY_HEADERS,
        build_nvidia_nim_headers,
        build_or_headers,
    )

    if base_url_host_matches(base_url, "openrouter.ai"):
        self._client_kwargs["default_headers"] = build_or_headers()
    elif base_url_host_matches(base_url, "ai-gateway.vercel.sh"):
        self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
    elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
        self._client_kwargs["default_headers"] = build_nvidia_nim_headers(base_url)
    elif base_url_host_matches(base_url, "api.routermint.com"):
        from agent.init_runtime import routermint_headers
        self._client_kwargs["default_headers"] = routermint_headers()
    elif base_url_host_matches(base_url, "githubcopilot.com"):
        from hermes_cli.models import copilot_default_headers

        self._client_kwargs["default_headers"] = copilot_default_headers()
    elif base_url_host_matches(base_url, "api.kimi.com"):
        from agent.auxiliary_client import _AI_GATEWAY_HEADERS
        self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
    elif base_url_host_matches(base_url, "portal.qwen.ai"):
        from agent.init_runtime import qwen_portal_headers
        self._client_kwargs["default_headers"] = qwen_portal_headers()
    elif base_url_host_matches(base_url, "chatgpt.com"):
        from agent.auxiliary_client import _codex_cloudflare_headers
        self._client_kwargs["default_headers"] = _codex_cloudflare_headers(
            self._client_kwargs.get("api_key", "")
        )
    elif base_url_host_matches(base_url, "x.ai"):
        # Cover both provider=xai and provider=xai-oauth (api.x.ai).
        from tools.xai_http import hermes_xai_default_headers

        self._client_kwargs["default_headers"] = hermes_xai_default_headers()
    else:
        # No URL-specific headers — check profile.default_headers before clearing.
        _ph_headers = None
        try:
            from providers import get_provider_profile as _gpf2
            _ph2 = _gpf2(self.provider)
            if _ph2 and _ph2.default_headers:
                _ph_headers = dict(_ph2.default_headers)
        except Exception:
            pass
        if _ph_headers:
            self._client_kwargs["default_headers"] = _ph_headers
        else:
            self._client_kwargs.pop("default_headers", None)

    # User-configured overrides win over URL/profile defaults for the same
    # route. A credential swap to another endpoint must not inherit them.
    if apply_user_headers:
        _apply_user_default_headers(self)

    # Per-provider extra HTTP headers (providers.<name>.extra_headers /
    # custom_providers[].extra_headers) — applied last so the most
    # specific config level survives credential swaps and rebuilds too.
    # SECURITY: values may carry credentials — never log them.
    if self.api_mode not in ("anthropic_messages", "bedrock_converse"):
        try:
            from hermes_cli.config import (
                apply_custom_provider_extra_headers_to_client_kwargs,
            )

            apply_custom_provider_extra_headers_to_client_kwargs(
                self._client_kwargs, base_url,
            )
        except Exception:
            logger.debug("custom-provider extra_headers skipped", exc_info=True)

def _apply_user_default_headers(self) -> None:
    """Merge user-configured request headers onto the OpenAI client.

    Reads ``model.default_headers`` from config.yaml and merges it onto
    ``self._client_kwargs["default_headers"]``, with user values taking
    precedence over provider- and SDK-supplied defaults.

    This exists for ``custom`` OpenAI-compatible endpoints sitting behind
    a gateway/WAF that rejects the OpenAI Python SDK's identifying headers
    (``User-Agent: OpenAI/Python ...``, ``X-Stainless-*``). Setting e.g.
    ``model.default_headers: {User-Agent: curl/8.7.1}`` lets the request
    reach such an upstream instead of failing with an opaque 4xx/502 even
    though the same body works under ``curl``. (#40033)

    Delegates the config read + merge to
    ``agent.auxiliary_client._apply_user_default_headers`` so the main and
    auxiliary clients can never drift on precedence or value handling.

    No-op for Anthropic/Bedrock modes, which don't use the OpenAI client,
    and when no overrides are configured.
    """
    if self.api_mode in ("anthropic_messages", "bedrock_converse"):
        return
    from agent.auxiliary_client import (
        _apply_user_default_headers as _merge_user_headers,
    )
    merged = _merge_user_headers(self._client_kwargs.get("default_headers"))
    if merged:
        self._client_kwargs["default_headers"] = merged

def _swap_credential(self, entry) -> None:
    runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")

    runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url
    self._credential_pool_entry_id = getattr(entry, "id", None)
    from hermes_cli.route_identity import normalize_route_base_url

    route_changed = normalize_route_base_url(self.base_url) != normalize_route_base_url(
        runtime_base
    )

    if self.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        self._anthropic_api_key = runtime_key
        self._anthropic_base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._anthropic_client = build_anthropic_client(
            runtime_key, self._anthropic_base_url,
            timeout=get_provider_request_timeout(self.provider, self.model),
        )
        self._is_anthropic_oauth = _is_oauth_token(runtime_key) if self.provider == "anthropic" else False
        self.api_key = runtime_key
        set_base_url(self, runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base)
        return

    self.api_key = runtime_key
    set_base_url(self, runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base)
    self._client_kwargs["api_key"] = self.api_key
    self._client_kwargs["base_url"] = self.base_url
    _reapply_route_client_config(self, route_changed=route_changed)
    _replace_primary_openai_client(self, reason="credential_rotation")

def _reapply_route_client_config(self, *, route_changed: bool) -> None:
    """Recompute route-derived client kwargs for the current ``self.base_url``.

    TLS material (``ssl_verify``/``ssl_ca_cert``) and default headers are
    derived from the endpoint, not the credential — any client rebuild
    that may have moved ``base_url`` must recompute them or the new
    endpoint inherits configuration computed for the old one. Shared by
    credential-pool rotation and the per-turn env refresh so the two
    paths cannot drift.
    """
    self._client_kwargs.pop("ssl_verify", None)
    self._client_kwargs.pop("ssl_ca_cert", None)
    try:
        from hermes_cli.config import (
            apply_custom_provider_tls_to_client_kwargs,
            get_compatible_custom_providers,
            load_config_readonly,
        )

        apply_custom_provider_tls_to_client_kwargs(
            self._client_kwargs,
            str(self.base_url or ""),
            get_compatible_custom_providers(load_config_readonly()),
        )
    except Exception:
        logger.debug(
            "custom-provider TLS resolution skipped on credential rotation",
            exc_info=True,
        )
    _apply_client_headers_for_base_url(self,
        self.base_url,
        apply_user_headers=not route_changed,
    )


def _credential_pool_may_recover_rate_limit(self) -> bool:
    """Whether a rate-limit retry should wait for same-provider credentials."""
    return pool_may_recover_from_rate_limit(self._credential_pool)

def _anthropic_messages_create(self, api_kwargs: dict, *, client: Any = None):
    # When a request-local client is supplied it was already credential-
    # refreshed in ``_create_request_anthropic_client``; only the shared
    # fallback path refreshes here.
    if client is None and self.api_mode == "anthropic_messages":
        _try_refresh_anthropic_client_credentials(self)
    # Defensive: strip Responses-only kwargs that can leak in under an
    # api_mode-flip race (the Anthropic SDK raises a non-retryable
    # TypeError on them). See #31673.
    from agent.anthropic_adapter import create_anthropic_message
    return create_anthropic_message(
        client or self._anthropic_client,
        api_kwargs,
        log_prefix=getattr(self, "log_prefix", ""),
        prefer_stream=not bool(getattr(self, "_disable_streaming", False)),
        # Rate-limit + credits state live in response headers, which the
        # parsed Message drops. No-ops on providers that don't send the
        # matching header families (x-ratelimit-* / x-nous-credits-*).
        on_response=self._capture_anthropic_response_headers,
    )

def _rebuild_anthropic_client(self) -> None:
    """Rebuild the Anthropic client after an interrupt or stale call.

    Handles both direct Anthropic and Bedrock-hosted Anthropic models
    correctly — rebuilding with the Bedrock SDK when provider is bedrock,
    rather than always falling back to build_anthropic_client() which
    requires a direct Anthropic API key.

    Honors ``self._oauth_1m_beta_disabled`` (set by the reactive recovery
    path when an OAuth subscription rejects the 1M-context beta) so the
    rebuilt client carries the reduced beta set.
    """
    _drop_1m = bool(getattr(self, "_oauth_1m_beta_disabled", False))
    if getattr(self, "provider", None) == "bedrock":
        from agent.anthropic_adapter import build_anthropic_bedrock_client
        region = getattr(self, "_bedrock_region", "us-east-1") or "us-east-1"
        self._anthropic_client = build_anthropic_bedrock_client(region)
    else:
        from agent.anthropic_adapter import build_anthropic_client
        self._anthropic_client = build_anthropic_client(
            self._anthropic_api_key,
            getattr(self, "_anthropic_base_url", None),
            timeout=get_provider_request_timeout(self.provider, self.model),
            drop_context_1m_beta=_drop_1m,
        )

def _interruptible_api_call(self, api_kwargs: dict):
    """Forwarder — see ``agent.chat_completion_helpers.interruptible_api_call``."""
    from agent.chat_completion_helpers import interruptible_api_call
    return interruptible_api_call(self, api_kwargs)

def _try_activate_fallback(self, reason: "FailoverReason | None" = None) -> bool:
    """Forwarder — see ``agent.chat_completion_helpers.try_activate_fallback``."""
    from agent.chat_completion_helpers import try_activate_fallback
    return try_activate_fallback(self, reason)

def _has_pending_fallback(self) -> bool:
    """Whether a fallback provider is actually available to switch to.

    Used to gate user-facing "trying fallback..." status so we don't
    announce a fallback that will never be attempted (the user has no
    fallback chain configured).  Mirrors the early-return guard in
    ``try_activate_fallback`` (#35314, #17446).
    """
    chain = getattr(self, "_fallback_chain", None) or []
    index = getattr(self, "_fallback_index", 0)
    return index < len(chain)



def _content_has_image_parts(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:

        if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
            return True
    return False

def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
    header, _, data = str(image_url or "").partition(",")
    if len(data) > _MAX_DATA_URL_BASE64_BYTES:
        logger.warning(
            "data-URL payload too large (%d bytes), skipping", len(data)
        )
        return "", None
    mime = "image/jpeg"
    if header.startswith("data:"):
        mime_part = header[len("data:"):].split(";", 1)[0].strip()
        if mime_part.startswith("image/"):
            mime = mime_part
    suffix = {
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
    }.get(mime, ".jpg")
    tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
    try:
        with tmp:
            tmp.write(base64.b64decode(data))
    except Exception:
        # delete=False means a corrupt/unsupported data URL would otherwise
        # leak a zero-byte temp file on every failed materialization.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    path = Path(tmp.name)
    return str(path), path

def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
    cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
    cached = self._anthropic_image_fallback_cache.get(cache_key)
    if cached:
        return cached

    role_label = {
        "assistant": "assistant",
        "tool": "tool result",
    }.get(role, "user")
    analysis_prompt = (
        "Describe everything visible in this image in thorough detail. "
        "Include any text, code, UI, data, objects, people, layout, colors, "
        "and any other notable visual information."
    )

    vision_source = str(image_url or "")
    cleanup_path: Optional[Path] = None
    if vision_source.startswith("data:"):
        vision_source, cleanup_path = _materialize_data_url_for_vision(vision_source)

    description = ""
    try:
        from tools.vision_tools import vision_analyze_tool

        result_json = asyncio.run(
            vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
        )
        result = json.loads(result_json) if isinstance(result_json, str) else {}
        description = (result.get("analysis") or "").strip()
    except Exception as e:
        description = f"Image analysis failed: {e}"
    finally:
        if cleanup_path and cleanup_path.exists():
            try:
                cleanup_path.unlink()
            except OSError:
                pass

    if not description:
        description = "Image analysis failed."

    note = f"[The {role_label} attached an image. Here's what it contains:\n{description}]"
    if vision_source and not str(image_url or "").startswith("data:"):
        note += (
            f"\n[If you need a closer look, use vision_analyze with image_url: {vision_source}]"
        )

    self._anthropic_image_fallback_cache[cache_key] = note
    return note

def _model_supports_vision(self) -> bool:
    """Return True if the active provider+model reports native vision.

    Used to decide whether to strip image content parts from API-bound
    messages (for non-vision models) or let the provider adapter handle
    them natively (for vision-capable models).

    Resolution order (see ``agent.image_routing._supports_vision_override``):
      1. ``model.supports_vision`` (top-level, single-model shortcut)
      2. ``providers.<provider>.models.<model>.supports_vision``
      3. models.dev capability lookup
    Custom/local models absent from models.dev would otherwise be
    misclassified as non-vision and have their images stripped.
    """
    try:
        from hermes_cli.config import load_config
        from agent.image_routing import _lookup_supports_vision
        cfg = load_config()
        provider = (getattr(self, "provider", "") or "").strip()
        model = (getattr(self, "model", "") or "").strip()
        return _lookup_supports_vision(provider, model, cfg) is True
    except Exception:
        return False

def _provider_supports_vision_tool_messages(self) -> bool:
    """Return True if the active provider accepts list-type tool content.

    Some providers (e.g. Xiaomi MiMo) support multimodal user messages
    but reject list-type tool message content with 400 errors.  This
    checks the provider profile's ``supports_vision_tool_messages`` field.
    """
    try:
        from providers import get_provider_profile
        provider = (getattr(self, "provider", "") or "").strip()
        profile = get_provider_profile(provider)
        if profile is not None:
            return getattr(profile, "supports_vision_tool_messages", True)
    except Exception:
        pass
    return True  # default: assume compatible

def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
    if not _content_has_image_parts(content):
        return content

    text_parts: List[str] = []
    image_notes: List[str] = []
    for part in content:
        if isinstance(part, str):
            if part.strip():
                text_parts.append(part.strip())
            continue

        if not isinstance(part, dict):
            continue

        ptype = part.get("type")
        if ptype in {"text", "input_text"}:
            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)
            continue

        if ptype in {"image_url", "input_image"}:
            image_data = part.get("image_url", {})
            image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
            if image_url:
                image_notes.append(_describe_image_for_anthropic_fallback(self, image_url, role))
            else:
                image_notes.append("[An image was attached but no image source was available.]")
            continue

        text = str(part.get("text", "") or "").strip()
        if text:
            text_parts.append(text)

    prefix = "\n\n".join(note for note in image_notes if note).strip()
    suffix = "\n".join(text for text in text_parts if text).strip()
    if prefix and suffix:
        return f"{prefix}\n\n{suffix}"
    if prefix:
        return prefix
    if suffix:
        return suffix
    return "[A multimodal message was converted to text for Anthropic compatibility.]"

def _get_transport(self, api_mode: str = None):
    """Return the cached transport for the given (or current) api_mode.

    Lazy-initializes on first call per api_mode. Returns None if no
    transport is registered for the mode.
    """
    mode = api_mode or self.api_mode
    cache = getattr(self, "_transport_cache", None)
    if cache is None:
        cache = {}
        self._transport_cache = cache
    t = cache.get(mode)
    if t is None:
        from agent.transports import get_transport
        t = get_transport(mode)
        cache[mode] = t
    return t

def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
    # Fast exit when no message carries image content at all.
    if not any(
        isinstance(msg, dict) and _content_has_image_parts(msg.get("content"))
        for msg in api_messages
    ):
        return api_messages

    # The Anthropic adapter (agent/anthropic_adapter.py:_convert_content_part_to_anthropic)
    # already translates OpenAI-style image_url/input_image parts into
    # native Anthropic ``{"type": "image", "source": ...}`` blocks. When
    # the active model supports vision we let the adapter do its job and
    # skip this legacy text-fallback preprocessor entirely.
    if _model_supports_vision(self):
        return api_messages

    # Non-vision Anthropic model (rare today, but keep the fallback for
    # compat): replace each image part with a vision_analyze text note.
    transformed = copy.deepcopy(api_messages)
    for msg in transformed:
        if not isinstance(msg, dict):
            continue
        msg["content"] = _preprocess_anthropic_content(self,
            msg.get("content"),
            str(msg.get("role", "user") or "user"),
        )
    return transformed

def _prepare_messages_for_non_vision_model(self, api_messages: list) -> list:
    """Strip native image parts when the active model lacks vision.

    Runs on the chat.completions / codex_responses paths. Vision-capable
    models pass through unchanged (provider and any downstream translator
    handle the image parts natively). Non-vision models get each image
    replaced by a cached vision_analyze text description so the turn
    doesn't fail with "model does not support image input".
    """
    if not any(
        isinstance(msg, dict) and _content_has_image_parts(msg.get("content"))
        for msg in api_messages
    ):
        return api_messages

    if _model_supports_vision(self):
        return api_messages

    transformed = copy.deepcopy(api_messages)
    for msg in transformed:
        if not isinstance(msg, dict):
            continue
        # Reuse the Anthropic text-fallback preprocessor — the behaviour is
        # identical (walk content parts, replace images with cached
        # descriptions, merge back into a single text or structured
        # content). Naming is historical.
        msg["content"] = _preprocess_anthropic_content(self,
            msg.get("content"),
            str(msg.get("role", "user") or "user"),
        )
    return transformed

def _tool_result_content_for_active_model(self, tool_name: str, result: Any) -> Any:
    """Return the tool message content that is safe for the active model.

    Multimodal tool results normally unwrap to OpenAI-style content parts so
    vision-capable models can inspect screenshots.  Text-only providers must
    not receive those image parts, because a rejected tool result becomes
    part of the canonical history and can make the next user turn fail before
    the agent has a chance to recover.
    """
    if not _is_multimodal_tool_result(result):
        return result

    content = result.get("content") or []
    if not _content_has_image_parts(content):
        return content

    if _model_supports_vision(self):
        # Vision-capable on paper — but if the provider rejects list-type
        # tool content (e.g. Xiaomi MiMo's 400 "text is not set"), or if
        # we've already learned this lesson in-session, short-circuit to
        # a text summary so we don't burn a round-trip relearning it.
        if not _provider_supports_vision_tool_messages(self):
            logger.debug(
                "Tool %s: provider %s does not accept list-type tool "
                "content — sending text summary",
                tool_name, getattr(self, "provider", ""),
            )
            return _multimodal_text_summary(result)
        key = (
            (getattr(self, "provider", "") or "").strip().lower(),
            (getattr(self, "model", "") or "").strip(),
        )
        no_list = getattr(self, "_no_list_tool_content_models", None)
        if no_list and key in no_list:
            logger.debug(
                "Tool %s: model %s/%s known to reject list-type tool "
                "content this session — sending text summary",
                tool_name, key[0], key[1],
            )
            return _multimodal_text_summary(result)
        return content

    summary = _multimodal_text_summary(result)
    if tool_name == "computer_use":
        return json.dumps({
            "error": (
                "computer_use returned screenshot/image content, but the active "
                "model/provider does not support image input. Switch to a "
                "vision-capable model for desktop computer use, or use browser "
                "tools for browser tasks."
            ),
            "text_summary": summary,
        })

    logger.warning(
        "Tool %s returned image content for non-vision model %s/%s; "
        "falling back to text summary",
        tool_name,
        self.provider,
        self.model,
    )
    return summary

def _try_shrink_image_parts_in_messages(
    self,
    api_messages: list,
    *,
    max_dimension: int = 8000,
) -> bool:
    """Forwarder — see ``agent.conversation_compression.try_shrink_image_parts_in_messages``."""
    from agent.conversation_compression import try_shrink_image_parts_in_messages
    return try_shrink_image_parts_in_messages(
        api_messages,
        max_dimension=max_dimension,
    )

def _try_strip_image_parts_from_tool_messages(
    self,
    api_messages: list,
    *,
    remember_model: bool = True,
) -> bool:
    """Downgrade list-type tool messages to text summaries in-place.

    Recovery path for providers that reject list-type tool message content
    (e.g. Xiaomi MiMo's 400 "text is not set"; see issue #27344).  Walks
    ``api_messages`` for any ``role: "tool"`` message whose ``content`` is
    a list containing image parts, replaces the content with the existing
    text part(s) (or a minimal placeholder if none survive), and by default
    records the active (provider, model) in
    ``self._no_list_tool_content_models`` so subsequent
    ``_tool_result_content_for_active_model`` calls in this session
    preemptively downgrade screenshots without a round-trip.

    413 payload-size recovery passes ``remember_model=False`` because that
    error means this request body was too large, not that the provider/model
    rejects list-type tool content in general.


    Returns True when at least one tool message was downgraded — the
    caller (the 400 recovery branch in ``agent.conversation_loop``) uses
    this to decide whether to retry the API call with the modified
    history or surface the original error.
    """
    if not isinstance(api_messages, list):
        return False

    if remember_model:
        # Record (provider, model) so we don't relearn this lesson.
        key = (
            (getattr(self, "provider", "") or "").strip().lower(),
            (getattr(self, "model", "") or "").strip(),
        )
        if not hasattr(self, "_no_list_tool_content_models"):
            self._no_list_tool_content_models = set()
        if key[1]:  # only record when we actually have a model id
            self._no_list_tool_content_models.add(key)

    changed = False
    for msg in api_messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        # Salvage any text parts so the model still sees some signal.
        text_parts: List[str] = []
        had_image = False
        for part in content:
            if not isinstance(part, dict):
                if isinstance(part, str) and part.strip():
                    text_parts.append(part.strip())
                continue
            ptype = part.get("type")
            if ptype == "image_url" or ptype == "input_image":
                had_image = True
                continue
            if ptype in {"text", "input_text"}:
                text = str(part.get("text") or "").strip()
                if text:
                    text_parts.append(text)

        if not had_image:
            # List-type content but no image parts — leave alone (some
            # providers reject ANY list content, but stripping a
            # text-only list doesn't reduce ambiguity; let the caller
            # surface the original error if this turns out to be the
            # case).
            continue

        if text_parts:
            msg["content"] = "\n\n".join(text_parts)
        else:
            msg["content"] = (
                "[image content removed — provider does not accept "
                "list-type tool message content]"
            )
        changed = True

    return changed

def _anthropic_preserve_dots(self) -> bool:
    """True when using an anthropic-compatible endpoint that preserves dots in model names.
    Alibaba/DashScope keeps dots (e.g. qwen3.5-plus).
    MiniMax keeps dots (e.g. MiniMax-M2.7).
    Xiaomi MiMo keeps dots (e.g. mimo-v2.5, mimo-v2.5-pro).
    OpenCode Go/Zen keeps dots for non-Claude models (e.g. minimax-m2.5-free).
    ZAI/Zhipu keeps dots (e.g. glm-4.7, glm-5.1).
    AWS Bedrock uses dotted inference-profile IDs
    (e.g. ``global.anthropic.claude-opus-4-7``,
    ``us.anthropic.claude-sonnet-4-5-20250929-v1:0``) and rejects
    the hyphenated form with
    ``HTTP 400 The provided model identifier is invalid``.
    Regression for #11976; mirrors the opencode-go fix for #5211
    (commit f77be22c), which extended this same allowlist."""
    if (getattr(self, "provider", "") or "").lower() in {
        "alibaba", "minimax", "minimax-cn",
        "opencode-go", "opencode-zen",
        "zai", "bedrock",
        "xiaomi", "vertex",
    }:
        return True
    base = (getattr(self, "base_url", "") or "").lower()
    host = base_url_hostname(base)
    return (
        "dashscope" in host
        or base_url_host_matches(base, "aliyuncs.com")
        or "minimax" in host
        or (base_url_host_matches(base, "opencode.ai") and "/zen/" in base)
        or base_url_host_matches(base, "bigmodel.cn")
        or base_url_host_matches(base, "xiaomimimo.com")
        # Vertex AI OpenAI-compat endpoint — Gemini model ids keep dots
        # (e.g. google/gemini-3.5-flash); the hyphenated form is wrong.
        or base_url_host_matches(base, "aiplatform.googleapis.com")
        # AWS Bedrock runtime endpoints — defense-in-depth when
        # ``provider`` is unset but ``base_url`` still names Bedrock.
        or host.startswith("bedrock-runtime.")
    )

def _is_qwen_portal(self) -> bool:
    """Return True when the base URL targets Qwen Portal."""
    return base_url_host_matches(self._base_url_lower, "portal.qwen.ai")

def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
    prepared = copy.deepcopy(api_messages)
    if not prepared:
        return prepared

    for msg in prepared:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            # Normalize: convert bare strings to text dicts, keep dicts as-is.
            # deepcopy already created independent copies, no need for dict().
            normalized_parts = []
            for part in content:
                if isinstance(part, str):
                    normalized_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    normalized_parts.append(part)
            if normalized_parts:
                msg["content"] = normalized_parts

    # Inject cache_control on the last part of the system message.
    for msg in prepared:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                content[-1]["cache_control"] = {"type": "ephemeral"}
            break

    return prepared

def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
    """In-place variant — mutates an already-copied message list."""
    if not messages:
        return

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            normalized_parts = []
            for part in content:
                if isinstance(part, str):
                    normalized_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    normalized_parts.append(part)
            if normalized_parts:
                msg["content"] = normalized_parts

    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                content[-1]["cache_control"] = {"type": "ephemeral"}
            break

def _build_api_kwargs(self, api_messages: list, tools_for_api: Optional[list] = None) -> dict:
    """Forwarder — see ``agent.chat_completion_helpers.build_api_kwargs``."""
    from agent.chat_completion_helpers import build_api_kwargs
    return build_api_kwargs(self, api_messages, tools_for_api=tools_for_api)

def _supports_reasoning_extra_body(self) -> bool:
    """Return True when reasoning extra_body is safe to send for this route/model.

    OpenRouter forwards unknown extra_body fields to upstream providers.
    Some providers/routes reject `reasoning` with 400s, so gate it to
    known reasoning-capable model families and direct Nous Portal.
    """
    if base_url_host_matches(self._base_url_lower, "nousresearch.com"):
        return True
    if base_url_host_matches(self._base_url_lower, "ai-gateway.vercel.sh"):
        return True
    if (
        base_url_host_matches(self._base_url_lower, "models.github.ai")
        or base_url_host_matches(self._base_url_lower, "githubcopilot.com")
    ):
        try:
            from hermes_cli.models import github_model_reasoning_efforts

            return bool(github_model_reasoning_efforts(self.model))
        except Exception:

            return False
    if (self.provider or "").strip().lower() == "lmstudio":
        opts = _lmstudio_reasoning_options_cached(self)
        # "off-only" (or absent) means no real reasoning capability.
        return any(opt and opt != "off" for opt in opts)
    # Ollama Cloud (and any Ollama-compatible server): the native
    # /api/show capabilities list is authoritative — emit reasoning_effort
    # only for models that declare the "thinking" capability. deepseek-v4
    # has it; gemma3 / qwen3-coder don't. Cached per (model, base_url).
    if base_url_host_matches(self._base_url_lower, "ollama.com"):
        return _ollama_supports_thinking_cached(self)
    if not _is_openrouter_url(self):
        return False
    if base_url_host_matches(self._base_url_lower, "api.mistral.ai"):
        return False

    model = (self.model or "").lower()
    # Live-catalog metadata first (ported from
    # PrimeIntellect-ai/prime-agent#1258): OpenRouter's /v1/models entries
    # advertise reasoning support via supported_parameters + a reasoning
    # object, which covers every routed vendor without a hand-maintained
    # prefix list. The static prefix allowlist below repeatedly went
    # stale one vendor at a time (nvidia/ missing → #75386; same class
    # as tencent/, xiaomi/ additions before it) — metadata makes new
    # vendors work without a code change. One catalog fetch per process,
    # cached; unknown (catalog unreachable / unlisted model) falls back
    # to the static list.
    try:
        from hermes_cli.models import (
            openrouter_model_reasoning_capabilities,
            warm_openrouter_reasoning_caps_async,
        )
        caps = openrouter_model_reasoning_capabilities(self.model)
        if caps is None:
            # Cache cold (no picker run this process) — warm it in the
            # background so subsequent turns get metadata; never block
            # this turn on HTTP.
            warm_openrouter_reasoning_caps_async()
    except Exception:
        caps = None
    if caps is not None:
        return bool(caps.get("supports_reasoning"))
    reasoning_model_prefixes = (
        "deepseek/",
        "anthropic/",
        "openai/",
        "x-ai/",
        "google/gemini-2",
        "google/gemma-4",
        "qwen/qwen3",
        "tencent/hy3",
        "xiaomi/",
    )
    return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

def _lmstudio_reasoning_options_cached(self) -> list[str]:
    """Probe LM Studio's published reasoning ``allowed_options`` once per
    (model, base_url). The list (e.g. ``["off","on"]`` or
    ``["off","minimal","low"]``) is needed both for the supports-reasoning
    gate and for clamping the emitted ``reasoning_effort`` so toggle-style
    models don't 400 on ``high``. Cache is keyed on (model, base_url) so
    ``/model`` swaps and base-URL changes don't reuse a stale list.
    Non-empty results are cached permanently (model capabilities don't
    change). Empty results (transient probe failure OR genuinely
    non-reasoning model) are cached with a 60-second TTL to avoid an
    HTTP round-trip on every turn while still retrying reasonably soon.
    """
    import time as _time

    cache = getattr(self, "_lm_reasoning_opts_cache", None)
    if cache is None:
        cache = self._lm_reasoning_opts_cache = {}
    key = (self.model, self.base_url)
    cached = cache.get(key)
    if cached is not None:
        opts, ts = cached
        # Non-empty → permanent. Empty → 60s TTL.
        if opts or (_time.monotonic() - ts) < 60:
            return opts
    try:
        from hermes_cli.models import lmstudio_model_reasoning_options
        opts = lmstudio_model_reasoning_options(
            self.model, self.base_url, getattr(self, "api_key", ""),
        )
    except Exception:
        opts = []
    cache[key] = (opts, _time.monotonic())
    return opts

def _ollama_supports_thinking_cached(self) -> bool:
    """Probe Ollama's ``/api/show`` capabilities once per (model, base_url).

    Returns True only when the model declares the ``thinking`` capability.
    Caching mirrors the LM Studio probe: a True/False result is permanent
    (capabilities don't change), while a probe failure (None) is cached
    with a 60-second TTL so a transient outage doesn't suppress reasoning
    for the rest of the session but also doesn't round-trip every turn.
    """
    import time as _time

    cache = getattr(self, "_ollama_thinking_cache", None)
    if cache is None:
        cache = self._ollama_thinking_cache = {}
    key = (self.model, self.base_url)
    cached = cache.get(key)
    if cached is not None:
        supported, ts = cached
        # Definitive True/False → permanent. Unknown (None) → 60s TTL.
        if supported is not None or (_time.monotonic() - ts) < 60:
            return bool(supported)
    try:
        from hermes_cli.models import ollama_model_supports_thinking
        supported = ollama_model_supports_thinking(
            self.model, self.base_url, getattr(self, "api_key", "")
        )
    except Exception:
        supported = None
    cache[key] = (supported, _time.monotonic())
    return bool(supported)

def _resolve_lmstudio_summary_reasoning_effort(self) -> Optional[str]:
    """Resolve a safe top-level ``reasoning_effort`` for LM Studio.

    The iteration-limit summary path calls ``chat.completions.create()``
    directly, bypassing the transport. Share the helper so the two paths
    can't drift on effort resolution and clamping.
    """
    from agent.lmstudio_reasoning import resolve_lmstudio_effort
    return resolve_lmstudio_effort(
        self.reasoning_config,
        _lmstudio_reasoning_options_cached(self),
    )

def _github_models_reasoning_extra_body(self) -> dict | None:
    """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
    try:
        from hermes_cli.models import github_model_reasoning_efforts
    except Exception:
        return None

    supported_efforts = github_model_reasoning_efforts(self.model)
    if not supported_efforts:
        return None

    if self.reasoning_config and isinstance(self.reasoning_config, dict):
        if self.reasoning_config.get("enabled") is False:
            return None
        requested_effort = str(
            self.reasoning_config.get("effort", "medium")
        ).strip().lower()
    else:
        requested_effort = "medium"

    if requested_effort == "xhigh" and "xhigh" not in supported_efforts and "high" in supported_efforts:
        requested_effort = "high"
    elif requested_effort not in supported_efforts:
        if requested_effort == "minimal" and "low" in supported_efforts:

            requested_effort = "low"
        elif "medium" in supported_efforts:
            requested_effort = "medium"
        else:
            requested_effort = supported_efforts[0]

    return {"effort": requested_effort}

def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
    """Forwarder — see ``agent.chat_completion_helpers.build_assistant_message``."""
    from agent.chat_completion_helpers import build_assistant_message
    return build_assistant_message(self, assistant_message, finish_reason)

def _needs_thinking_reasoning_pad(self) -> bool:
    """Return True when the active provider enforces reasoning_content echo-back.

    DeepSeek v4 thinking and Kimi / Moonshot thinking both reject replays
    of assistant tool-call messages that omit ``reasoning_content`` (refs
    #15250, #17400). Xiaomi MiMo thinking mode has the same requirement.

    Result cached on the AgentState instance keyed by (provider, model,
    base_url); invalidated whenever ``switch_model()`` /
    ``_try_activate_fallback()`` mutate any of those. This is hot — the
    agent loop hits ~16 invocations per turn, each of which would
    otherwise re-run ~5 ``base_url_host_matches`` (and therefore
    ``urlparse``) calls under it. Caching drops the per-turn cost from
    ~5us × 16 = ~80us to <1us.
    """
    key = (self.provider, self.model, getattr(self, "_base_url_lower", self.base_url))
    cached = getattr(self, "_thinking_pad_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    result = (
        _needs_deepseek_tool_reasoning(self)
        or _needs_kimi_tool_reasoning(self)
        or _needs_mimo_tool_reasoning(self)
    )
    self._thinking_pad_cache = (key, result)
    return result

def _needs_kimi_tool_reasoning(self) -> bool:
    """Return True when the current provider is Kimi / Moonshot thinking mode.

    Kimi ``/coding`` and Moonshot thinking mode both require
    ``reasoning_content`` on every assistant tool-call message; omitting
    it causes the next replay to fail with HTTP 400.

    Detection is host-driven, not model-name-driven: aggregators like
    OpenRouter that re-export Kimi/Moonshot models speak their own
    protocol and reject ``reasoning_content`` echoes. We only enable the
    kimi-reasoning replay when the request actually targets a
    kimi/moonshot endpoint or the dedicated kimi-coding provider.

    Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
    """
    from agent.message_sanitization import matches_reasoning_echo_family
    return matches_reasoning_echo_family(
        "kimi", self.provider, None, self.base_url
    )

def _needs_deepseek_tool_reasoning(self) -> bool:
    """Return True when the current provider is DeepSeek thinking mode.

    DeepSeek V4 thinking mode requires ``reasoning_content`` on every
    assistant tool-call turn; omitting it causes HTTP 400 when the
    message is replayed in a subsequent API request (#15250).

    Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
    """
    from agent.message_sanitization import matches_reasoning_echo_family
    return matches_reasoning_echo_family(
        "deepseek", (self.provider or "").lower(), self.model, self.base_url
    )

def _needs_mimo_tool_reasoning(self) -> bool:
    """Return True when the current provider is Xiaomi MiMo thinking mode.

    MiMo thinking mode requires ``reasoning_content`` on every assistant
    tool-call message when replaying history; omitting it causes HTTP 400.
    Refs: https://platform.xiaomimimo.com/docs/zh-CN/usage-guide/passing-back-reasoning_content

    Rule table owner: ``agent.message_sanitization.reasoning_echo_family``.
    """
    from agent.message_sanitization import matches_reasoning_echo_family
    return matches_reasoning_echo_family(
        "mimo", (self.provider or "").lower(), self.model, self.base_url
    )



def _sanitize_tool_calls_for_strict_api(api_msg: dict, model: "str | None" = None) -> dict:
    """Strip Codex Responses API fields from tool_calls for strict providers.

    Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
    validate the Chat Completions schema and reject unknown fields (call_id,
    response_item_id) with 400 or 422 errors. These fields are preserved in
    the internal message history — this method only modifies the outgoing
    API copy.

    ``extra_content`` (Gemini thought_signature) is also stripped — strict
    providers reject it with "Extra inputs are not permitted" — UNLESS the
    outgoing ``model`` is itself Gemini-family, in which case it must be
    replayed (Gemini 3 thinking models 400 without it). Defaults to
    stripping when no model is supplied.

    Creates new tool_call dicts rather than mutating in-place, so the
    original messages list retains call_id/response_item_id for Codex
    Responses API compatibility (e.g. if the session falls back to a
    Codex provider later).

    Fields stripped: call_id, response_item_id, extra_content (model-gated)
    """
    tool_calls = api_msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return api_msg
    from agent.transports.chat_completions import _model_consumes_thought_signature
    _STRIP_KEYS = {"call_id", "response_item_id"}
    if not _model_consumes_thought_signature(model):
        _STRIP_KEYS = _STRIP_KEYS | {"extra_content"}
    api_msg["tool_calls"] = [
        {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
        if isinstance(tc, dict) else tc
        for tc in tool_calls
    ]
    return api_msg


def _should_sanitize_tool_calls(self) -> bool:
    """Determine if tool_calls need sanitization for strict APIs.

    Codex Responses API uses fields like call_id and response_item_id
    that are not part of the standard Chat Completions schema. These
    fields must be stripped when calling any other API to avoid
    validation errors (400 Bad Request).

    Returns:
        bool: True if sanitization is needed (non-Codex API), False otherwise.
    """
    return self.api_mode != "codex_responses"

def sync_credential_pool_entry_id(agent) -> None:
    """Rebind ``agent._credential_pool_entry_id`` from the current pool + key.

    OAuth refreshes can replace the runtime token before a failed request is
    recovered, so the mutable API-key value alone cannot reliably attribute
    the failure to its source entry.  This resolves the stable pool-entry ID
    for the agent's current ``api_key`` and clears it when no pool is bound.
    """
    pool = getattr(agent, "_credential_pool", None)
    try:
        agent._credential_pool_entry_id = (
            pool.entry_id_for_api_key(getattr(agent, "api_key", None))
            if pool is not None
            else None
        )
    except Exception:
        agent._credential_pool_entry_id = None

def recover_with_credential_pool(
    agent,
    *,
    status_code: Optional[int],
    has_retried_429: bool,
    classified_reason: Optional[FailoverReason] = None,
    error_context: Optional[Dict[str, Any]] = None,
    billing_unverified: bool = False,
) -> tuple[bool, bool]:
    """Attempt credential recovery via pool rotation.

    Returns (recovered, has_retried_429).
    On rate limits: first occurrence retries same credential (sets flag True).
                    second consecutive failure rotates to next credential.
    On billing exhaustion: immediately rotates.
    On auth failures: attempts token refresh before rotating.

    `classified_reason` lets the recovery path honor the structured error
    classifier instead of relying only on raw HTTP codes. This matters for
    providers that surface billing/rate-limit/auth conditions under a
    different status code, such as Anthropic returning HTTP 400 for
    "out of extra usage".

    `billing_unverified` marks a billing verdict that rests on an ambiguous
    body (``ClassifiedError.billing_unverified``, #82154): the pool persists
    it as ``billing_unverified`` so the exhausted entry gets a short cooldown
    instead of the one-hour billing bench — the same 400 can be a
    content-filter rejection that leaves the credential healthy.
    """
    import agent.error_reporting as error_reporting
    import agent.provider_runtime as provider_runtime
    pool = agent._credential_pool
    if pool is None:
        return False, has_retried_429

    # Defensive guard: if a fallback provider is active and its provider name
    # doesn't match the pool's provider, the pool belongs to the PRIMARY
    # provider.  Mutating it based on fallback errors would corrupt the
    # primary's credential state (see #33088) and, via _swap_credential,
    # overwrite the agent's base_url back to the primary's endpoint — every
    # subsequent request then goes to the wrong host and 404s (see #33163).
    # The pool should only act when the agent is still on the same provider
    # that seeded the pool.
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    pool_provider = (getattr(pool, "provider", "") or "").strip().lower()
    # Guard: skip credential pool recovery when the pool is scoped to a
    # different provider than the agent.  Only guard when the pool has a
    # known provider — an empty pool provider means "unscoped" (applies to
    # any provider).  An empty agent provider is treated as a mismatch
    # because swapping the pool's credentials would set base_url/api_key
    # without fixing the empty provider field, leaving the agent in a
    # corrupted state (provider="" model="").
    if pool_provider and current_provider != pool_provider:
        # Custom endpoints use two naming conventions for the SAME provider:
        # the agent carries the generic ``custom`` label while the pool is
        # keyed ``custom:<name>`` (see CUSTOM_POOL_PREFIX). A literal string
        # compare treats them as a mismatch and skips recovery for every
        # custom-provider user — 401s/429s then burn the full retry cycle
        # with no rotation or refresh. Accept the pair as matching only when
        # the agent's CURRENT base_url actually resolves to this pool key,
        # so a fallback provider (or a different custom endpoint) still
        # triggers the guard.
        _custom_match = False
        if current_provider == "custom" and pool_provider.startswith("custom:"):
            try:
                from agent.credential_pool import get_custom_provider_pool_key
                _agent_base = (getattr(agent, "base_url", "") or "").strip()
                _custom_match = bool(_agent_base) and (
                    (get_custom_provider_pool_key(_agent_base) or "").strip().lower()
                    == pool_provider
                )
            except Exception:
                _custom_match = False
        if not _custom_match:
            logger.warning(
                "Credential pool provider mismatch: pool=%s, agent=%s — "
                "skipping pool mutation to avoid cross-provider contamination",
                pool_provider, current_provider,
            )
            return False, has_retried_429

    # Attribute the failure to the API key the agent actually dispatched the
    # request with, not to pool.current(). The current() pointer is shared,
    # mutable state — round-robin select() advances it on every call, and
    # concurrent turns or a second process (gateway/dashboard) reloading the
    # pool reset it to None — so by the time recovery runs it routinely points
    # at a DIFFERENT, healthy entry. Marking that entry exhausted copies this
    # request's error/reset time onto it and can take the whole pool offline
    # from a single rate-limited key (#43747). ``_swap_credential`` keeps
    # ``agent.api_key`` in sync with the entry in use, so it identifies the
    # failing entry exactly; fall back to current()'s key only when the agent
    # carries no key at all.
    _api_key_hint = getattr(agent, "api_key", None) or None
    _raw_credential_id = getattr(agent, "_credential_pool_entry_id", None)
    _credential_id = (
        _raw_credential_id
        if isinstance(_raw_credential_id, str) and _raw_credential_id
        else None
    )
    if not _api_key_hint:
        _cur = pool.current()
        if _cur:
            _api_key_hint = getattr(_cur, "runtime_api_key", None)
            if not _credential_id:
                _current_id = getattr(_cur, "id", None)
                if isinstance(_current_id, str) and _current_id:
                    _credential_id = _current_id

    def _rotate_failed_credential(rotate_status: int):
        kwargs = {
            "status_code": rotate_status,
            "error_context": error_context,
            "api_key_hint": _api_key_hint,
        }
        if _credential_id:
            kwargs["credential_id"] = _credential_id
        # Hand the pool the classified semantics, not just the status. A
        # billing 403 (OpenRouter "key limit exceeded", xAI spending limit)
        # and an edge-throttle 403 are the same number but need opposite
        # cooldowns — the pool can only tell them apart if we say which.
        # ``effective_reason`` is resolved below; this closure runs after.
        if effective_reason is not None:
            _failure_reason = effective_reason.value
            if effective_reason == FailoverReason.billing and billing_unverified:
                # Ambiguous billing body (#82154): persist the ambiguity so
                # the cooldown is sized as transient, not a 1-hour bench.
                from agent.credential_pool import FAILURE_REASON_BILLING_UNVERIFIED
                _failure_reason = FAILURE_REASON_BILLING_UNVERIFIED
            kwargs["failure_reason"] = _failure_reason
        return pool.mark_exhausted_and_rotate(**kwargs)

    effective_reason = classified_reason
    if effective_reason is None:
        if status_code == 402:
            effective_reason = FailoverReason.billing
        elif status_code == 429:
            effective_reason = FailoverReason.rate_limit
        elif status_code in {401, 403}:
            effective_reason = FailoverReason.auth

    if effective_reason == FailoverReason.upstream_rate_limit:
        # An upstream provider (e.g. DeepSeek behind OpenRouter) is
        # rate-limiting the aggregator's traffic — the user's credential is
        # healthy. Do NOT rotate or mark exhausted; let the caller's fallback
        # path switch to a different model entirely.
        upstream = (error_context or {}).get("upstream_provider") if error_context else None
        if upstream:
            logger.info(
                "Upstream provider %s rate-limited via aggregator — skipping "
                "credential rotation, deferring to fallback chain",
                upstream,
            )
        else:
            logger.info(
                "Upstream aggregator 429 (provider unknown) — skipping "
                "credential rotation, deferring to fallback chain"
            )
        return False, has_retried_429

    if effective_reason == FailoverReason.billing:
        rotate_status = status_code if status_code is not None else 402
        # Runtime credentials can be resolved by a separate pool instance,
        # leaving this recovery pool without ``current_id``. Match the key
        # that actually failed instead of quarantining a different account.
        next_entry = _rotate_failed_credential(rotate_status)
        if next_entry is not None:
            logger.info(
                "Credential %s (billing) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            provider_runtime._swap_credential(agent, next_entry)
            return True, False
        return False, has_retried_429

    if effective_reason == FailoverReason.rate_limit:
        # If current credential is already marked exhausted, skip retry and
        # rotate immediately. This prevents the "cancel-between-429s" trap
        # where has_retried_429 (a local var) gets reset on each new prompt,
        # causing the pool to retry the same exhausted credential forever.
        # Prefer the entry matching the failing key over the shared current()
        # pointer, for the same attribution reason as above.
        current_entry = None
        if _credential_id:
            current_entry = next(
                (e for e in pool.entries() if e.id == _credential_id),
                None,
            )
        if _api_key_hint:
            current_entry = current_entry or next(
                (e for e in pool.entries() if e.runtime_api_key == _api_key_hint),
                None,
            )
        if current_entry is None:
            current_entry = pool.current()
        current_last_status = getattr(current_entry, "last_status", None) if current_entry else None
        if current_last_status == STATUS_EXHAUSTED:
            logger.info(
                "Credential already exhausted (last_status=%s) — rotating immediately instead of retrying",
                current_last_status,
            )
            rotate_status = status_code if status_code is not None else 429
            next_entry = _rotate_failed_credential(rotate_status)
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit, pre-exhausted) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                provider_runtime._swap_credential(agent, next_entry)
                return True, False
            return False, True

        usage_limit_reached = False
        if error_context:
            context_reason = str(error_context.get("reason") or "").lower()
            context_message = str(error_context.get("message") or "").lower()
            usage_limit_reached = (
                "usage_limit_reached" in context_reason
                or "gousagelimit" in context_reason
                or "usage limit reached" in context_message
                or "usage limit has been reached" in context_message
            )
        if not has_retried_429 and not usage_limit_reached:
            return False, True
        rotate_status = status_code if status_code is not None else 429
        next_entry = _rotate_failed_credential(rotate_status)
        if next_entry is not None:
            logger.info(
                "Credential %s (rate limit) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            provider_runtime._swap_credential(agent, next_entry)
            return True, False
        return False, True

    if effective_reason == FailoverReason.auth:
        # Subscription/entitlement 403s look like auth failures on the wire
        # but refresh cannot fix them — the OAuth token is already valid,
        # the account simply lacks the entitlement.  Without this guard,
        # the refresh path keeps minting fresh tokens against the
        # same unsubscribed account and the main agent loop spins re-issuing
        # the same 403 until the user Ctrl+C's.
        #
        # Defense-in-depth for #26847: xAI's backend has been seen to 403
        # standard SuperGrok subscribers with bodies that don't match the
        # existing entitlement keyword set in ``_is_entitlement_failure``.
        # Any 403 against ``xai-oauth`` is treated as entitlement here so
        # the refresh loop can't spin in those cases either.
        #
        # Exception (#29344): xAI's ``[WKE=unauthenticated:...]`` suffix and
        # the ``OAuth2 access token could not be validated`` phrasing are
        # xAI's authoritative "this is a stale token, not entitlement"
        # signal.  When either fires we must NOT apply the catch-all
        # override — refresh is the recoverable path for these bodies, and
        # blanket-classifying them as entitlement was the bug that left
        # long-running TUI sessions stuck on stale tokens until the user
        # exited and reopened.
        is_entitlement = error_reporting._is_entitlement_failure(error_context, status_code)
        _auth_haystack = " ".join(
            str(error_context.get(k) or "").lower()
            for k in ("message", "reason", "code", "error")
            if isinstance(error_context, dict)
        )
        if (
            not is_entitlement
            and status_code == 403
            and "oauth authentication is currently not allowed for this organization" in _auth_haystack
        ):
            is_entitlement = True
        if (
            not is_entitlement
            and status_code == 403
            and (agent.provider or "") == "anthropic"
            and getattr(agent, "api_mode", "") == "anthropic_messages"
        ):
            is_entitlement = True
        if not is_entitlement and status_code == 403 and (agent.provider or "") == "xai-oauth":
            _is_xai_auth_failure = (
                "[wke=unauthenticated:" in _auth_haystack
                or "oauth2 access token could not be validated" in _auth_haystack
            )
            if not _is_xai_auth_failure:
                is_entitlement = True
        if is_entitlement:
            logger.info(
                "Credential %s — entitlement-shaped 403 from %s; "
                "skipping pool refresh (account lacks subscription, "
                "not a transient auth failure).",
                status_code if status_code is not None else "auth",
                agent.provider or "provider",
            )
            return False, has_retried_429
        # Refresh the entry that supplied the failing key, not current():
        # the shared pointer can reference a different, healthy entry, and
        # refreshing it would consume that entry's single-use refresh token
        # (or mark it exhausted on failure) for a failure it never had.
        refresh_kwargs = {"api_key_hint": _api_key_hint}
        if _credential_id:
            refresh_kwargs["credential_id"] = _credential_id
        refreshed = pool.try_refresh_matching(**refresh_kwargs)
        if refreshed is not None:
            # ``try_refresh_matching()`` re-mints a fresh OAuth token and reports
            # success even when the upstream keeps rejecting it — a single-entry
            # pool (common for OAuth/Max subscribers) has nothing to rotate to,
            # so a bare "refreshed → retry" loop spins forever on the same dead
            # token and the configured fallback never activates. Cap consecutive
            # same-entry refreshes and fall through to fallback once exceeded.
            # See #26080.
            refreshed_id = getattr(refreshed, "id", None)
            if refreshed_id is not None:
                refresh_counts = getattr(agent, "_auth_pool_refresh_counts", None)
                if refresh_counts is None:
                    refresh_counts = {}
                    agent._auth_pool_refresh_counts = refresh_counts
                refresh_key = (agent.provider, refreshed_id)
                refresh_counts[refresh_key] = refresh_counts.get(refresh_key, 0) + 1
                if refresh_counts[refresh_key] > _MAX_AUTH_REFRESH_ATTEMPTS:
                    logger.warning(
                        "Credential auth failure persists after %s refreshes for "
                        "pool entry %s — treating as unrecoverable and allowing "
                        "fallback to activate.",
                        refresh_counts[refresh_key] - 1,
                        refreshed_id,
                    )
                    return False, has_retried_429
            logger.info("Credential auth failure — refreshed pool entry %s", getattr(refreshed, 'id', '?'))
            provider_runtime._swap_credential(agent, refreshed)
            return True, has_retried_429
        # Refresh failed — rotate to next credential instead of giving up.
        # The failed entry is already marked exhausted by the refresh attempt.
        rotate_status = status_code if status_code is not None else 401
        next_entry = _rotate_failed_credential(rotate_status)
        if next_entry is not None:
            logger.info(
                "Credential %s (auth refresh failed) — rotated to pool entry %s",
                rotate_status,
                getattr(next_entry, "id", "?"),
            )
            provider_runtime._swap_credential(agent, next_entry)
            return True, False

    return False, has_retried_429

def try_recover_primary_transport(
    agent, api_error: Exception, *, retry_count: int, max_retries: int,
) -> bool:
    """Attempt one extra primary-provider recovery cycle for transient transport failures.

    After ``max_retries`` exhaust, rebuild the primary client (clearing
    stale connection pools) and give it one more attempt before falling
    back.  This is most useful for direct endpoints (custom, Z.AI,
    Anthropic, OpenAI, local models) where a TCP-level hiccup does not
    mean the provider is down.

    Skipped for proxy/aggregator providers (OpenRouter, Nous) which
    already manage connection pools and retries server-side — if our
    retries through them are exhausted, one more rebuilt client won't help.
    """
    import agent.provider_runtime as provider_runtime
    import agent.status_output as status_output
    if agent._fallback_activated:
        return False

    # Only for transient transport errors
    error_type = type(api_error).__name__
    if error_type not in _TRANSIENT_TRANSPORT_ERRORS:
        return False

    # Skip for aggregator providers — they manage their own retry infra
    if provider_runtime._is_openrouter_url(agent):
        return False
    provider_lower = (agent.provider or "").strip().lower()
    # Portal OpenAI-wire traffic still rides aggregator retry infra, so one
    # more rebuilt OpenAI client won't help. Portal Claude on the native
    # Messages route holds a local Anthropic SDK client whose connection
    # pool *does* need the rebuild every other anthropic_messages provider
    # already gets — don't blanket-skip the dual-wire path.
    if (
        provider_lower in {"nous", "nous-portal", "nousresearch"}
        and getattr(agent, "api_mode", None) != "anthropic_messages"
    ):
        return False

    try:
        # Retire the existing client to release stale connections. #70773:
        # never hard-close the shared client here — this runs on the
        # conversation-loop thread while workers from stale-killed streaming
        # attempts may still be unwinding their SSL BIOs on the old pool.
        # ``_retire_shared_openai_client`` shuts the sockets down (FD-safe
        # from any thread) and defers the FD release to GC, which cannot
        # complete until every borrowing thread has unwound.
        if getattr(agent, "client", None) is not None:
            try:
                provider_runtime._retire_shared_openai_client(agent,
                    agent.client, reason="primary_recovery",
                )
            except Exception:
                pass

        # Rebuild from primary snapshot
        rt = agent._primary_runtime
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.requested_provider = rt.get("requested_provider", agent.provider)
        provider_runtime.set_base_url(agent, rt["base_url"])
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]

        if agent.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client
            agent._anthropic_api_key = rt["anthropic_api_key"]
            agent._anthropic_base_url = rt["anthropic_base_url"]
            agent._anthropic_client = build_anthropic_client(
                rt["anthropic_api_key"], rt["anthropic_base_url"],
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
            agent.client = None
        elif (agent.provider or "").strip().lower() == "moa":
            # MoA is a virtual provider with empty client_kwargs — rebuilding
            # via _create_openai_client would raise "api_key client option
            # must be set". Recreate the facade through the shared factory so
            # the reference_callback relay survives recovery (#53802).
            from agent.moa_loop import build_moa_facade

            agent.client = build_moa_facade(agent, agent.model)
        else:
            agent.client = provider_runtime.create_openai_client(agent,
                dict(rt["client_kwargs"]),
                reason="primary_recovery",
                shared=True,
            )

        wait_time = min(3 + retry_count, 8)
        status_output._vprint(agent,
            f"{agent.log_prefix}🔁 Transient {error_type} on {agent.provider} — "
            f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
            force=True,
        )
        time.sleep(wait_time)
        return True
    except Exception as e:
        logger.warning("Primary transport recovery failed: %s", e)
        return False

def restore_primary_runtime(agent) -> bool:
    """Restore the primary runtime at the start of a new turn.

    In long-lived CLI sessions a single create_agent instance spans multiple
    turns.  Without restoration, one transient failure pins the session
    to the fallback provider for every subsequent turn.  Calling this at
    the top of ``run_conversation()`` makes fallback turn-scoped.

    The gateway caches agents across messages (``_agent_cache`` in
    ``gateway/run.py``), so this restoration IS needed there too.
    """
    import agent.provider_runtime as provider_runtime
    if not agent._fallback_activated:
        # Reset the chain index even when no fallback was activated this
        # turn.  Without this, a turn where _try_activate_fallback() was
        # called but returned False (chain exhausted or provider not
        # configured) leaves _fallback_index >= len(_fallback_chain) while
        # _fallback_activated stays False.  The next turn skips this block
        # entirely, stranding the index and silently blocking all future
        # fallback attempts for the session.  Fixes #20465.
        agent._fallback_index = 0
        return False

    if getattr(agent, "_rate_limited_until", 0) > time.monotonic():
        return False  # primary still in rate-limit cooldown, stay on fallback

    # ── Reset-aware gate ──
    # The 60s ``_rate_limited_until`` cooldown covers transient rate limits,
    # but subscription-style providers (Claude Pro/Max 5-hour windows, ChatGPT
    # weekly limits) report reset times hours or days away.  The credential
    # pool already stores those timestamps (``last_error_reset_at``); until
    # the earliest one elapses, every restore attempt is a *guaranteed*
    # failure that costs two prompt-cache invalidations per turn (switch to
    # primary, fail, switch back to fallback) and re-marshals the full
    # context each way.  Skip the restore while the pool says nobody can
    # serve, and come back the moment the reset time passes.
    #
    # Fail-open by design: any error (unreadable auth store, legacy pool
    # adapter without ``next_available_at``) falls through to the existing
    # every-turn retry.  A pool with no reset info returns ``None`` and also
    # falls through — this gate only ever *adds* skips for provably
    # limited windows, so recovery can never be later than it is today.
    #
    # When the attached pool belongs to the fallback provider (cross-provider
    # fallback rebinds it), the primary pool is loaded here and handed to the
    # pool-rebind block below via ``prefetched_primary_pool`` so the load
    # happens at most once per restore.
    prefetched_primary_pool = None
    try:
        primary_provider = str(
            (agent._primary_runtime or {}).get("provider") or ""
        ).strip().lower()
        pool = getattr(agent, "_credential_pool", None)
        if not credential_pool_matches_provider(
            pool,
            primary_provider,
            base_url=str((agent._primary_runtime or {}).get("base_url") or ""),
        ):
            from agent.credential_pool import load_pool

            prefetched_primary_pool = (
                load_pool(primary_provider) if primary_provider else None
            )
            pool = prefetched_primary_pool
        next_at = getattr(pool, "next_available_at", lambda: None)()
        if next_at is not None and next_at > time.time():
            if not getattr(agent, "_restore_wait_logged", False):
                agent._restore_wait_logged = True
                logger.info(
                    "Primary %s rate-limited until %s; staying on fallback "
                    "%s/%s until the reset elapses",
                    primary_provider or "?",
                    datetime.fromtimestamp(next_at).isoformat(timespec="seconds"),
                    agent.provider,
                    agent.model,
                )
            return False
    except Exception:
        logger.debug(
            "Reset-aware restore gate failed; falling back to per-turn retry",
            exc_info=True,
        )
    agent._restore_wait_logged = False

    rt = agent._primary_runtime
    try:
        # ── Core runtime state ──
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.requested_provider = rt.get("requested_provider", agent.provider)
        provider_runtime.set_base_url(agent, rt["base_url"])  # setter updates _base_url_lower
        agent.api_mode = rt["api_mode"]
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent.api_key = rt["api_key"]
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent._use_prompt_caching = rt["use_prompt_caching"]
        # Default to native layout when the restored snapshot predates the
        # native-vs-proxy split (older sessions saved before this PR).
        agent._use_native_cache_layout = rt.get(
            "use_native_cache_layout",
            agent.api_mode == "anthropic_messages" and agent.provider == "anthropic",
        )
        # If the operator has disabled caching via config (cache_ttl is
        # falsy → _cache_disabled flag is set), the disable must survive
        # runtime snapshot restoration (#33555).
        if getattr(agent, "_cache_disabled", False):
            agent._use_prompt_caching = False
            agent._use_native_cache_layout = False

        # ── Rebuild client for the primary provider ──
        if agent.provider == "moa":
            # MoA is a virtual chat-completions provider.  It never has real
            # OpenAI client kwargs; restoring it after a fallback must recreate
            # the facade, not call OpenAI() with an empty api_key.  Use the
            # shared factory so the restored facade keeps the reference_callback
            # relay wired at init — a bare MoAClient() would silently stop
            # emitting moa.reference/moa.aggregating display events (#53802).
            from agent.moa_loop import build_moa_facade

            agent.client = build_moa_facade(agent, agent.model)
            agent._anthropic_client = None
        elif agent.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client
            agent._anthropic_api_key = rt["anthropic_api_key"]
            agent._anthropic_base_url = rt["anthropic_base_url"]
            agent._anthropic_client = build_anthropic_client(
                rt["anthropic_api_key"], rt["anthropic_base_url"],
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
            agent.client = None
        else:
            agent.client = provider_runtime.create_openai_client(agent,
                dict(rt["client_kwargs"]),
                reason="restore_primary",
                shared=True,
            )

        # ── Restore context engine state ──
        cc = agent.context_compressor
        cc.update_model(
            model=rt["compressor_model"],
            context_length=rt["compressor_context_length"],
            base_url=rt["compressor_base_url"],
            api_key=rt["compressor_api_key"],
            provider=rt["compressor_provider"],
            api_mode=rt.get("compressor_api_mode", ""),
        )

        # ── Rebind and re-select the primary credential pool ──
        # A cross-provider fallback attaches the fallback provider's pool. The
        # runtime fields above restore the primary, but leaving that pool in
        # place makes the next primary 401/429 hit the provider-mismatch guard
        # and disables credential rotation. Reload the primary pool first; if
        # auth storage is temporarily unreadable, clear the mismatched pool.
        primary_provider = str(rt.get("provider") or "").strip().lower()
        pool = getattr(agent, "_credential_pool", None)
        pool_provider = str(getattr(pool, "provider", "") or "").strip().lower()
        pool_matches_primary = pool_provider == primary_provider
        if (
            primary_provider == "custom"
            and pool_provider.startswith("custom:")
        ):
            try:
                from agent.credential_pool import get_custom_provider_pool_key

                primary_key = (
                    get_custom_provider_pool_key(str(rt.get("base_url") or "")) or ""
                ).strip().lower()
                pool_matches_primary = bool(primary_key) and primary_key == pool_provider
            except Exception:
                pool_matches_primary = False
        if pool is not None and pool_provider and not pool_matches_primary:
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
            try:
                if prefetched_primary_pool is not None:
                    # Reuse the pool the reset-aware gate already loaded for
                    # this restore — avoids a second disk read of auth.json.
                    agent._credential_pool = prefetched_primary_pool
                else:
                    from agent.credential_pool import load_pool

                    agent._credential_pool = load_pool(primary_provider)
            except Exception as exc:
                logger.warning(
                    "Restore could not reload primary credential pool for %s: %s",
                    primary_provider,
                    exc,
                )

        # The snapshot's api_key was captured at construction time.  Across
        # turns the pool may have rotated (token revocation, billing/rate-limit
        # exhaustion, cooldown), leaving the snapshot key stale.  Restoring it
        # blindly re-fails on the first request and burns through the remaining
        # pool entries before cross-provider fallback even gets a chance.  Ask
        # the pool for its current best entry and swap the live credential in.
        # When the pool is absent, empty, or the entry has no usable key, we
        # keep the snapshot key (the existing behavior).  Fixes #25205.
        agent._credential_pool_entry_id = None
        pool = getattr(agent, "_credential_pool", None)
        if pool is not None and pool.has_available():
            entry = pool.select()
            if entry is not None:
                entry_provider = str(getattr(entry, "provider", "") or "").strip().lower()
                entry_matches_primary = entry_provider == primary_provider
                # Custom endpoints all carry the generic ``custom`` provider on
                # the agent while the pool entry is keyed ``custom:<name>`` (see
                # CUSTOM_POOL_PREFIX). Resolve the primary's base_url to its
                # ``custom:<name>`` key via the canonical helper and compare
                # against the entry's key — this mirrors the sibling guard in
                # ``recover_with_credential_pool`` (see above) and correctly
                # disambiguates multiple custom providers that share one gateway
                # base_url. Fixes #56885.
                from agent.credential_pool import CUSTOM_POOL_PREFIX
                if (
                    primary_provider == "custom"
                    and entry_provider.startswith(CUSTOM_POOL_PREFIX)
                ):
                    entry_matches_primary = False
                    try:
                        from agent.credential_pool import get_custom_provider_pool_key
                        primary_base_url = str(rt.get("base_url") or "").strip()
                        primary_key = (
                            get_custom_provider_pool_key(primary_base_url) or ""
                        ).strip().lower()
                        entry_matches_primary = bool(primary_key) and primary_key == entry_provider
                    except Exception:
                        entry_matches_primary = False

                entry_key = (
                    getattr(entry, "runtime_api_key", None)
                    or getattr(entry, "access_token", "")
                )
                if entry_key and entry_matches_primary:
                    # ``_swap_credential`` rebuilds the OpenAI/Anthropic client,
                    # reapplies base-url-scoped headers, and carries the
                    # accumulated base_url / OAuth-detection fixes (#33163).
                    provider_runtime._swap_credential(agent, entry)
                    logger.info(
                        "Restore re-selected pool entry %s (%s)",
                        getattr(entry, "id", "?"),
                        getattr(entry, "label", "?"),
                    )
                elif entry_key:
                    logger.info(
                        "Restore skipped pool entry %s (%s): provider %s does not match primary provider %s",
                        getattr(entry, "id", "?"),
                        getattr(entry, "label", "?"),
                        entry_provider or "?",
                        primary_provider or "?",
                    )

        # ── Restore reasoning_config if it was saved ──
        # switch_model saves reasoning_config in _primary_runtime. If the
        # snapshot predates that (older sessions), keep the current value.
        saved_reasoning = rt.get("reasoning_config")
        if saved_reasoning is not None:
            agent.reasoning_config = dict(saved_reasoning)

        # ── Reset fallback chain for the new turn ──
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._rate_limit_backoff_count = 0  # reset exponential backoff counter

        # Reset the stale-call circuit breaker (#58962): the streak measured
        # the FALLBACK provider we're leaving; the restored primary deserves
        # a fresh stream attempt before the breaker can trip again.
        from agent.chat_completion_helpers import _reset_stale_streak
        _reset_stale_streak(agent)

        # Undo the fallback's identity rewrite so the prompt is
        # byte-identical to the stored copy again (prefix cache match).
        from agent.chat_completion_helpers import rewrite_prompt_model_identity
        rewrite_prompt_model_identity(agent, rt["model"], rt["provider"])

        logger.info(
            "Primary runtime restored for new turn: %s (%s)",
            agent.model, agent.provider,
        )
        return True
    except Exception as e:
        logger.warning("Failed to restore primary runtime: %s", e)
        return False

def extract_reasoning(agent, assistant_message) -> Optional[str]:
    """
    Extract reasoning/thinking content from an assistant message.

    OpenRouter and various providers can return reasoning in multiple formats:
    1. message.reasoning - Direct reasoning field (DeepSeek, Qwen, etc.)
    2. message.reasoning_content - Alternative field (Moonshot AI, Novita, etc.)
    3. message.reasoning_details - Array of {type, summary, ...} objects (OpenRouter unified)

    Args:
        assistant_message: The assistant message object from the API response

    Returns:
        Combined reasoning text, or None if no reasoning found
    """
    reasoning_parts = []

    # Check direct reasoning field
    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
        reasoning_parts.append(assistant_message.reasoning)

    # Check reasoning_content field (alternative name used by some providers)
    if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
        # Don't duplicate if same as reasoning
        if assistant_message.reasoning_content not in reasoning_parts:
            reasoning_parts.append(assistant_message.reasoning_content)

    # Check reasoning_details array (OpenRouter unified format)
    # Format: [{"type": "reasoning.summary", "summary": "...", ...}, ...]
    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        for detail in assistant_message.reasoning_details:
            if isinstance(detail, dict):
                # Extract summary from reasoning detail object
                summary = (
                    detail.get('summary')
                    or detail.get('thinking')
                    or detail.get('content')
                    or detail.get('text')
                )
                if summary and summary not in reasoning_parts:
                    reasoning_parts.append(summary)

    # Some providers embed reasoning directly inside assistant content
    # instead of returning structured reasoning fields.  Only fall back
    # to inline extraction when no structured reasoning was found.
    content = getattr(assistant_message, "content", None)
    if not reasoning_parts and isinstance(content, list):
        # DeepSeek V4 Pro (and compatible providers) return content as a
        # list of typed blocks, e.g.:
        #   [{"type": "thinking", "thinking": "..."}, {"type": "output", ...}]
        # Without this branch the thinking text is silently dropped and the
        # next turn fails with HTTP 400 ("thinking must be passed back").
        # Refs #21944.
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                thinking_text = block.get("thinking") or block.get("text") or ""
                thinking_text = thinking_text.strip()
                if thinking_text and thinking_text not in reasoning_parts:
                    reasoning_parts.append(thinking_text)
    if not reasoning_parts and isinstance(content, str) and content:
        inline_patterns = (
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
            r"<thought>(.*?)</thought>",
            r"<reasoning>(.*?)</reasoning>",
            r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
        )
        for pattern in inline_patterns:
            flags = re.DOTALL | re.IGNORECASE
            for block in re.findall(pattern, content, flags=flags):
                cleaned = block.strip()
                if cleaned and cleaned not in reasoning_parts:
                    reasoning_parts.append(cleaned)

    # Combine all reasoning parts
    if reasoning_parts:
        return "\n\n".join(reasoning_parts)

    return None

def _direct_native_anthropic_tool_cache_capability(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """Return whether this resolved destination accepts native tool markers."""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    return (
        eff_api_mode == "anthropic_messages"
        and base_url_hostname(eff_base_url) == "api.anthropic.com"
    )

def cache_ttl_means_disabled(ttl: Any) -> bool:
    """Return True when a ``prompt_caching.cache_ttl`` value means caching off.

    Single source of truth for the disable-synonym detection shared by
    ``agent_init`` (live-agent ``_cache_disabled`` flag) and the stub policy
    paths below. Keeping one predicate prevents the two sites from drifting
    (a synonym added in only one place would recreate #76085).

    Unknown values (e.g. ``"2h"``, integers) are NOT a disable — callers keep
    caching enabled with the default TTL, matching ``agent_init``.
    """
    if ttl in ("5m", "1h"):
        return False
    if ttl is False or ttl is None:
        return True
    return str(ttl).lower() in ("off", "false", "disabled", "no", "none")

def _raw_cache_ttl_from_config() -> Any:
    """Read the raw ``prompt_caching.cache_ttl`` config value (may raise)."""
    from hermes_cli.config import load_config_readonly

    pc_cfg = load_config_readonly().get("prompt_caching", {}) or {}
    return pc_cfg.get("cache_ttl", "5m")

def prompt_caching_disabled_from_config() -> bool:
    """Return True when ``prompt_caching.cache_ttl`` is configured as off.

    Same disable detection as ``agent_init`` (via ``cache_ttl_means_disabled``)
    so stub-based policy paths (MoA slot decoration, auxiliary fallback
    replan) honor the same config contract without holding a live
    ``create_agent`` (#76085 / #33555).
    """
    try:
        ttl = _raw_cache_ttl_from_config()
    except Exception:
        return False
    return cache_ttl_means_disabled(ttl)

def configured_cache_ttl() -> Optional[str]:
    """Return the configured ``prompt_caching.cache_ttl`` tier, if valid.

    Mirrors ``agent_init``'s reading of the same key (``5m``/``1h`` accepted,
    anything else ignored) so stub-based paths without a live ``create_agent``
    (auxiliary fallback replan) stop regressing a configured ``1h`` to the
    5m default (#84733). Returns ``None`` for unset/disabled/unknown values;
    ``effective_cache_ttl`` resolves ``None`` to ``5m`` downstream.
    """
    try:
        ttl = _raw_cache_ttl_from_config()
    except Exception:
        return None
    return ttl if ttl in VALID_CACHE_TTLS else None

def blank_cache_policy_stub(cache_disabled: Optional[bool] = None):
    """Build the destination-identity-blank stub for ``anthropic_prompt_cache_policy``.

    Single sanctioned constructor for that stub. Callers that resolve cache
    policy against a destination identified out-of-band (not a live
    ``create_agent``) must go through here so ``_cache_disabled`` is never left
    off a hand-rolled ``SimpleNamespace`` (#76085).

    When ``cache_disabled`` is omitted, falls back to the global config so
    stub paths without an agent snapshot still honor an operator disable.
    """
    from types import SimpleNamespace

    if cache_disabled is None:
        cache_disabled = prompt_caching_disabled_from_config()
    return SimpleNamespace(
        provider="",
        base_url="",
        api_mode="",
        model="",
        _cache_disabled=bool(cache_disabled),
    )

def plan_cache_sections_for_destination(
    messages: list,
    tools: Optional[list],
    *,
    provider: str,
    base_url: str,
    api_mode: str,
    model: str,
    cache_disabled: Optional[bool] = None,
    cache_ttl: Optional[str] = None,
    static_system_prefix: Optional[str] = None,
) -> Tuple[list, list]:
    """Plan request-local cache sections for one resolved destination.

    Shared core of the synchronous acting-aggregator (MoA) and auxiliary
    fallback senders: resolve the cache policy for the destination's real
    provider/base_url/api_mode/model, then either return stripped canonical
    copies (non-caching route) or a :func:`build_prompt_cache_plan` layout
    (caching route, with the direct-native tool marker when the destination
    is api.anthropic.com on the Messages wire).

    Never mutates ``messages`` or ``tools`` — both return values are
    request-local copies.

    ``cache_disabled`` threads the operator's ``prompt_caching.cache_ttl``
    disable into the blank policy stub. When omitted, the live config is
    consulted so MoA/auxiliary paths cannot re-enable markers after the
    user turned caching off (#76085).

    ``cache_ttl`` threads the operator's configured tier (default ``5m``)
    into the destination plan so MoA/auxiliary requests stop regressing to
    the 5m default while the main loop honors ``1h`` (#84733); it is
    clamped per-destination by :func:`effective_cache_ttl` (Qwen → 5m).
    ``static_system_prefix`` threads the builder-declared stable prefix so
    the destination system prompt receives the same early breakpoint the
    main loop applies instead of marking the whole prompt as a breakpoint.
    """
    from agent.prompt_caching import (
        build_prompt_cache_plan,
        effective_cache_ttl,
        strip_anthropic_cache_control,
        strip_anthropic_tool_cache_control,
    )

    stub = blank_cache_policy_stub(cache_disabled)
    should_cache, native_layout = anthropic_prompt_cache_policy(
        stub,
        provider=provider,
        base_url=base_url,
        api_mode=api_mode,
        model=model,
    )
    if not should_cache:
        canonical_messages = copy.deepcopy(messages or [])
        strip_anthropic_cache_control(canonical_messages)
        return canonical_messages, strip_anthropic_tool_cache_control(tools)
    plan = build_prompt_cache_plan(
        messages,
        tools,
        cache_ttl=effective_cache_ttl(
            # effective_cache_ttl resolves None → "5m"; markers are only
            # emitted at all when should_cache passed above, so a
            # cache-disabled agent (_cache_ttl=None) never reaches here
            # with caching active.
            cache_ttl,
            provider=provider,
            model=model,
        ),
        native_anthropic=native_layout,
        static_system_prefix=(
            static_system_prefix if isinstance(static_system_prefix, str) else None
        ),
        direct_native_tool_cache=_direct_native_anthropic_tool_cache_capability(
            stub,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            model=model,
        ),
    )
    return plan.messages, plan.tools

def _is_litellm_route(provider_lower: str, base_url: str) -> bool:
    """True when a route is a LiteLLM proxy, by provider id or host token.

    Provider naming varies per install (``litellm``, ``custom:litellm``, or a
    bare ``custom`` alias pointed at a LiteLLM host), so both signals are
    checked. Both match ``litellm`` as a whole delimited token rather than a
    raw substring: ``base_url_hostname``'s own docstring names substring host
    matching as the false-positive class to avoid, and a plain
    ``"litellm" in ...`` grants Anthropic markers to unrelated routes like
    ``notlitellm.example.com`` or a provider named ``custom:notlitellm``.
    A ``litellm`` *path* segment never qualifies — only the host does.
    """
    if _has_litellm_token(provider_lower, ":-_/"):
        return True
    return _has_litellm_token(base_url_hostname(base_url), ".-")

def _has_litellm_token(value: str, delimiters: str) -> bool:
    """True when ``value`` contains ``litellm`` as a whole delimited token."""
    if not value:
        return False
    for delimiter in delimiters:
        value = value.replace(delimiter, " ")
    return "litellm" in value.split()

def anthropic_prompt_cache_policy(
    agent,
    *,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[bool, bool]:
    """Decide whether to apply Anthropic prompt caching and which layout to use.

    Returns ``(should_cache, use_native_layout)``:
      * ``should_cache`` — inject ``cache_control`` breakpoints for this
        request (applies to OpenRouter Claude, native Anthropic, and
        third-party gateways that speak the native Anthropic protocol).
      * ``use_native_layout`` — place markers on the *inner* content
        blocks (native Anthropic accepts and requires this layout);
        when False markers go on the message envelope (OpenRouter and
        OpenAI-wire proxies expect the looser layout).

    Third-party providers using the native Anthropic transport
    (``api_mode == 'anthropic_messages'`` + Claude-named model) get
    caching with the native layout so they benefit from the same
    cost reduction as direct Anthropic callers, provided their
    gateway implements the Anthropic cache_control contract
    (MiniMax, Zhipu GLM, LiteLLM's Anthropic proxy mode all do).

    Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct
    Alibaba (DashScope) also honour Anthropic-style ``cache_control``
    markers on OpenAI-wire chat completions. Upstream pi-mono #3392 /
    pi #3393 documented this for opencode-go Qwen. Without markers
    these providers serve zero cache hits, re-billing the full prompt
    on every turn.

    If the operator has set ``prompt_caching.cache_ttl`` to a falsy value
    (``false``, ``null``, ``"off"``, etc.) in config.yaml, prompt caching
    is fully disabled — this early return ensures the disable survives
    ``/model`` switches, fallback re-derivation, and runtime snapshot
    restoration (#33555). We check ``"_cache_disabled"`` (set by
    init_agent when the disable is detected) rather than ``_cache_ttl``
    directly, because ``_cache_ttl`` is not yet set when the policy runs
    during the initial ``init_agent`` call.
    """
    if getattr(agent, "_cache_disabled", False):
        return (False, False)

    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""

    # MoA virtual provider: the agent's model/provider are the preset name and
    # "moa" — neither matches any caching branch, so the ACTING AGGREGATOR
    # (often Claude on OpenRouter) silently lost prompt caching entirely
    # (measured: 85% cache share solo vs 2% on the identical model via MoA —
    # tens of millions of re-billed input tokens per benchmark run). Resolve
    # the policy from the preset's real aggregator slot instead.
    if eff_provider.strip().lower() == "moa":
        try:
            from hermes_cli.config import load_config as _load_moa_cfg
            from hermes_cli.moa_config import resolve_moa_preset
            from hermes_cli.runtime_provider import resolve_runtime_provider

            _preset = resolve_moa_preset(
                _load_moa_cfg().get("moa") or {}, eff_model or None
            )
            _agg = _preset.get("aggregator") or {}
            _agg_provider = str(_agg.get("provider") or "").strip()
            _agg_model = str(_agg.get("model") or "").strip()
            if _agg_provider and _agg_model:
                _agg_base_url = ""
                _agg_api_mode = ""
                try:
                    _rt = resolve_runtime_provider(
                        requested=_agg_provider, target_model=_agg_model
                    )
                    _agg_base_url = _rt.get("base_url") or ""
                    _agg_api_mode = _rt.get("api_mode") or ""
                except Exception:
                    pass
                return anthropic_prompt_cache_policy(
                    agent,
                    provider=_agg_provider,
                    base_url=_agg_base_url,
                    api_mode=_agg_api_mode,
                    model=_agg_model,
                )
        except Exception as _moa_exc:  # pragma: no cover - defensive
            logger.debug("MoA aggregator cache-policy resolution failed: %s", _moa_exc)
        return False, False

    if isinstance(eff_model, dict):
        eff_model = eff_model.get('model') or eff_model.get('default') or ''
    eff_model = eff_model if isinstance(eff_model, str) else str(eff_model or '')
    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    # Kimi / Moonshot family via OpenRouter: same cache_control wire format
    # as Claude on OpenRouter (envelope layout).  Without this branch
    # moonshotai/kimi-k2.6 falls through to (False, False), serving ~1%
    # cache hits on 64K-token prompts and re-billing the full prompt on
    # every turn.  Observed within-turn progression with cache enabled:
    # 1% → 67% → 84% → 97% (#25970).  Reuses the canonical family matcher
    # (covers bare k1./k2./k25 release slugs the substring check missed).
    from agent.anthropic_adapter import _model_name_is_kimi_family
    is_kimi = (
        _model_name_is_kimi_family(eff_model) or "moonshot" in model_lower
    )
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter behind the scenes — identical
    # OpenAI-wire envelope cache_control semantics. Treat it as an
    # OpenRouter-equivalent endpoint for caching layout purposes.
    is_nous_portal = base_url_host_matches(eff_base_url, "nousresearch.com")
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = (
        is_anthropic_wire
        and (eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com")
    )

    # A custom Anthropic-compatible route may use a bare model alias that is
    # canonicalized only after Hermes sends the request. In that case model
    # spelling cannot prove cache support. Honor an exact route+model
    # capability declaration instead; explicit false is authoritative too.
    # This preserves the runtime model id (and therefore request/cache keys)
    # while avoiding unsafe alias-name guesses.
    #
    # Also consulted for a LiteLLM route on the OpenAI wire: that grant is
    # inferred from the provider/host name, so an operator who explicitly
    # declares prompt_caching for the route+model must still win over the
    # inference — in either direction. Narrowed to the routes the LiteLLM
    # branch below can actually grant (chat_completions + Claude): the lookup
    # calls get_compatible_custom_providers, which rebuilds its normalized
    # view on every call (~1.5ms uncached), and this function runs per
    # request destination. Widening it unconditionally regressed the
    # non-declaring common case ~200x (7.5us -> 1528us).
    custom_prompt_caching = None
    _litellm_openai_wire = (
        eff_api_mode == "chat_completions"
        and is_claude
        and _is_litellm_route(provider_lower, eff_base_url)
    )
    if is_anthropic_wire or _litellm_openai_wire:
        try:
            from hermes_cli.config import get_custom_provider_model_capability

            custom_prompt_caching = get_custom_provider_model_capability(
                model=eff_model,
                base_url=eff_base_url,
                capability="prompt_caching",
                custom_providers=getattr(agent, "_custom_providers", None),
            )
        except Exception as _cap_exc:
            logger.debug(
                "custom-provider prompt_caching capability lookup failed: %s",
                _cap_exc,
            )
    if custom_prompt_caching is not None:
        # Layout follows the transport, not the declaration: the native
        # inner-block form is only honored on the Anthropic Messages wire
        # (see the LiteLLM OpenAI-wire branch below for why a top-level
        # marker is dropped or 400s on chat_completions).
        return custom_prompt_caching, custom_prompt_caching and is_anthropic_wire

    # MiniMax-M3 rides MiniMax's server-side automatic prefix cache on the
    # Anthropic wire (content-keyed, no marker needed); explicit cache_control
    # is documented for M2.7/M2.5/M2.1/M2 only, so markers on M3 are dead
    # weight — never observable (cache_creation always 0) nor billable.
    # Checked BEFORE the native-Anthropic return: provider="anthropic"
    # pointed at a MiniMax /anthropic proxy is a supported override
    # (_anthropic_base_url_override_ok) that would otherwise return
    # (True, True) above this exclusion.
    # Docs: https://platform.minimax.io/docs/api-reference/text-prompt-caching
    is_minimax_provider = provider_lower in {"minimax", "minimax-cn"}
    is_minimax_host = (
        base_url_host_matches(eff_base_url, "api.minimax.io")
        or base_url_host_matches(eff_base_url, "api.minimaxi.com")
    )
    is_minimax_route = is_minimax_provider or is_minimax_host
    if is_anthropic_wire and is_minimax_route:
        from agent.model_metadata import _model_name_suggests_minimax_m3

        if _model_name_suggests_minimax_m3(eff_model):
            return False, False

    if is_native_anthropic:
        return True, True
    # Envelope layout is an OpenAI-wire construct. Portal Claude on the native
    # Messages route must fall through to the third-party anthropic_messages
    # branch below, which emits inner-block cache_control breakpoints; the
    # envelope form would be dropped and serve 0% cache hits.
    if (
        (is_openrouter or is_nous_portal)
        and (is_claude or is_kimi)
        and not is_anthropic_wire
    ):
        return True, False
    # Nous Portal Qwen (e.g. qwen3.6-plus) takes the same envelope-layout
    # cache_control path as Portal Claude. Portal proxies to OpenRouter
    # and the upstream Qwen route accepts cache_control markers; without
    # this branch the alibaba-family check below only matches
    # provider=opencode/alibaba and Portal traffic falls through to
    # (False, False), serving 0% cache hits and re-billing the full
    # prompt on every turn.
    if is_nous_portal and "qwen" in model_lower:
        return True, False
    if is_anthropic_wire and is_claude:
        # Third-party Anthropic-compatible gateway.
        return True, True

    # LiteLLM fronting a Claude model on the OpenAI-compatible wire.
    # The branch above only matches LiteLLM in Anthropic proxy mode
    # (api_mode == "anthropic_messages"). A LiteLLM deployment that
    # exposes /v1/chat/completions instead matched no grant branch above
    # and fell through to (False, False): no cache_control is injected, the
    # system prompt goes on the wire as a plain string, and the provider
    # serves zero cache hits — the entire prompt is re-billed at full price
    # every turn. Same failure class already documented above for
    # Qwen/DashScope. The endpoint supports Anthropic-style cache_control
    # fine; only the provider detection missed it (#84506).
    #
    # Gated on the Claude family only: a Gemini/GPT/Qwen route through the
    # same proxy must not receive markers — some strict OpenAI-wire relays
    # reject the cache_control block format outright (cf. the DeepSeek /
    # OpenCode exclusion below, #77217).
    #
    # Envelope layout (native_anthropic=False), matching every other
    # OpenAI-wire grant in this function. The native inner-block layout
    # writes a TOP-LEVEL msg["cache_control"] on role:tool and
    # empty-content messages and relies on the Anthropic adapter to
    # relocate it — but that adapter only runs for api_mode ==
    # "anthropic_messages" (agent/transports/anthropic.py), and the
    # chat_completions transport performs no relocation. On this wire the
    # native layout therefore (a) silently loses those breakpoints, spending
    # 2 of the 4 available on markers the provider never sees, and (b) when
    # LiteLLM relocates a top-level marker itself for an OpenRouter-backed
    # Claude route, lands it on an empty text block — the HTTP 400
    # "text content blocks must contain" shape handled in
    # agent/anthropic_adapter.py (#69512).
    #
    # Gated on chat_completions explicitly rather than `not
    # is_anthropic_wire`: codex_responses / bedrock_converse are separate
    # transports with their own marker handling and must not be swept in.
    if _litellm_openai_wire:
        return True, False

    # MiniMax on its Anthropic-compatible endpoint serves its own
    # model family (MiniMax-M2.7, M2.5, M2.1, M2) with documented
    # cache_control support (0.1× read pricing, 5-minute TTL).  The
    # blanket is_claude gate above excludes these — opt them in
    # explicitly via provider id or host match so users on
    # provider=minimax / minimax-cn (or custom endpoints pointing at
    # api.minimax.io/anthropic / api.minimaxi.com/anthropic) get the
    # same cost reduction as Claude traffic.  MiniMax-M3 never reaches
    # here — it is excluded before the native-Anthropic return above.
    # Docs: https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache
    if is_anthropic_wire and is_minimax_route:
        return True, True

    # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire
    # transport that accepts Anthropic-style cache_control markers and
    # rewards them with real cache hits.  Without this branch
    # qwen3.6-plus on opencode-go reports 0% cached tokens and burns
    # through the subscription on every turn.
    #
    # NOTE: DeepSeek models on OpenCode are intentionally excluded.
    # OpenCode Zen's relay rejects the Anthropic-style content block
    # format that cache markers produce (content becomes a block array
    # instead of a plain string), causing HTTP 400 (#77217).
    # Single source of truth for the family set and the qwen-model
    # predicate — shared with the effective_cache_ttl clamp so the
    # opt-in and the TTL clamp can never desync (#84733).
    from agent.prompt_caching import ALIBABA_FAMILY_PROVIDERS, is_qwen_model

    model_is_qwen = is_qwen_model(model_lower)
    provider_is_alibaba_family = provider_lower in ALIBABA_FAMILY_PROVIDERS
    if provider_is_alibaba_family and model_is_qwen:
        # Envelope layout (native_anthropic=False): markers on inner
        # content parts, not top-level tool messages.  Matches
        # pi-mono's "alibaba" cacheControlFormat.
        return True, False

    return False, False

def create_openai_client(agent, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
    import agent.provider_runtime as provider_runtime
    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify
    # Treat client_kwargs as read-only. Callers pass agent._client_kwargs (or shallow
    # copies of it) in; any in-place mutation leaks back into the stored dict and is
    # reused on subsequent requests. #10933 hit this by injecting an httpx.Client
    # transport that was torn down after the first request, so the next request
    # wrapped a closed transport and raised "Cannot send a request, as the client
    # has been closed" on every retry. The revert resolved that specific path; this
    # copy locks the contract so future transport/keepalive work can't reintroduce
    # the same class of bug.
    client_kwargs = dict(client_kwargs)
    # The MoA virtual provider has no real OpenAI wire endpoint - the facade
    # *is* the client. Rebuilding a native OpenAI client while
    # agent.provider == "moa" (client replacement, stream-retry pool cleanup,
    # credential rotation, fallback+restore) drops the facade: the next primary
    # call either raises a `_moa_prepared_request` TypeError (#78382) or, when
    # _client_kwargs carry an unrelated relay base_url, leaks the request to a
    # foreign gateway. Rebuild the facade instead (build_moa_facade also
    # re-wires the reference relay, see #53802).
    if (getattr(agent, "provider", "") or "").strip().lower() == "moa":
        from agent.moa_loop import build_moa_facade
        return build_moa_facade(agent, getattr(agent, "model", None) or "default")
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)
    httpx_verify = resolve_httpx_verify(ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg)
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))
    if agent.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
        from agent.copilot_acp_client import CopilotACPClient

        client = CopilotACPClient(**client_kwargs)
        logger.info(
            "Copilot ACP client created (%s, shared=%s) %s",
            reason,
            shared,
            provider_runtime._client_log_context(agent),
        )
        return client
    if agent.provider == "gemini":
        from agent.gemini_native_adapter import GeminiNativeClient, is_native_gemini_base_url

        base_url = str(client_kwargs.get("base_url", "") or "")
        if is_native_gemini_base_url(base_url):
            safe_kwargs = {
                k: v for k, v in client_kwargs.items()
                if k in {"api_key", "base_url", "default_headers", "timeout", "http_client"}
            }
            if "http_client" not in safe_kwargs:
                keepalive_http = provider_runtime._build_keepalive_http_client(
                    base_url, verify=httpx_verify,
                )
                if keepalive_http is not None:
                    safe_kwargs["http_client"] = keepalive_http
            client = GeminiNativeClient(**safe_kwargs)
            logger.info(
                "Gemini native client created (%s, shared=%s) %s",
                reason,
                shared,
                provider_runtime._client_log_context(agent),
            )
            return client
    # Inject TCP keepalives so the kernel detects dead provider connections
    # instead of letting them sit silently in CLOSE-WAIT (#10324).  Without
    # this, a peer that drops mid-stream leaves the socket in a state where
    # epoll_wait never fires, ``httpx`` read timeout may not trigger, and
    # the agent hangs until manually killed.  Probes after 30s idle, retry
    # every 10s, give up after 3 → dead peer detected within ~60s.
    #
    # Safety against #10933: the ``client_kwargs = dict(client_kwargs)``
    # above means this injection only lands in the local per-call copy,
    # never back into ``agent._client_kwargs``.  Each ``_create_openai_client``
    # invocation therefore gets its OWN fresh ``httpx.Client`` whose
    # lifetime is tied to the OpenAI client it is passed to.  When the
    # OpenAI client is closed (rebuild, teardown, credential rotation),
    # the paired ``httpx.Client`` closes with it, and the next call
    # constructs a fresh one — no stale closed transport can be reused.
    # Tests in ``tests/run_agent/test_create_openai_client_reuse.py`` and
    # ``tests/run_agent/test_sequential_chats_live.py`` pin this invariant.
    if "http_client" not in client_kwargs:
        keepalive_http = provider_runtime._build_keepalive_http_client(
            client_kwargs.get("base_url", ""), verify=httpx_verify,
        )
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http
    # Delegate all rate-limit / 5xx retry to hermes's outer conversation loop,
    # which honors Retry-After and applies adaptive/jittered backoff. The OpenAI
    # SDK default (max_retries=2) uses its own 1-2s backoff that ignores
    # Retry-After and double-retries inside our loop — the same deadlock the
    # Anthropic clients hit (#26293). This is the single chokepoint every primary
    # OpenAI/aggregator client passes through (init, switch_model, recovery,
    # restore, request-scoped); auxiliary_client builds its own clients and keeps
    # SDK retries because it is NOT wrapped by the conversation loop.
    client_kwargs.setdefault("max_retries", 0)
    # Defense-in-depth: guarantee Copilot requests carry the integration
    # headers regardless of which build path we came through. The primary
    # header wiring lives in `_apply_client_headers_for_base_url`, but two
    # rebuild paths (`primary_recovery`, `restore_primary` in this module)
    # reconstruct the client purely from a `_primary_runtime` snapshot and do
    # NOT re-run that wiring. If the snapshot's client_kwargs ever lacks
    # `default_headers` (older snapshot, header-less resolver result), the
    # client goes out WITHOUT `Copilot-Integration-Id: vscode-chat`; the
    # Copilot server then routes it to the "copilot-language-server" integrator
    # whose model allowlist omits enterprise-only models (claude-opus-4.8) →
    # HTTP 400 model_not_available_for_integrator on every turn. This chokepoint
    # is the single place every primary OpenAI client passes through, so filling
    # missing Copilot headers here closes the whole class. We only ADD missing
    # keys — never override headers a caller deliberately set.
    try:
        if base_url_host_matches(str(client_kwargs.get("base_url", "")), "githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers
            existing = dict(client_kwargs.get("default_headers") or {})
            existing_lower = {k.lower() for k in existing}
            for hk, hv in copilot_default_headers().items():
                if hk.lower() not in existing_lower:
                    existing[hk] = hv
            client_kwargs["default_headers"] = existing
    except Exception:
        logger.debug("Copilot default-header guard skipped", exc_info=True)
    # Uses the module-level `OpenAI` name, resolved lazily on first
    # access via __getattr__ below. Tests patch via `run_agent.OpenAI`.
    client = OpenAI(**client_kwargs)
    logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        provider_runtime._client_log_context(agent),
    )
    return client

def switch_model(agent, new_model, new_provider, api_key='', base_url='', api_mode=''):
    """Switch the model/provider in-place for a live agent.

    Called by the /model command handlers (CLI and gateway) after
    ``model_switch.switch_model()`` has resolved credentials and
    validated the model.  This method performs the actual runtime
    swap: rebuilding clients, updating caching flags, and refreshing
    the context compressor.

    The implementation mirrors ``_try_activate_fallback()`` for the
    client-swap logic but also updates ``_primary_runtime`` so the
    change persists across turns (unlike fallback which is
    turn-scoped).
    """
    import agent.provider_runtime as provider_runtime
    from hermes_cli.providers import determine_api_mode

    # ── Determine api_mode if not provided ──
    # Pass model so dual-wire providers (Nous Portal anthropic/* → Messages)
    # resolve correctly; without it determine_api_mode falls back to the
    # openai_chat overlay default.
    if not api_mode:
        api_mode = determine_api_mode(new_provider, base_url, model=new_model)

    # Defense-in-depth: ensure OpenCode base_url doesn't carry a trailing
    # /v1 into the anthropic_messages client, which would cause the SDK to
    # hit /v1/v1/messages.  `model_switch.switch_model()` already strips
    # this, but we guard here so any direct callers (future code paths,
    # tests) can't reintroduce the double-/v1 404 bug.
    if (
        api_mode == "anthropic_messages"
        and new_provider in {"opencode-zen", "opencode-go"}
        and isinstance(base_url, str)
        and base_url
    ):
        base_url = re.sub(r"/v1/?$", "", base_url)

    old_model = agent.model
    old_provider = agent.provider

    # ── Snapshot all fields the swap+rebuild can mutate ──
    # If the rebuild raises (bad API key, network error, build_anthropic_client
    # failure, etc.) we restore these atomically so the agent isn't left with a
    # new model/provider name paired with the OLD client — that mismatch causes
    # HTTP 400s like "claude-sonnet-4-6 is not supported on openai-codex" on the
    # next turn.  Callers in cli.py and gateway/run.py
    # catch the re-raised exception and show the user a warning; without this
    # rollback the warning is misleading because the swap partially succeeded.
    # Use a sentinel so we can distinguish "attribute was unset" from
    # "attribute was None" and skip the restore for genuinely-missing
    # attributes (tests construct bare agents via __new__ without all fields).
    _MISSING = object()
    _snapshot = {
        name: getattr(agent, name, _MISSING)
        for name in (
            "model",
            "provider",
            "requested_provider",
            "base_url",
            "api_mode",
            "api_key",
            "client",
            "_anthropic_client",
            "_anthropic_api_key",
            "_anthropic_base_url",
            "_is_anthropic_oauth",
            "_config_context_length",
        )
    }
    # _client_kwargs is a dict — snapshot a shallow copy so mutating the
    # live dict doesn't poison the rollback target.
    _snapshot["_client_kwargs"] = dict(getattr(agent, "_client_kwargs", {}) or {})
    # Snapshot the credential pool reference so a failed client rebuild can
    # restore the original pool (issue #52727: pool reload is part of this
    # switch and must be reversible on rollback).
    _snapshot["_credential_pool"] = getattr(agent, "_credential_pool", _MISSING)
    _snapshot["_credential_pool_entry_id"] = getattr(
        agent, "_credential_pool_entry_id", _MISSING
    )

    def _restore_snapshot() -> None:
        for _name, _value in _snapshot.items():
            if _value is _MISSING:
                # Attribute did not exist before the swap — don't fabricate it.
                continue
            try:
                setattr(agent, _name, _value)
            except Exception:  # noqa: BLE001
                pass

    try:
        # Clear the per-config context_length override so the new model's
        # actual context window is resolved via get_model_context_length()
        # instead of inheriting the stale value from the previous model.
        agent._config_context_length = None

        # ── Swap core runtime fields ──
        agent.model = new_model
        agent.provider = new_provider
        agent.requested_provider = new_provider
        # Use the new base_url when provided. When it's empty AND the
        # provider is actually changing, do NOT fall back to the current
        # (old provider's) URL — that silently pairs the new provider label
        # with the previous provider's endpoint (e.g. new_provider=minimax
        # paired with the leftover api.githubcopilot.com URL), and every
        # request after the switch 400s at the wrong host. This mismatched
        # pair also gets snapshotted into _primary_runtime below, so it
        # keeps re-applying on every subsequent turn until a full restart.
        # Fail loud instead: the caller (model_switch.switch_model())
        # already resolves base_url for every real provider, so an empty
        # value here means resolution failed upstream, not that the
        # provider genuinely has none. Re-selecting the SAME provider with
        # an empty base_url (e.g. a credential-only refresh) is still fine
        # to keep the current URL. See #47828.
        old_norm_provider = (old_provider or "").strip().lower()
        new_norm_provider = (new_provider or "").strip().lower()
        if base_url:
            provider_runtime.set_base_url(agent, base_url)
        elif old_norm_provider != new_norm_provider:
            raise ValueError(
                f"switch_model: no base_url resolved for provider "
                f"'{new_provider}' (switching from '{old_provider}'); "
                "refusing to keep the previous provider's endpoint"
            )
        agent.api_mode = api_mode
        # Invalidate transport cache — new api_mode may need a different transport
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        if api_key:
            agent.api_key = api_key

        # ── Reload credential pool for the new provider (issue #52727) ──
        # Without this, ``recover_with_credential_pool`` sees a
        # ``pool.provider != agent.provider`` mismatch and short-circuits,
        # leaving the new provider with no rotation/recovery on 401/429 and
        # burning the original pool's entries. Only reload when the provider
        # actually changed (or the pool was missing) — re-selecting the same
        # provider must not churn the pool reference. A reload failure is
        # logged + swallowed: the switch itself must still complete.
        old_norm = (old_provider or "").strip().lower()
        new_norm = (new_provider or "").strip().lower()
        if old_norm != new_norm or getattr(agent, "_credential_pool", None) is None:
            # A pool bound to the old provider is worse than no pool: the
            # recovery guard rejects it and every later 401/429 skips rotation.
            agent._credential_pool = None
            agent._credential_pool_entry_id = None
            try:
                from agent.credential_pool import load_pool
                agent._credential_pool = load_pool(new_provider)
            except Exception as _pool_exc:  # noqa: BLE001
                logger.warning(
                    "switch_model: credential pool reload failed for %s (%s); "
                    "continuing without pool rotation this turn",
                    new_provider, _pool_exc,
                )
        # ── Build new client ──
        if (new_provider or "").strip().lower() == "moa":
            from agent.moa_loop import build_moa_facade

            # The MoA virtual provider speaks only chat.completions via the
            # MoAClient facade — the aggregator's real transport
            # (codex_responses / anthropic_messages) is resolved and applied
            # *inside* the reference/aggregator fan-out, never on the outer
            # primary call. determine_api_mode("moa", ...) above may have left
            # api_mode set to the aggregator's transport; if the conversation
            # loop sees that, it dispatches client.responses.create (which the
            # facade has no .responses for) and the call falls through to the
            # moa://local placeholder → HTTP 404 → fallback to a reference
            # model. Pin chat_completions here so the primary call always goes
            # through MoAClient.chat.completions, matching agent_init.py.
            agent.api_mode = "chat_completions"
            agent.api_key = api_key or "moa-virtual-provider"
            provider_runtime.set_base_url(agent, "moa://local")
            agent._client_kwargs = {}
            agent.client = build_moa_facade(agent, agent.model)
        elif api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # Only fall back to ANTHROPIC_TOKEN when the provider is actually Anthropic.
            # Other anthropic_messages providers (MiniMax, Alibaba, etc.) must use their own
            # API key — falling back would send Anthropic credentials to third-party endpoints.
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or agent.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or agent.api_key or "")

            # MiniMax OAuth: swap static string for a per-request callable token
            # provider so the rebuilt client survives 15-min token expiry. See
            # the matching block in agent_init.py for the full rationale.
            if new_provider == "minimax-oauth" and isinstance(effective_key, str) and effective_key:
                try:
                    from hermes_cli.auth import build_minimax_oauth_token_provider
                    effective_key = build_minimax_oauth_token_provider()
                except Exception as _mm_exc:  # noqa: BLE001
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "MiniMax OAuth: failed to install per-request token provider "
                        "on switch (%s); using static bearer.",
                        _mm_exc,
                    )

            agent.api_key = effective_key
            agent._anthropic_api_key = effective_key
            agent._anthropic_base_url = base_url or getattr(agent, "_anthropic_base_url", None)
            agent._anthropic_client = build_anthropic_client(
                effective_key, agent._anthropic_base_url,
                timeout=get_provider_request_timeout(agent.provider, agent.model),
            )
            agent._is_anthropic_oauth = _is_oauth_token(effective_key) if (_is_native_anthropic and isinstance(effective_key, str)) else False
            agent.client = None
            agent._client_kwargs = {}
        else:
            effective_key = api_key or agent.api_key
            effective_base = base_url or agent.base_url
            agent._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            try:
                from hermes_cli.config import (
                    apply_custom_provider_tls_to_client_kwargs,
                    get_compatible_custom_providers,
                    load_config_readonly,
                )

                # Read custom_providers from live config (not the init-time
                # snapshot on ``agent._custom_providers``) so ssl_ca_cert /
                # ssl_verify edits are honored when switching mid-session,
                # matching the context-length reload below (#15779).
                apply_custom_provider_tls_to_client_kwargs(
                    agent._client_kwargs,
                    str(effective_base or ""),
                    get_compatible_custom_providers(load_config_readonly()),
                )
            except Exception:
                logger.debug("custom-provider TLS resolution skipped on switch_model", exc_info=True)
            _sm_timeout = get_provider_request_timeout(agent.provider, agent.model)
            if _sm_timeout is not None:
                agent._client_kwargs["timeout"] = _sm_timeout
            # Reapply provider-specific headers (e.g. OpenRouter HTTP-Referer,
            # X-Title) that were lost when _client_kwargs was rebuilt from
            # scratch.  Without this, model switches clear attribution headers
            # and OpenRouter logs show "Unknown" for subsequent requests.
            provider_runtime._apply_client_headers_for_base_url(agent, effective_base)
            agent.client = provider_runtime.create_openai_client(agent,
                dict(agent._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        sync_credential_pool_entry_id(agent)
    except Exception:
        # Rollback every mutated field to the pre-swap snapshot so the agent
        # is left consistent (old model + old provider + old client) and the
        # caller's exception handler can surface a meaningful warning.  The
        # exception is re-raised; cli.py and gateway/run.py catch
        # it and print "Agent swap failed; change applied to next session".
        _restore_snapshot()
        raise

    # ── LM Studio: preload before probing context length ──
    _sm_custom_providers = None
    try:
        from hermes_cli.config import (
            get_compatible_custom_providers,
            get_custom_provider_context_length,
            load_config,
        )

        _sm_cfg = load_config()
        _sm_custom_providers = get_compatible_custom_providers(_sm_cfg)
        _destination_context_intent = get_custom_provider_context_length(
            model=agent.model,
            base_url=agent.base_url,
            custom_providers=_sm_custom_providers,
        )
    except Exception:
        _destination_context_intent = None
    agent._config_context_length = _destination_context_intent
    _runtime_context_length = provider_runtime._ensure_lmstudio_runtime_loaded(agent,
        _destination_context_intent
    )
    if provider_runtime._lmstudio_load_was_unverified(_runtime_context_length):
        logger.warning(
            "LM Studio model activation was rejected or completed without a "
            "verifiable active context length during model switch; continuing "
            "with configured context"
        )
    _effective_context_length = provider_runtime._effective_lmstudio_context_length(
        _destination_context_intent,
        _runtime_context_length,
    )

    # ── Re-evaluate prompt caching ──
    # Refresh the custom-provider snapshot from the config just loaded above
    # so the per-model ``prompt_caching`` capability lookup sees the same
    # live list the context-length resolution used — without this, a flag
    # added to config.yaml after session start is invisible to a /model
    # switch (the policy would read the stale init-time snapshot).
    if _sm_custom_providers is not None:
        agent._custom_providers = _sm_custom_providers
    agent._use_prompt_caching, agent._use_native_cache_layout = (
        provider_runtime.anthropic_prompt_cache_policy(agent,
            provider=new_provider,
            base_url=agent.base_url,
            api_mode=api_mode,
            model=new_model,
        )
    )

    # ── Update context compressor ──
    if hasattr(agent, "context_compressor") and agent.context_compressor:
        from agent.model_metadata import get_model_context_length
        if _sm_custom_providers is None:
            try:
                from hermes_cli.config import get_compatible_custom_providers, load_config
                _sm_custom_providers = get_compatible_custom_providers(load_config())
            except Exception:
                _sm_custom_providers = None
        # ``agent.api_key`` may be a callable (Azure Foundry Entra ID
        # token provider). ``get_model_context_length`` expects a
        # string for its live-probe paths; for Foundry the context
        # length normally resolves via config or static catalogs and
        # never hits a probe, but coerce to empty string defensively.
        _ctx_api_key = agent.api_key if isinstance(agent.api_key, str) else ""
        new_context_length = get_model_context_length(
            agent.model,
            base_url=agent.base_url,
            api_key=_ctx_api_key,
            provider=agent.provider,
            config_context_length=_effective_context_length,
            custom_providers=_sm_custom_providers,
        )
        agent.context_compressor.update_model(
            model=agent.model,
            context_length=new_context_length,
            base_url=agent.base_url,
            api_key=agent.api_key,  # context_compressor forwards to call_llm; callable preserved
            provider=agent.provider,
            api_mode=agent.api_mode,
        )

    # ── Re-resolve reasoning_config from per-model override ──
    # The new model may have a different reasoning_effort override. Re-read
    # config so the override takes effect immediately on /model switch —
    # resolved through the shared chokepoint (per-model > global; YAML
    # boolean False = disabled).
    try:
        from hermes_constants import resolve_reasoning_config
        from hermes_cli.config import load_config as _sm_load_config

        _reasoning_cfg = _sm_load_config() or {}
        agent.reasoning_config = resolve_reasoning_config(_reasoning_cfg, agent.model)
        logger.info(
            "switch_model: reasoning_config resolved for %s: %s",
            agent.model, agent.reasoning_config,
        )
    except Exception as _reasoning_err:
        logger.debug("switch_model: could not re-resolve reasoning_config: %s", _reasoning_err)

    # ── Invalidate cached system prompt so it rebuilds next turn ──
    agent._cached_system_prompt = None

    # ── Reset the cross-turn stale-call circuit breaker (#58962) ──
    # The breaker's error text tells the user to "switch models ... then
    # retry"; without this reset the streak stays latched and the freshly
    # selected (healthy) provider would keep short-circuiting before any
    # stream is even attempted.
    from agent.chat_completion_helpers import _reset_stale_streak
    _reset_stale_streak(agent)

    # ── Update _primary_runtime so the change persists across turns ──
    _cc = agent.context_compressor if hasattr(agent, "context_compressor") and agent.context_compressor else None
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "requested_provider": agent.requested_provider,
        "base_url": agent.base_url,
        "api_mode": agent.api_mode,
        "api_key": getattr(agent, "api_key", ""),
        "client_kwargs": dict(agent._client_kwargs),
        "use_prompt_caching": agent._use_prompt_caching,
        "use_native_cache_layout": agent._use_native_cache_layout,
        "reasoning_config": dict(agent.reasoning_config) if getattr(agent, "reasoning_config", None) else None,
        "compressor_model": getattr(_cc, "model", agent.model) if _cc else agent.model,
        "compressor_base_url": getattr(_cc, "base_url", agent.base_url) if _cc else agent.base_url,
        "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
        "compressor_provider": getattr(_cc, "provider", agent.provider) if _cc else agent.provider,
        "compressor_context_length": _cc.context_length if _cc else 0,
        "compressor_api_mode": getattr(_cc, "api_mode", agent.api_mode) if _cc else agent.api_mode,
        "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
    }
    if api_mode == "anthropic_messages":
        agent._primary_runtime.update({
            "anthropic_api_key": agent._anthropic_api_key,
            "anthropic_base_url": agent._anthropic_base_url,
            "is_anthropic_oauth": agent._is_anthropic_oauth,
        })

    # ── Reset fallback state ──
    agent._fallback_activated = False
    agent._fallback_index = 0

    # When the user deliberately swaps primary providers (e.g. openrouter
    # → anthropic), drop any fallback entries that target the OLD primary
    # or the NEW one.  The chain was seeded from config at agent init for
    # the original provider — without pruning, a failed turn on the new
    # primary silently re-activates the provider the user just rejected,
    # which is exactly what was reported during TUI v2 blitz testing
    # ("switched to anthropic, tui keeps trying openrouter").
    old_norm = (old_provider or "").strip().lower()
    new_norm = (new_provider or "").strip().lower()
    fallback_chain = list(getattr(agent, "_fallback_chain", []) or [])
    if old_norm and new_norm and old_norm != new_norm:
        fallback_chain = [
            entry for entry in fallback_chain
            if (entry.get("provider") or "").strip().lower() not in {old_norm, new_norm}
        ]
    agent._fallback_chain = fallback_chain

    logger.info(
        "Model switched in-place: %s (%s) -> %s (%s)",
        old_model, old_provider, new_model, new_provider,
    )

    # ── Persist billing route to session DB ──
    # The agent's _session_db / session_id may not be set in all contexts
    # (tests, bare agents without a session DB, etc.).  This ensures the
    # dashboard Model cards show the actual provider after a mid-session
    # /model switch instead of the stale session-creation provider.
    # See #48248 for the full bug description.
    _session_db = getattr(agent, "_session_db", None)
    _session_id = getattr(agent, "session_id", None)
    if _session_db is not None and _session_id:
        try:
            _session_db.update_session_billing_route(
                _session_id,
                provider=agent.provider,
                base_url=agent.base_url,
                billing_mode=getattr(agent, "api_mode", None),
            )
        except Exception:
            logger.warning(
                "Failed to persist billing route after model switch",
                exc_info=True,
            )

def looks_like_codex_intermediate_ack(
    agent,
    user_message: Any,
    assistant_content: str,
    messages: List[Dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.

    ``require_workspace`` (default True) keeps the original codex-coding scope:
    the ack must reference a filesystem/repo workspace. The conversation loop
    passes ``require_workspace=False`` when the user has explicitly opted into
    intent-ack continuation for all api_modes (``agent.intent_ack_continuation``
    is ``true`` or a model-list), so general autonomous workflows ("I'll run a
    health check on the server", "I'll start the deployment") — which carry a
    future-ack and an action verb but no filesystem reference — are caught too.
    The future-ack + short-content + no-prior-tools + action-verb requirements
    always apply, which is what keeps conversational "I'll help you brainstorm"
    replies from tripping it.
    """
    import agent.provider_runtime as provider_runtime
    if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
        return False

    assistant_text = message_protocol.strip_think_blocks(agent, assistant_content or "").strip().lower()
    if not assistant_text:
        return False
    if len(assistant_text) > 1200:
        return False

    has_future_ack = bool(
        re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
    )
    if not has_future_ack:
        return False

    action_markers = (
        "look into",
        "look at",
        "inspect",
        "scan",
        "check",
        "analyz",
        "review",
        "explore",
        "read",
        "open",
        "run",
        "test",
        "fix",
        "debug",
        "search",
        "find",
        "walkthrough",
        "report back",
        "summarize",
    )
    workspace_markers = (
        "directory",
        "current directory",
        "current dir",
        "cwd",
        "repo",
        "repository",
        "codebase",
        "project",
        "folder",
        "filesystem",
        "file tree",
        "files",
        "path",
    )

    assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
    if not assistant_mentions_action:
        return False

    # Opted-in (all-api_mode) path: a future-ack + action verb + no prior tool
    # call is enough — the user asked us to keep going when the model only
    # announces intent, regardless of whether a filesystem is involved.
    if not require_workspace:
        return True

    # ``user_message`` is typed ``str`` but can arrive as an OpenAI-style
    # multi-part content list (``[{type:"text",...}, {type:"image_url",...}]``)
    # for vision requests routed through the OpenAI-compat API server. A
    # truthy list survives ``(user_message or "")`` and then ``.strip()``
    # raises ``AttributeError`` — flatten to text first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log

    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    user_targets_workspace = (
        any(marker in user_text for marker in workspace_markers)
        or "~/" in user_text
        or "/" in user_text
    )
    assistant_targets_workspace = any(
        marker in assistant_text for marker in workspace_markers
    )
    return user_targets_workspace or assistant_targets_workspace

def intent_ack_continuation_mode(agent) -> str:
    """Classify the resolved intent-ack continuation mode for this turn.

    Returns one of:
      * ``"off"``        — never continue.
      * ``"codex_only"`` — historical scope: continue only on the
        ``codex_responses`` api_mode, and only for codebase/workspace acks
        (``require_workspace=True``).
      * ``"all"``        — user opted in for every api_mode; continue on any
        future-ack + action verb (``require_workspace=False``).

    Mirrors the four-mode shape of ``agent.tool_use_enforcement``: ``"auto"``
    (default) → codex_only; ``True``/"true"/"always"/"yes"/"on" → all;
    ``False``/"false"/"never"/"no"/"off" → off; ``list`` → all when a substring
    matches the active model name, else off.
    """
    mode = getattr(agent, "_intent_ack_continuation", "auto")

    if mode is True or (isinstance(mode, str) and mode.lower() in {"true", "always", "yes", "on"}):
        return "all"
    if mode is False or (isinstance(mode, str) and mode.lower() in {"false", "never", "no", "off"}):
        return "off"
    if isinstance(mode, list):
        model_lower = (agent.model or "").lower()
        return "all" if any(p.lower() in model_lower for p in mode if isinstance(p, str)) else "off"
    # "auto" or any unrecognised value — historical codex-only behavior.
    return "codex_only" if agent.api_mode == "codex_responses" else "off"

def intent_ack_continuation_enabled(agent) -> bool:
    """Whether intent-ack continuation should fire at all for this turn.

    The ``codex_ack_continuations < 2`` per-turn cap and the
    ``looks_like_codex_intermediate_ack`` detector are applied by the caller;
    this only decides the on/off gate. Callers that also need to know whether
    the workspace requirement applies should use ``intent_ack_continuation_mode``
    directly (``"codex_only"`` ⇒ require_workspace=True, ``"all"`` ⇒ False).
    """
    return intent_ack_continuation_mode(agent) != "off"

def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:
    """Copy provider-facing reasoning fields onto an API replay message.

    Forwarder — the strip-vs-repad POLICY is owned by
    ``agent.message_sanitization.apply_reasoning_content_policy`` (audit F4);
    this only supplies the agent's cached provider-direction flag.
    """
    import agent.provider_runtime as provider_runtime
    from agent.message_sanitization import apply_reasoning_content_policy

    apply_reasoning_content_policy(
        source_msg, api_msg, provider_runtime._needs_thinking_reasoning_pad(agent)
    )

def reapply_reasoning_echo_for_provider(agent, api_messages: list) -> int:
    """Re-pad (or strip) assistant turns' reasoning_content for the active provider.

    ``api_messages`` is built once, before the retry loop, while the *primary*
    provider is active.  A mid-conversation fallback can then switch providers,
    so the reasoning fields baked into ``api_messages`` are shaped for the
    *prior* provider and must be reconciled against the *current* one:

    * Switching TO a require-side provider (DeepSeek / Kimi / MiMo thinking
      mode): assistant turns built when the prior provider did NOT need the
      echo-back go out without ``reasoning_content`` and the new provider
      rejects them with HTTP 400 ("The reasoning_content in the thinking mode
      must be passed back").  Re-apply the pad.

    * Switching TO a strict provider that rejects the field (Mistral,
      Cerebras, Groq, SambaNova, …): assistant turns built under a reasoning
      primary carry a ``reasoning_content`` pad (often a single space ``" "``),
      and the strict provider rejects it with HTTP 400/422 ("Extra inputs are
      not permitted").  Strip the field.  This is the exact cross-provider
      fallback bug from #45655 — a DeepSeek primary pads history with ``" "``,
      the request falls back to Mistral, and Mistral 422s on the stale pad.

    Calling this immediately before building the request kwargs reconciles the
    fields against the *current* provider.  It is idempotent and safe to call
    every iteration; it covers every fallback path.

    Returns the number of assistant turns whose reasoning_content was added or
    removed.
    """
    import agent.provider_runtime as provider_runtime
    from agent.message_sanitization import reapply_reasoning_echo

    return reapply_reasoning_echo(
        api_messages, provider_runtime._needs_thinking_reasoning_pad(agent)
    )

def _iter_httpx_pool_objects(http_client: Any):
    """Yield httpcore pool objects reachable from an httpx client.

    Hermes' keepalive client (#10324 / ``_build_keepalive_http_client``) and
    any ``HTTP(S)_PROXY`` configuration put live connections on *mounted*
    transports (``client._mounts``), not only on the default
    ``client._transport``. Walking the default transport alone makes
    ``force_close_tcp_sockets`` return 0 while a stream is still mid-recv —
    the interrupt logs success and the provider keeps burning the slot
    (#72975).
    """
    seen_pools: set[int] = set()

    def _emit(pool: Any):
        if pool is None:
            return
        marker = id(pool)
        if marker in seen_pools:
            return
        seen_pools.add(marker)
        yield pool

    def _pools_for_transport(transport: Any):
        if transport is None:
            return
        # Normal httpx.HTTPTransport / HTTPProxy-as-transport: connections
        # live under ``_pool``. HTTPProxy itself *is* a ConnectionPool and
        # may be mounted directly — then ``_connections`` is on the
        # transport.
        pool = getattr(transport, "_pool", None)
        if pool is not None:
            yield from _emit(pool)
            return
        if getattr(transport, "_connections", None) is not None:
            yield from _emit(transport)

    try:
        yield from _pools_for_transport(getattr(http_client, "_transport", None))
        mounts = getattr(http_client, "_mounts", None) or {}
        for _pattern, mounted in list(mounts.items()):
            yield from _pools_for_transport(mounted)
    except Exception:
        return

def _connection_candidates(conn: Any):
    """Walk nested ``_connection`` wrappers (proxy tunnel → HTTP11/2)."""
    seen: set[int] = set()
    stack = [conn]
    while stack:
        candidate = stack.pop()
        if candidate is None:
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        yield candidate
        inner = getattr(candidate, "_connection", None)
        if inner is not None and id(inner) not in seen:
            stack.append(inner)

def _iter_pool_sockets(client: Any):
    """Yield raw sockets reachable from an OpenAI/httpx client pool.

    httpcore 1.x stores the concrete HTTP11/HTTP2 connection under
    ``conn._connection``; older versions exposed stream attributes directly
    on the pool entry. Proxy tunnels wrap another layer
    (``TunnelHTTPConnection`` / ``ForwardHTTPConnection``). Keep the
    traversal defensive because these are private transport internals and
    vary across httpx/httpcore releases.

    Also walks ``httpx`` mount transports — see ``_iter_httpx_pool_objects``
    — and in-flight httpcore ``PoolRequest.connection`` objects, which stay
    reachable even when ``_connections`` is empty during checkout (#85252).
    """
    try:
        http_client = getattr(client, "_client", None)
        if http_client is None:
            # Some SDK wrappers *are* the httpx client (or expose the pool
            # directly). Fall through so mount-aware discovery still runs.
            http_client = client
        pools = list(_iter_httpx_pool_objects(http_client))
    except Exception:
        return

    if not pools:
        return

    seen: set[int] = set()
    for pool in pools:
        # Empty-list is falsy: use ``is None`` so an empty ``_connections``
        # still lets us walk in-flight ``_requests`` rather than skipping
        # the pool entirely.
        raw_conns = getattr(pool, "_connections", None)
        if raw_conns is None:
            raw_conns = getattr(pool, "_pool", None)
        connections = list(raw_conns or [])
        for pool_req in list(getattr(pool, "_requests", None) or []):
            conn = getattr(pool_req, "connection", None)
            if conn is not None:
                connections.append(conn)
        for conn in connections:
            for candidate in _connection_candidates(conn):
                stream = (
                    getattr(candidate, "_network_stream", None)
                    or getattr(candidate, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    get_extra_info = getattr(stream, "get_extra_info", None)
                    if callable(get_extra_info):
                        try:
                            sock = get_extra_info("socket")
                        except Exception:
                            sock = None
                if sock is None:
                    wrapped = getattr(stream, "stream", None)
                    if wrapped is not None:
                        sock = getattr(wrapped, "_sock", None)
                if sock is None:
                    # anyio-backed streams expose the raw socket through
                    # SocketAttribute.raw_socket when available.
                    wrapped = getattr(stream, "_stream", None)
                    extra = getattr(wrapped, "extra", None)
                    if callable(extra):
                        try:
                            from anyio.abc import SocketAttribute
                            sock = extra(SocketAttribute.raw_socket)
                        except Exception:
                            sock = None
                if sock is None:
                    continue
                marker = id(sock)
                if marker in seen:
                    continue
                seen.add(marker)
                yield sock

def cleanup_dead_connections(agent) -> bool:
    """Detect and clean up dead TCP connections on the primary client.

    Inspects the httpx connection pool for sockets in unhealthy states
    (CLOSE-WAIT, errors).  If any are found, force-closes all sockets
    and rebuilds the primary client from scratch.

    Returns True if dead connections were found and cleaned up.
    """
    import agent.provider_runtime as provider_runtime
    client = getattr(agent, "client", None)
    if client is None:
        return False
    try:
        dead_count = 0
        for sock in _iter_pool_sockets(client):
            # Probe socket health with a non-blocking recv peek
            import socket as _socket
            try:
                sock.setblocking(False)
                data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                if data == b"":
                    dead_count += 1
            except BlockingIOError:
                pass  # No data available — socket is healthy
            except OSError:
                dead_count += 1
            finally:
                try:
                    sock.setblocking(True)
                except OSError:
                    pass
        if dead_count > 0:
            logger.warning(
                "Found %d dead connection(s) in client pool — rebuilding client",
                dead_count,
            )
            provider_runtime._replace_primary_openai_client(agent, reason="dead_connection_cleanup")
            return True
    except Exception as exc:
        logger.debug("Dead connection check error: %s", exc)
    return False

def force_close_tcp_sockets(client: Any) -> int:
    """Abort in-flight TCP I/O by shutting down sockets WITHOUT closing FDs.

    When a provider drops a connection mid-stream — or the user issues an
    interrupt — we want to unblock httpx's reader/writer immediately rather
    than waiting for the kernel's per-connection timeout. ``shutdown(SHUT_RDWR)``
    achieves that: it sends FIN, breaks any pending ``recv``/``send`` with EOF
    or ``EPIPE``, but does NOT release the file descriptor.

    Historically this helper also called ``socket.close()`` so the FD got
    released immediately, but that's unsafe when (as is the case for both the
    interrupt-abort path and stale-call kill path) the helper runs on a
    different thread than the one driving the request:

      * The Python ``socket.socket`` we close here is the SAME object held by
        httpx's pool, so closing it via Python sets its ``_fd`` to -1 and
        future operations on that Python object fail safely.
      * BUT the SSL wrapper (``ssl.SSLSocket``'s underlying OpenSSL ``BIO``)
        caches the raw integer FD. Once ``os.close(fd)`` runs, the kernel may
        immediately recycle that integer to the next ``open()`` call — e.g.
        the kanban dispatcher opening ``kanban.db``.
      * The owning worker thread then unwinds httpx, the SSL layer flushes a
        pending TLS record, and the encrypted bytes get written into the
        wrong file (issue #29507: 24-byte TLS application-data record
        clobbering SQLite header bytes 5..28).

    The fix is to let the owning thread own the close. ``shutdown()`` from any
    thread is FD-safe; ``close()`` is not. The httpx connection's own close
    path — which runs from the worker thread when it unwinds — will release
    the FD via the same ``socket.socket`` object, and because Python's socket
    close atomically swaps ``_fd`` to -1 *before* issuing ``os.close``, there
    is no FD-aliasing window when only one thread closes.

    Returns the number of sockets shut down. (Field kept as
    ``tcp_force_closed=N`` in the log line for backwards-compatible parsing.)
    """
    import socket as _socket

    shutdown_count = 0
    try:
        for sock in _iter_pool_sockets(client):
            try:
                # Clear a blocking timeout first so a hung SSL_read on the
                # owner thread notices the shutdown. Some stacks ignore
                # SHUT_RDWR alone while recv is blocked with timeout=None
                # (#85252). Still no close() — that is the #29507 race.
                settimeout = getattr(sock, "settimeout", None)
                if callable(settimeout):
                    try:
                        settimeout(0)
                    except OSError:
                        pass
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                # Already shut down / not connected / FD invalid — all benign.
                pass
            # IMPORTANT (#29507): do NOT call sock.close() here. See docstring.
            shutdown_count += 1
    except Exception as exc:
        logger.debug("Force-close TCP sockets sweep error: %s", exc)
    return shutdown_count
