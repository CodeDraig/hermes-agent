"""Responsibility-owned agent error reporting behavior."""
import copy
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from agent.init_session import safe_session_filename_component
from agent.prompt_builder import format_steer_marker
from agent.redact import redact_sensitive_text
from agent.tool_result_classification import (
    FILE_MUTATING_TOOL_NAMES as _FILE_MUTATING_TOOLS,
    file_mutation_result_landed,
)
from agent.trajectory import (
    convert_scratchpad_to_think,
)
from agent.tool_dispatch_helpers import (
    _extract_file_mutation_targets,
    _extract_landed_file_mutation_paths,
    _extract_error_preview,
)
from agent.usage_pricing import normalize_usage
from utils import atomic_json_write, env_var_enabled


logger = logging.getLogger(__name__)

_FOOTER_PATH_RE = re.compile(
    r"(?<![/:\w.`])(?:~/|/|[A-Za-z]:[/\\])(?:[\w.\-]+[/\\])*[\w.\-]+\.[\w]+",
)

def _is_entitlement_failure(
    error_context: Optional[Dict[str, Any]],
    status_code: Optional[int],
) -> bool:
    """Detect subscription/entitlement 403s that masquerade as auth failures.

    Returned True only when the body text matches a known entitlement
    shape AND the status is 401/403.  Refreshing an OAuth token cannot
    fix an unsubscribed account, so callers should surface the error
    instead of looping the credential pool.

    Current matches:
      * xAI OAuth: "do not have an active Grok subscription" /
        "out of available resources" / "does not have permission" + "grok"

    Disambiguator for xAI (#29344): the same ``code`` text ("The caller
    does not have permission to execute the specified operation") is
    returned for BOTH an unsubscribed account AND a stale OAuth access
    token.  xAI ships an explicit signal in the ``error`` field that
    tells the two apart: a ``[WKE=unauthenticated:...]`` suffix (and/or
    the ``OAuth2 access token could not be validated`` phrasing) means
    the credentials failed validation — that's recoverable by refreshing
    the token, NOT by surfacing an entitlement message.  When either
    signal is present we return False eagerly so the credential-pool
    refresh path runs, letting long-running TUI sessions recover from
    stale tokens without an exit/reopen cycle.

    Extend here for new providers as we discover them (Anthropic's
    Claude Max OAuth entitlement errors look distinct enough today that
    the existing 1M-context-beta branch handles them; revisit if other
    subscription tiers start producing the same loop signature).
    """
    if status_code not in {401, 403, None}:
        return False
    if not isinstance(error_context, dict):
        return False
    # Build a single lowercase haystack covering every field shape the
    # body might land in.  ``_extract_api_error_context`` normalises to
    # ``message``/``reason``, but callers (and the test suite) may also
    # hand us the raw body with ``code``/``error`` keys; cover both so
    # the WKE disambiguator below fires regardless of entry point.
    message = str(error_context.get("message") or "").lower()
    reason = str(error_context.get("reason") or "").lower()
    code = str(error_context.get("code") or "").lower()
    err = str(error_context.get("error") or "").lower()
    haystack = f"{message} {reason} {code} {err}"
    if not haystack.strip():
        return False
    # xAI's authoritative disambiguator for "stale token" vs
    # "unsubscribed account".  Both conditions share the same
    # permission-denied ``code`` text; only one carries this suffix.
    # Bail out before the entitlement keyword checks so a stale OAuth
    # token routes through the credential-refresh path instead of the
    # surface-error-as-entitlement path.  See #29344 for the long-
    # running TUI failure mode this closes.
    if "[wke=unauthenticated:" in haystack:
        return False
    if "oauth2 access token could not be validated" in haystack:
        return False
    if "do not have an active grok subscription" in haystack:
        return True
    if "out of available resources" in haystack and "grok" in haystack:
        return True
    if "does not have permission" in haystack and "grok" in haystack:
        return True
    return False

def _decorate_xai_entitlement_error(detail: str) -> str:
    """Append a neutral hint when xAI's OAuth surface returns the
    permission-denied 403.

    xAI's ``/v1/responses`` endpoint replies to several distinct failure
    modes with the SAME body::

        {"code": "The caller does not have permission to execute the
         specified operation", "error": "You have either run out of
         available resources or do not have an active Grok subscription.
         Manage subscriptions at https://grok.com/?_s=usage or subscribe
         at https://grok.com/supergrok"}

    That body covers several real causes we cannot distinguish without
    more info from xAI.  The most common (and least obvious) one is
    that **X Premium+ does NOT include API access** — only standalone
    SuperGrok subscribers can use Hermes against xai-oauth.  Lots of
    users see Grok in their X app, assume it works here too, and hit
    this 403 with no idea why.  Lead the hint with that.

    Other possible causes:
      * No Grok subscription at all
      * SuperGrok tier doesn't include the requested model (e.g.
        grok-4.3 may need a higher tier)
      * Monthly quota exhausted (the ``?_s=usage`` URL hints at this)

    Surface the raw xAI text verbatim and point at
    https://grok.com/?_s=usage where the user can see WHICH applies.

    Matched once per detail string — won't double-decorate if the
    upstream already concatenated the same text.
    """
    if not detail:
        return detail
    lower = detail.lower()
    is_entitlement = (
        "do not have an active grok subscription" in lower
        or ("out of available resources" in lower and "grok" in lower)
        or ("does not have permission" in lower and "grok" in lower)
    )
    if not is_entitlement:
        return detail
    hint = (
        " — xAI rejected this OAuth account. NOTE: X Premium+ does NOT "
        "include xAI API access — only standalone SuperGrok subscribers "
        "can use this provider. Other possible causes: no Grok "
        "subscription, your tier doesn't include this model, or your "
        "quota is exhausted. Check https://grok.com/?_s=usage to see "
        "which, or run `/model` to switch providers."
    )
    # Idempotency: detect prior decoration by a substring unique to the
    # hint (not present in xAI's own body text).
    if "X Premium+ does NOT include" in detail:
        return detail
    return f"{detail}{hint}"

def _coerce_api_error_detail(value: Any) -> str:
    """Return a display-safe string for structured provider error fields."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("message", "detail", "error", "code", "type"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested
        for key in ("message", "detail", "error", "code", "type"):
            if key in value:
                nested_detail = _coerce_api_error_detail(value[key])
                if nested_detail:
                    return nested_detail
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
    if isinstance(value, (list, tuple)):
        parts = [
            _coerce_api_error_detail(item)
            for item in value
        ]
        return "; ".join(part for part in parts if part)
    if value is None:
        return ""
    return str(value)

def _summarize_api_error(error: Exception) -> str:
    """Extract a human-readable one-liner from an API error.

    Handles Cloudflare HTML error pages (502, 503, etc.) by pulling the
    <title> tag instead of dumping raw HTML. Network/DNS failures are
    translated into an offline hint, including when an SDK wraps the
    original OS error. Falls back to a truncated str(error) otherwise.
    """
    raw = str(error)

    # Linux, macOS, and Windows use different low-level messages when DNS
    # cannot resolve the provider while the device is offline. SDKs often
    # wrap that OSError in a generic "Connection error", so inspect the
    # exception chain before showing the top-level message to the user.
    network_resolution_markers = (
        "temporary failure in name resolution",
        "name or service not known",
        "nodename nor servname provided, or not known",
        "getaddrinfo failed",
        "no address associated with hostname",
        "network is unreachable",
    )
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(
            marker in str(current).lower()
            for marker in network_resolution_markers
        ):
            return (
                "Hermes can't reach the model provider. You may be offline. "
                "Check your internet connection and try again."
            )
        current = current.__cause__ or current.__context__

    if (
        isinstance(error, ValueError)
        and "expected ident at line" in raw.lower()
    ):
        return f"Malformed provider streaming response: {raw[:300]}"

    # Cloudflare / proxy HTML pages: grab the <title> for a clean summary
    if "<!DOCTYPE" in raw or "<html" in raw:
        m = re.search(r"<title[^>]*>([^<]+)</title>", raw, re.IGNORECASE)
        title = m.group(1).strip() if m else "HTML error page (title not found)"
        # Also grab Cloudflare Ray ID if present
        ray = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
        ray_id = ray.group(1).strip() if ray else None
        status_code = getattr(error, "status_code", None)
        parts = []
        if status_code:
            parts.append(f"HTTP {status_code}")
        parts.append(title)
        if ray_id:
            parts.append(f"Ray {ray_id}")
        return " — ".join(parts)

    # GeminiAPIError (agent/gemini_native_adapter.py) already composes a
    # clean one-liner and may have appended actionable guidance (free-tier
    # 429, legacy Standard-key 401). Prefer its message over re-extracting
    # the raw response body below, which would strip that guidance.
    if type(error).__name__ == "GeminiAPIError":
        return redact_sensitive_text(raw[:1000])

    # JSON body errors from OpenAI/Anthropic SDKs
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
        if msg:
            status_code = getattr(error, "status_code", None)
            prefix = f"HTTP {status_code}: " if status_code else ""
            msg = _coerce_api_error_detail(msg)

            return _decorate_xai_entitlement_error(f"{prefix}{msg[:300]}")

    # SDK may leave body empty while httpx still has the payload (#36109).
    # Redact before returning: the raw provider/proxy error body is
    # attacker-influenced and may echo Authorization / x-api-key / request
    # JSON, which would otherwise leak into final_response + logs (this path
    # widens exposure vs the old empty-body "HTTP 400" string).
    response = getattr(error, "response", None)
    if response is not None:
        try:
            snippet = (getattr(response, "text", None) or "").strip()
        except Exception:
            snippet = ""
        if snippet:
            status_code = getattr(error, "status_code", None)
            prefix = f"HTTP {status_code}: " if status_code else ""
            try:
                payload = json.loads(snippet)
            except (json.JSONDecodeError, TypeError):
                payload = None
            if isinstance(payload, dict):
                err = payload.get("error")
                if isinstance(err, dict) and err.get("message"):
                    return redact_sensitive_text(f"{prefix}{str(err['message'])[:300]}")
                if payload.get("message"):
                    return redact_sensitive_text(f"{prefix}{str(payload['message'])[:300]}")
            return redact_sensitive_text(f"{prefix}{snippet[:300]}")

    # Fallback: truncate the raw string but give more room than 200 chars
    status_code = getattr(error, "status_code", None)
    prefix = f"HTTP {status_code}: " if status_code else ""
    return _decorate_xai_entitlement_error(f"{prefix}{raw[:500]}")

def _mask_api_key_for_logs(self, key: Any) -> Optional[str]:
    # Azure Foundry Entra ID bearer providers are callables — never
    # invoke them in log paths; identify the auth surface instead.
    if callable(key) and not isinstance(key, str):
        return "<entra-id-bearer>"
    if not key:
        return None
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"

def _clean_error_message(self, error_msg: str) -> str:
    """
    Clean up error messages for user display, removing HTML content and truncating.

    Args:
        error_msg: Raw error message from API or exception

    Returns:
        Clean, user-friendly error message
    """
    if not error_msg:
        return "Unknown error"

    # Remove HTML content (common with CloudFlare and gateway error pages)
    if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
        return "Service temporarily unavailable (HTML error page returned)"

    # Remove newlines and excessive whitespace
    cleaned = ' '.join(error_msg.split())

    # Truncate if too long
    if len(cleaned) > 150:
        cleaned = cleaned[:150] + "..."

    return cleaned


def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
    """Token buckets for ``post_api_request`` plugins (no raw ``response`` object)."""
    if response is None:
        return None
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return None
    from dataclasses import asdict


    cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
    summary = asdict(cu)
    summary.pop("raw_usage", None)
    summary["prompt_tokens"] = cu.prompt_tokens
    summary["total_tokens"] = cu.total_tokens
    return summary

def _hook_payload_max_chars() -> int:
    raw = os.getenv("HERMES_PLUGIN_PAYLOAD_MAX_CHARS", "50000")
    try:
        return max(1000, int(raw))
    except (TypeError, ValueError):
        return 50000

def _is_sensitive_hook_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.lower().replace("-", "_")
    exact = {
        "api_key",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
    }
    return lowered in exact or lowered.endswith("_api_key")

def _hook_jsonable(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 8,
    max_string: int = 8000,
    max_sequence: int = 200,
) -> Any:
    if depth > max_depth:
        return f"<{type(value).__name__} depth limit>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > max_string:
            return value[:max_string] + f"...[truncated {len(value) - max_string} chars]"
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_sequence:
                out["_truncated_items"] = len(value) - max_sequence
                break
            str_key = str(key)
            if _is_sensitive_hook_key(str_key):
                out[str_key] = "<redacted>"
            else:
                out[str_key] = _hook_jsonable(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_string=max_string,
                    max_sequence=max_sequence,
                )
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        out = [
            _hook_jsonable(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
            for item in seq[:max_sequence]
        ]
        if len(seq) > max_sequence:
            out.append({"_truncated_items": len(seq) - max_sequence})
        return out
    try:
        if hasattr(value, "model_dump"):
            try:
                # warnings=False: pydantic's serializer UserWarnings on
                # generic-union SDK models (Anthropic ParsedMessage etc.)
                # would otherwise leak to the terminal mid-response.
                dumped = value.model_dump(mode="json", warnings=False)
            except TypeError:
                try:
                    dumped = value.model_dump(mode="json")
                except TypeError:
                    dumped = value.model_dump()
            return _hook_jsonable(
                dumped,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
    except Exception:
        pass
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(value):
            return _hook_jsonable(
                asdict(value),
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
    except Exception:
        pass
    if isinstance(value, SimpleNamespace):
        return _hook_jsonable(
            vars(value),
            depth=depth + 1,
            max_depth=max_depth,
            max_string=max_string,
            max_sequence=max_sequence,
        )
    if hasattr(value, "__dict__"):
        try:
            public_attrs = {
                k: v
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
            return _hook_jsonable(
                public_attrs,
                depth=depth + 1,
                max_depth=max_depth,
                max_string=max_string,
                max_sequence=max_sequence,
            )
        except Exception:
            pass
    return str(value)[:max_string]

def _sanitize_hook_payload(value: Any) -> Any:
    payload = _hook_jsonable(value)
    limit = _hook_payload_max_chars()
    try:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)[:limit]
    if len(encoded) <= limit:
        return payload
    payload = _hook_jsonable(value, max_string=1000, max_sequence=50)
    try:
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        return str(payload)[:limit]
    if len(encoded) <= limit:
        return payload
    return {
        "_truncated": True,
        "original_type": type(value).__name__,
        "preview": encoded[:limit],
    }

def _api_request_payload_for_hook(self, api_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    body = {
        key: value
        for key, value in (api_kwargs or {}).items()
        if key not in {"timeout", "http_client"}
    }
    return _sanitize_hook_payload(
        {
            "method": "POST",
            "body": body,
        }
    )

def _api_response_payload_for_hook(
    self,
    response: Any,
    assistant_message: Any,
    *,
    finish_reason: Optional[str],
) -> Dict[str, Any]:
    # ``tool_calls`` is the raw list of provider SDK objects (e.g.
    # OpenAI ``ChatCompletionMessageToolCall``).  We deliberately hand
    # the raw objects to ``_sanitize_hook_payload`` and rely on
    # ``_hook_jsonable`` to normalise them via ``model_dump`` /
    # ``__dict__`` / dataclass introspection — a future refactor of
    # the sanitiser MUST preserve that capability or hook subscribers
    # will receive opaque ``str(obj)`` blobs here.
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    return _sanitize_hook_payload(
        {
            "model": getattr(response, "model", None),
            "finish_reason": finish_reason,
            "assistant_message": {
                "role": getattr(assistant_message, "role", "assistant"),
                "content": getattr(assistant_message, "content", None),
                "tool_calls": tool_calls,
            },
            "usage": _usage_summary_for_api_request_hook(self, response),
        }
    )

def _invoke_api_request_error_hook(
    self,
    *,
    task_id: str,
    turn_id: str,
    api_request_id: str,
    api_call_count: int,
    api_start_time: float,
    api_kwargs: Optional[Dict[str, Any]],
    error_type: str,
    error_message: str,
    status_code: Optional[int] = None,
    retry_count: Optional[int] = None,
    max_retries: Optional[int] = None,
    retryable: Optional[bool] = None,
    reason: Optional[str] = None,
) -> None:
    # Lazy module import (not from-import) so tests can replace lifecycle
    # dispatch at this call site. After first call the import is a
    # ``sys.modules`` dict lookup, so retries don't repay any real cost.
    try:
        from hermes_cli import lifecycle as _lifecycle

        if not _lifecycle.has_hook("api_request_error"):
            return
        ended_at = time.time()
        _lifecycle.invoke_hook(
            "api_request_error",
            task_id=task_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            session_id=self.session_id or "",
            platform=self.platform or "",
            model=self.model,
            provider=self.provider,
            base_url=self.base_url,
            api_mode=self.api_mode,
            api_call_count=api_call_count,
            api_duration=ended_at - api_start_time,
            started_at=api_start_time,
            ended_at=ended_at,
            status_code=status_code,
            retry_count=retry_count,
            max_retries=max_retries,
            retryable=retryable,
            reason=reason,
            error={
                "type": error_type,
                "message": error_message,
            },
            request=_api_request_payload_for_hook(self, api_kwargs),
        )
    except Exception:
        pass


def _clean_session_content(content: str) -> str:
    """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
    if not content:
        return content
    content = convert_scratchpad_to_think(content)
    content = re.sub(r'\n+(<think>)', r'\n\1', content)
    content = re.sub(r'(</think>)\n+', r'\1\n', content)
    return content.strip()

def _redact_message_content(content):
    """Apply secret redaction to message content (str or list-of-parts).

    Handles both plain-string content and the OpenAI/Anthropic multimodal
    shape where ``content`` is a list of ``{"type": "text", "text": ...}``
    / ``{"type": "image_url", ...}`` / ``{"type": "input_text", "content": ...}``
    parts. Image / binary parts are left untouched; only text fields are
    passed through ``redact_sensitive_text``.

    Respects ``HERMES_REDACT_SECRETS`` via ``redact_sensitive_text`` —
    when disabled the helper is effectively a no-op.
    """
    if content is None:
        return content
    if isinstance(content, str):
        return redact_sensitive_text(content)
    if isinstance(content, list):
        redacted = []
        for part in content:
            if isinstance(part, dict):
                part = dict(part)
                if isinstance(part.get("text"), str):
                    part["text"] = redact_sensitive_text(part["text"])
                if isinstance(part.get("content"), str):
                    part["content"] = redact_sensitive_text(part["content"])
            redacted.append(part)
        return redacted
    return content

def _record_file_mutation_result(
    self,
    tool_name: str,
    args: Dict[str, Any],
    result: Any,
    is_error: bool,
) -> None:
    """Record a ``write_file`` / ``patch`` outcome for the turn-end verifier.

    On failure, store ``{path: {error_preview, tool}}`` entries.  On
    success, remove any prior failure entries for the same paths (the
    model recovered within the turn).  Silently no-ops if the per-turn
    state dict hasn't been initialised yet (e.g. a tool dispatched
    outside ``run_conversation``).
    """
    if tool_name not in _FILE_MUTATING_TOOLS:
        return
    state = getattr(self, "_turn_failed_file_mutations", None)
    if state is None:
        return
    targets = _extract_file_mutation_targets(tool_name, args)
    if not targets:
        return
    landed = file_mutation_result_landed(tool_name, result)
    if landed:
        landed_paths = _extract_landed_file_mutation_paths(tool_name, args, result)
        changed = getattr(self, "_turn_file_mutation_paths", None)
        if changed is not None:
            changed.update(landed_paths)
        # Feed the checkpoint agent-write ledger so /rollback's safe mode
        # can tell Hermes-authored content from later user hand-edits.
        mgr = getattr(self, "_checkpoint_mgr", None)
        if mgr is not None and getattr(mgr, "enabled", False):
            for _p in landed_paths:
                try:
                    mgr.record_agent_write(_p)
                except Exception:
                    pass
    if is_error and not landed:
        preview = _extract_error_preview(result)
        for path in targets:
            # Keep the FIRST error we saw for a given path unless we
            # later see success.  A repeated failure with a different
            # message shouldn't silently overwrite the original.
            if path not in state:
                state[path] = {

                    "tool": tool_name,
                    "error_preview": preview,
                }
    else:
        for path in targets:
            state.pop(path, None)

def _file_mutation_verifier_enabled(self) -> bool:
    """Check whether the per-turn file-mutation verifier footer is on.

    Config path: ``display.file_mutation_verifier`` (bool, default True).
    ``HERMES_FILE_MUTATION_VERIFIER`` env var overrides config.  Exposed
    as a method so tests can patch a single seam without reaching into
    the private ``_turn_failed_file_mutations`` state dict.

    The config lookup is read once per agent and cached (mirroring
    ``_credits_notices_enabled``) — the footer gate runs at the end of
    every turn, and a config flip applying on the next session is fine.
    The env-var override stays authoritative on every call and is never
    cached, so tests and operators can still flip it at runtime.
    """
    try:
        import os as _os
        env = _os.environ.get("HERMES_FILE_MUTATION_VERIFIER")
        if env is not None:
            return env.strip().lower() not in {"0", "false", "no", "off"}
        cached = getattr(self, "_file_mutation_verifier_enabled_cache", None)
        if cached is not None:
            return cached
        # Read from the persisted config.yaml so gateway and CLI share
        # the same setting.  Import lazily to avoid a startup-time cycle.
        try:
            from hermes_cli.config import load_config as _load_config
            _cfg = _load_config() or {}
        except Exception:
            _cfg = {}
        _display = _cfg.get("display") if isinstance(_cfg, dict) else None
        if isinstance(_display, dict) and "file_mutation_verifier" in _display:
            enabled = bool(_display.get("file_mutation_verifier"))
        else:
            enabled = True  # safe default: verifier on
        self._file_mutation_verifier_enabled_cache = enabled
        return enabled
    except Exception:
        pass
    return True  # safe default: verifier on

def _neutralize_footer_paths(text: str) -> str:
    """Wrap bare file paths in backticks so they aren't auto-delivered.

    The gateway's ``extract_local_files`` scans response text for bare
    absolute/home paths ending in a deliverable extension and uploads
    any that exist on disk as native attachments — but it explicitly
    skips paths inside inline-code (`` `...` ``) spans.  Backticking
    every path the footer renders defeats that auto-detection while
    keeping the path fully human-readable.  Paths already wrapped in a
    backtick (the negative lookbehind excludes a preceding `` ` ``) are
    left untouched so we never double-wrap.
    """
    if not text:
        return text
    return _FOOTER_PATH_RE.sub(lambda m: f"`{m.group(0)}`", text)

def _format_file_mutation_failure_footer(failed: Dict[str, Dict[str, Any]]) -> str:
    """Render the per-turn failed-mutation dict as a user-facing footer.

    Displays up to 10 paths with their first error preview, then a
    count of any additional failures.  Returns an empty string when
    the dict is empty so callers can concatenate unconditionally.

    Every file path that reaches the user-facing text — both the bullet
    path and any path echoed inside the tool's error preview — is
    backtick-wrapped via ``_neutralize_footer_paths`` so the gateway's
    bare-path media extractor can never auto-attach a protected file
    (e.g. ``~/.hermes/config.yaml``) to a messaging channel (#35584).
    """
    if not failed:
        return ""
    lines = [
        "⚠️ File-mutation verifier: "
        f"{len(failed)} file(s) were NOT modified this turn despite any "
        "wording above that may suggest otherwise. Run `git status` or "
        "`read_file` to confirm."
    ]
    shown = 0
    for path, info in failed.items():
        if shown >= 10:
            break
        preview = (info.get("error_preview") or "").strip()
        tool = info.get("tool") or "patch"
        if preview:
            lines.append(f"  • `{path}` — [{tool}] {preview}")
        else:
            lines.append(f"  • `{path}` — [{tool}] failed")
        shown += 1
    remaining = len(failed) - shown
    if remaining > 0:
        lines.append(f"  • … and {remaining} more")
    # Neutralize any path the preview text echoed (the bullet path is
    # already backticked above; the lookbehind keeps it from being
    # double-wrapped).
    return _neutralize_footer_paths("\n".join(lines))

def _turn_completion_explainer_enabled(self) -> bool:
    """Check whether the end-of-turn completion explainer footer is on.

    Config path: ``display.turn_completion_explainer`` (bool, default
    True).  ``HERMES_TURN_COMPLETION_EXPLAINER`` env var overrides
    config.  Exposed as a method so tests can patch a single seam,
    mirroring ``_file_mutation_verifier_enabled``.

    The config lookup is read once per agent and cached (mirroring
    ``_credits_notices_enabled``) — the gate runs at the end of every
    turn, and a config flip applying on the next session is fine.
    The env-var override stays authoritative on every call and is never
    cached, so tests and operators can still flip it at runtime.
    """
    try:
        import os as _os
        env = _os.environ.get("HERMES_TURN_COMPLETION_EXPLAINER")
        if env is not None:
            return env.strip().lower() not in {"0", "false", "no", "off"}
        cached = getattr(self, "_turn_completion_explainer_enabled_cache", None)
        if cached is not None:
            return cached
        # Read from the persisted config.yaml so gateway and CLI share
        # the same setting.  Import lazily to avoid a startup-time cycle.
        try:
            from hermes_cli.config import load_config as _load_config
            _cfg = _load_config() or {}
        except Exception:
            _cfg = {}
        _display = _cfg.get("display") if isinstance(_cfg, dict) else None
        if isinstance(_display, dict) and "turn_completion_explainer" in _display:
            enabled = bool(_display.get("turn_completion_explainer"))
        else:
            enabled = True  # safe default: explainer on
        self._turn_completion_explainer_enabled_cache = enabled
        return enabled
    except Exception:
        pass
    return True  # safe default: explainer on

def _format_turn_completion_explanation(
    turn_exit_reason: str, persistence_cause: Optional[str] = None
) -> str:
    """Render a user-facing explanation for an abnormal turn ending.

    Maps the internal ``turn_exit_reason`` to a short, actionable
    message so a turn that produced no usable assistant reply (empty
    content after retries, a partial/truncated stream, a still-pending
    tool result, or an iteration/budget limit) is never silent from
    the UI's perspective — the symptom users report in #34452.

    ``persistence_cause`` refines the ``session_persistence_failed``
    wording (see ``classify_persistence_error``): lock contention gets
    "storage was busy, send it again" instead of the disk-space advice,
    which was a misdiagnosis for that failure mode. It is optional and
    ignored for every other reason, so one-argument callers keep the
    exact behavior they had before.

    Returns an empty string for reasons that are NOT abnormal (e.g.
    a normal ``text_response(...)`` exit), so callers can concatenate
    or substitute unconditionally without warning on healthy turns
    like a terse ``Done.``.
    """
    if not turn_exit_reason:
        return ""
    reason = str(turn_exit_reason)

    # Normal completion — stay quiet.  ``text_response(...)`` is the
    # healthy terminal; anything that produced a real reply is fine.
    if reason.startswith("text_response"):
        return ""

    prefix = "⚠️ No reply: "
    if reason == "empty_response_exhausted":
        return (
            prefix
            + "the model returned empty content after retries and any "
            "fallback providers. Try `continue`, switch model/provider, "
            "or inspect the tool output above."
        )
    if reason == "all_retries_exhausted_no_response":
        return (
            prefix
            + "all API retries were exhausted before a response was "
            "produced (provider errors / rate limits). Try `continue` "
            "or switch provider."
        )
    if reason == "partial_stream_recovery":
        return (
            prefix
            + "streaming stopped early and only a partial response was "
            "recovered. Send `continue` to resume from where it stopped."
        )
    if reason == "fallback_prior_turn_content":
        return (
            prefix
            + "no new content was produced this turn; showing recovered "
            "prior context. Send `continue` to retry."
        )
    if reason == "interrupted_during_api_call":
        return (
            prefix
            + "the request was interrupted mid-call before a reply was "
            "received. Send `continue` to retry."
        )
    if reason == "budget_exhausted":
        return (
            prefix
            + "the per-turn iteration/cost budget was exhausted before a "
            "final answer. Send `continue` to keep going."
        )
    if reason == "ollama_runtime_context_too_small":
        return (
            prefix
            + "the local model's context window was too small to finish. "
            "Increase the context size or use a larger model."
        )
    if reason.startswith("max_iterations_reached"):
        return (
            prefix
            + "the maximum tool-iteration limit was reached before a "
            "final answer. Send `continue` to keep going, or raise "
            "`max_iterations`."
        )
    if reason.startswith("error_near_max_iterations"):
        return (
            prefix
            + "an error occurred near the iteration limit before a final "
            "answer. Check the tool output above, then send `continue`."
        )
    if reason == "pending_tool_result":
        return (
            prefix
            + "the turn stopped while a tool result was still pending and "
            "the model produced no follow-up text. Send `continue` to "
            "let it summarize."
        )
    if reason == "session_persistence_failed":
        cause = persistence_cause or "unknown"
        if cause == "compression":
            return (
                prefix
                + "the turn was stopped because another process was "
                "compressing this session. Your message should already be "
                "saved — please send it again after compression completes."
            )
        if cause == "compression_closed":
            return (
                prefix
                + "the turn was stopped because this session was rotated "
                "by context compression and its live continuation could "
                "not be adopted. The storage itself is healthy — refresh "
                "the client (or start a new turn) so it picks up the new "
                "session id, then send your message again."
            )
        if cause == "turn_lease":
            return (
                prefix
                + "the turn was stopped because another Hermes process "
                "took over this session. Your reply was not saved — wait "
                "for the other process to finish, then send your message "
                "again."
            )
        if cause == "locked":
            return (
                prefix
                + "the turn was stopped because session storage was busy "
                "(another Hermes process was writing to the state "
                "database). Your message should already be saved — "
                "please send it again in a moment."
            )
        if cause == "corrupt":
            return (
                prefix
                + "the turn was stopped because the state database "
                "reported structural corruption (the transcript would "
                "have been lost on restart). Freeing disk space will "
                "not help. Recovery options:\n"
                "1. Run `hermes doctor --fix`\n"
                "2. Salvage with: sqlite3 ~/.hermes/state.db \".recover\" "
                "(then replace state.db)\n"
                "3. Restore from a backup in ~/.hermes/backups/\n"
                "Then send your message again."
            )
        if cause == "disk":
            return (
                prefix
                + "the turn was stopped because session storage could not "
                "be written (the transcript would have been lost on "
                "restart). This is often a full disk — free some space "
                "(or fix state.db permissions), then send your message "
                "again."
            )
        return (
            prefix
            + "the turn was stopped because session storage could not be "
            "written (the transcript would have been lost on restart). "
            "Check the state database health (`hermes doctor`), then "
            "send your message again."
        )
    # Unknown/diagnostic-only reasons (e.g. "unknown", guardrail_halt
    # which already surfaces its own message) — don't second-guess.
    return ""


def dump_api_request_debug(
    agent,
    api_kwargs: Dict[str, Any],
    *,
    reason: str,
    error: Optional[Exception] = None,
) -> Optional[Path]:
    """
    Dump a debug-friendly HTTP request record for the active inference API.

    Captures the request body from api_kwargs (excluding transport-only keys
    like timeout). Intended for debugging provider-side 4xx failures where
    retries are not useful.
    """
    import agent.error_reporting as error_reporting
    import agent.status_output as status_output
    try:
        body = copy.deepcopy(api_kwargs)
        body.pop("timeout", None)
        body = {k: v for k, v in body.items() if v is not None}

        api_key = None
        try:
            api_key = getattr(agent.client, "api_key", None)
        except Exception as e:
            logger.debug("Could not extract API key for debug dump: %s", e)

        dump_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "session_id": agent.session_id,
            "reason": reason,
            "request": {
                "method": "POST",
                "url": f"{agent.base_url.rstrip('/')}{'/responses' if agent.api_mode == 'codex_responses' else '/chat/completions'}",
                "headers": {
                    "Authorization": f"Bearer {error_reporting._mask_api_key_for_logs(agent, api_key)}",
                    "Content-Type": "application/json",
                },
                "body": body,
            },
        }

        if error is not None:
            error_info: Dict[str, Any] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            for attr_name in ("status_code", "request_id", "code", "param", "type"):
                attr_value = getattr(error, attr_name, None)
                if attr_value is not None:
                    error_info[attr_name] = attr_value

            body_attr = getattr(error, "body", None)
            if body_attr is not None:
                error_info["body"] = body_attr

            response_obj = getattr(error, "response", None)
            if response_obj is not None:
                try:
                    error_info["response_status"] = getattr(response_obj, "status_code", None)
                    error_info["response_text"] = response_obj.text
                except Exception as e:
                    logger.debug("Could not extract error response details: %s", e)

            dump_payload["error"] = error_info

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Sanitize the session ID into a traversal-free path segment — it can
        # originate from untrusted input (X-Hermes-Session-Id header), and an
        # unsanitized "../"-shaped ID would write the dump outside logs_dir.
        safe_sid = safe_session_filename_component(agent.session_id)
        dump_file = agent.logs_dir / f"request_dump_{safe_sid}_{timestamp}.json"

        # Redact secrets before persisting/printing. This dump captures the
        # full request body (system prompt, tool defs, context-embedded
        # values), and this path fires unconditionally on API errors — so it
        # otherwise lands any context-embedded secret in cleartext on disk.
        # Run the serialized dump through the same scrubber used for logs/tool
        # output, then hand the resulting payload back to the shared atomic
        # JSON writer so request dumps keep the same write semantics as before.
        from agent.redact import redact_sensitive_text
        _serialized = json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str)
        _redacted_payload = json.loads(redact_sensitive_text(_serialized, force=True))
        atomic_json_write(dump_file, _redacted_payload, default=str)

        status_output._vprint(agent, f"{agent.log_prefix}🧾 Request debug dump written to: {dump_file}")

        if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
            print(json.dumps(_redacted_payload, ensure_ascii=False, indent=2, default=str))

        return dump_file
    except Exception as dump_error:
        if agent.verbose_logging:
            logger.warning("Failed to dump API request debug payload: %s", dump_error)
        return None

def extract_api_error_context(error: Exception) -> Dict[str, Any]:
    """Extract structured rate-limit details from provider errors."""
    context: Dict[str, Any] = {}

    body = getattr(error, "body", None)
    payload = None
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("type") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if not message and isinstance(payload.get("error"), str):
            # xAI uses a top-level string ``error`` beside a structured
            # ``code`` (for example personal-team-blocked:spending-limit).
            message = payload.get("error")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        for key in ("resets_at", "reset_at"):
            value = payload.get(key)
            if value not in {None, ""}:
                context["reset_at"] = value
                break
        retry_after = payload.get("retry_after")
        if retry_after not in {None, ""} and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after and "reset_at" not in context:
            try:
                context["reset_at"] = time.time() + float(retry_after)
            except (TypeError, ValueError):
                pass
        ratelimit_reset = headers.get("x-ratelimit-reset")
        if ratelimit_reset and "reset_at" not in context:
            context["reset_at"] = ratelimit_reset

    if "message" not in context:
        raw_message = str(error).strip()
        if raw_message:
            context["message"] = raw_message[:500]

    if "reset_at" not in context:
        message = context.get("message") or ""
        if isinstance(message, str):
            delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
            if delay_match:
                value = float(delay_match.group(1))
                seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                context["reset_at"] = time.time() + seconds
            else:
                resets_in_match = re.search(
                    r"resets?\s+in\s+"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b\s*)?"
                    r"(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b)?",
                    message,
                    re.IGNORECASE,
                )
                if resets_in_match and any(resets_in_match.groups()):
                    hours = float(resets_in_match.group(1) or 0)
                    minutes = float(resets_in_match.group(2) or 0)
                    seconds = float(resets_in_match.group(3) or 0)
                    context["reset_at"] = time.time() + (hours * 3600) + (minutes * 60) + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

    return context

def apply_pending_steer_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """Append any pending /steer text to the last tool result in this turn.

    Called at the end of a tool-call batch, before the next API call.
    The steer is appended to the last ``role:"tool"`` message's content
    with a clear marker so the model understands it came from the user
    and NOT from the tool itself. Role alternation is preserved —
    nothing new is inserted, we only modify existing content.

    Args:
        messages: The running messages list.
        num_tool_msgs: Number of tool results appended in this batch;
            used to locate the tail slice safely.
    """
    import agent.interruption as interruption
    if num_tool_msgs <= 0 or not messages:
        return
    steer_text = interruption._drain_pending_steer(agent)
    if not steer_text:
        return
    # Find the last tool-role message in the recent tail. Skipping
    # non-tool messages defends against future code appending
    # something else at the boundary.
    target_idx = None
    for j in range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1):
        msg = messages[j]
        if isinstance(msg, dict) and msg.get("role") == "tool":
            target_idx = j
            break
    if target_idx is None:
        # No tool result in this batch (e.g. all skipped by interrupt);
        # put the steer back so the caller's fallback path can deliver
        # it as a normal next-turn user message.
        _lock = getattr(agent, "_pending_steer_lock", None)
        if _lock is not None:
            with _lock:
                if agent._pending_steer:
                    agent._pending_steer = agent._pending_steer + "\n" + steer_text
                else:
                    agent._pending_steer = steer_text
        else:
            existing = getattr(agent, "_pending_steer", None)
            agent._pending_steer = (existing + "\n" + steer_text) if existing else steer_text
        return
    marker = format_steer_marker(steer_text)
    existing_content = messages[target_idx].get("content", "")
    if not isinstance(existing_content, str):
        # Anthropic multimodal content blocks — preserve them and append
        # a text block at the end.
        try:
            blocks = list(existing_content) if existing_content else []
            blocks.append({"type": "text", "text": marker.lstrip()})
            messages[target_idx]["content"] = blocks
        except Exception:
            # Fall back to string replacement if content shape is unexpected.
            messages[target_idx]["content"] = f"{existing_content}{marker}"
    else:
        messages[target_idx]["content"] = existing_content + marker
    logger.info(
        "Delivered /steer to agent after tool batch (%d chars): %s",
        len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )
