"""Per-attempt recovery bookkeeping for the conversation turn loop.

The inner retry loop in ``run_conversation`` (``while retry_count <
max_retries``) makes several distinct recovery attempts on a single model API
call: a credential-pool 429 retry, a per-provider OAuth refresh (codex,
anthropic, nous, copilot), a long-context compression restart, a length-
continuation restart, and a handful of format-recovery branches (thinking-
signature stripping, multimodal-tool-content stripping, llama.cpp grammar
fallback, image shrink, invalid-encrypted-content, 1M-beta header).

Each of those branches is guarded by a one-shot boolean so it fires at most
once per attempt. They used to be ~16 bare ``*_attempted`` / ``has_retried_*``
/ restart locals declared inline before the loop and threaded
through its 2,400-line body. ``TurnRetryState`` collapses them into one object
the loop mutates in place (``state.codex_auth_retry_attempted = True``), giving
the recovery bookkeeping a single named, testable home.

Loop-control variables (``retry_count``, ``max_retries``,
``max_compression_attempts``) intentionally stay as plain locals — they are the
``while`` mechanics, not recovery bookkeeping, and putting them on the object
would add indirection without clarifying anything.

This module is dependency-free so it can be unit-tested in isolation and
imported by the turn loop without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class RestartDirective(Enum):
    NONE = "none"
    COMPRESSED_MESSAGES = "compressed_messages"
    LENGTH_CONTINUATION = "length_continuation"
    REBUILT_MESSAGES = "rebuilt_messages"
    REDIRECTED_MESSAGES = "redirected_messages"


@dataclass
class TurnRetryState:
    """One-shot recovery guards + restart signals for a single API-call attempt.

    A fresh instance is created for each iteration of the outer turn loop
    (once per ``api_call_count``). Each guard fires its recovery branch at most
    once; the restart directive is read by the loop after the attempt
    to decide whether to rebuild the request and retry.
    """

    # ── Per-provider OAuth / credential refresh guards ───────────────────
    codex_auth_retry_attempted: bool = False
    anthropic_auth_retry_attempted: bool = False
    copilot_auth_retry_attempted: bool = False
    # Copilot surfaces a stale/degraded credential as a 400
    # ``model_not_available_for_integrator`` / ``model_not_supported`` instead
    # of a clean 401 (e.g. a raw OAuth token seeded when the token exchange
    # degraded at startup, routing the request to the restricted
    # ``copilot-language-server`` integrator). Guard a single-shot forced
    # re-exchange + client rebuild for that case, separate from the 401 guard
    # so both can fire within one attempt if needed.
    copilot_stale_cred_retry_attempted: bool = False
    vertex_auth_retry_attempted: bool = False

    # ── Format / payload recovery guards ─────────────────────────────────
    thinking_sig_retry_attempted: bool = False
    invalid_encrypted_content_retry_attempted: bool = False
    native_compaction_reject_retry_attempted: bool = False
    image_shrink_retry_attempted: bool = False
    multimodal_tool_content_retry_attempted: bool = False
    oauth_1m_beta_retry_attempted: bool = False
    llama_cpp_grammar_retry_attempted: bool = False

    # ── Transport / rate-limit recovery ──────────────────────────────────
    primary_recovery_attempted: bool = False
    has_retried_429: bool = False

    # ── Auth-failure provider failover ───────────────────────────────────
    # Set once we've escalated a persistent 401/403 (after the per-provider
    # credential-refresh attempt above failed) to the fallback chain, so we
    # don't loop on the same auth failover within one attempt.
    auth_failover_attempted: bool = False

    # ── Restart directive (read by the outer loop after the attempt) ─────
    restart: RestartDirective = RestartDirective.NONE

    def __iter__(self):
        # Convenience for debugging / tests: iterate (name, value) pairs.
        for f in fields(self):
            yield f.name, getattr(self, f.name)
