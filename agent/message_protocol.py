"""Responsibility-owned agent message protocol behavior."""
import json
import logging
import re
from typing import Any, Dict, List, Optional
from agent.message_sanitization import (
    _FULL_ARGS_LOG_BOUND,
    TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER,
    VALID_API_ROLES,
    coalesce_tool_call_id,
    get_tool_call_name,
    is_thinking_only_assistant,
)
from agent.tool_dispatch_helpers import make_tool_result_message
from agent.message_sanitization import (  # noqa: F401
    coalesce_tool_call_id as _sanitize_coalesce_tool_call_id,
    uniquify_tool_call_ids as _sanitize_uniquify_tool_call_ids,
)
from agent.codex_responses_adapter import (
    _derive_responses_function_call_id as _codex_derive_responses_function_call_id,
    _deterministic_call_id as _codex_deterministic_call_id,
    _split_responses_tool_id as _codex_split_responses_tool_id,  # also used by _sync_external_memory_for_turn (memory boundary)
)


logger = logging.getLogger(__name__)

_REASONING_TAG_NAMES = (
    "think", "thinking", "reasoning", "REASONING_SCRATCHPAD", "thought"
)
_TOOL_CALL_TAG_NAMES = (
    "tool_call", "tool_calls", "tool_result", "function_call", "function_calls"
)
_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _REASONING_TAG_NAMES
)
_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)
_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r'(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*'
    r'<function\b[^>]*\bname\s*=[^>]*>'
    r'(?:(?:(?!</function>).)*)</function>',
    re.DOTALL | re.IGNORECASE,
)
_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf'(?:^|\n)[ \t]*<(?:{"|".join(_REASONING_TAG_NAMES)})\b[^>]*>.*$',
    re.DOTALL | re.IGNORECASE,
)
_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf'</?(?:{"|".join(_REASONING_TAG_NAMES)})>\s*', re.IGNORECASE
)
_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf'</(?:{"|".join(_TOOL_CALL_TAG_NAMES)}|function)>\s*', re.IGNORECASE
)
_INTERRUPTED_PLACEHOLDER = "[response interrupted]"

EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_thinking_prefill",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_kanban_stop_synthetic",
    "_dropped_toolcall_nudge",
)


def is_ephemeral_scaffolding(message: Any) -> bool:
    """Return whether a message is internal retry-only scaffolding."""
    return isinstance(message, dict) and any(
        message.get(flag) for flag in EPHEMERAL_SCAFFOLDING_FLAGS
    )

_TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER = (
    "[hermes-agent: tool call arguments were corrupted in this session and "
    "have been dropped to keep the conversation alive. See issue #15236.]"
)

_VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

def _build_system_prompt_parts(self, system_message: str = None) -> Dict[str, str]:
    """Forwarder — see ``agent.system_prompt.build_system_prompt_parts``."""
    from agent.system_prompt import build_system_prompt_parts
    return build_system_prompt_parts(self, system_message=system_message)

def _build_system_prompt(self, system_message: str = None) -> str:
    """Forwarder — see ``agent.system_prompt.build_system_prompt``."""
    from agent.system_prompt import build_system_prompt
    return build_system_prompt(self, system_message=system_message)

def _get_tool_call_id_static(tc) -> str:
    """Extract call ID from a tool_call entry (dict or object).

    Forwarder — policy owner is
    ``agent.message_sanitization.coalesce_tool_call_id`` (audit F4).
    """
    return _sanitize_coalesce_tool_call_id(tc)

def _get_tool_call_name_static(tc) -> str:
    """Extract function name from a tool_call entry (dict or object).

    Gemini's OpenAI-compatibility endpoint requires every `role: tool`
    message to carry the matching function name. OpenAI/Anthropic/ollama
    tolerate its absence, so the field is best-effort: callers fall back
    to "" and the message still works elsewhere.
    """
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return fn.get("name", "") or ""
        return ""
    fn = getattr(tc, "function", None)
    return getattr(fn, "name", "") or ""


def _is_thinking_only_assistant(
    msg: Dict[str, Any],
    *,
    drop_codex_reasoning_items: bool = True,
) -> bool:
    """Return True if ``msg`` is an assistant turn whose only payload is reasoning.

    "Thinking-only" means the model emitted reasoning (``reasoning`` or
    ``reasoning_content``) but no visible text and no tool_calls. When sent
    back to providers that convert reasoning into thinking blocks (native
    Anthropic, OpenRouter Anthropic, third-party Anthropic-compatible
    gateways), the resulting message has only thinking blocks — which
    Anthropic rejects with HTTP 400 "The final block in an assistant
    message cannot be `thinking`."

    Symmetric with Claude Code's ``filterOrphanedThinkingOnlyMessages``
    (src/utils/messages.ts). We drop the whole turn from the API copy
    rather than fabricating stub text — the message log (UI transcript)
    keeps the reasoning block; only the wire copy is cleaned.
    """
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return False
    # Prefill stubs are thinking-only by construction; check before content
    # inspection since repair_empty_non_final_messages may have healed content.
    if msg.get("_thinking_prefill"):
        return True
    # Does it have any actual output?
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            return False
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                if block:  # non-empty non-dict string etc.
                    return False
                continue
            btype = block.get("type")
            if btype in {"thinking", "redacted_thinking"}:
                continue
            if btype == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    return False
                continue
            # tool_use, image, document, etc. — real payload
            return False
    elif content is not None and content != "":
        return False
    # A native compaction checkpoint makes a carrier never thinking-only,
    # regardless of api_mode or which reasoning field is populated. The
    # checkpoint is the server-side stand-in for already-pruned history
    # and exists in exactly one place; the codex_responses adapter also
    # surfaces commentary text via msg["reasoning"], so the string branch
    # below would otherwise drop a carrier before the sidecar is ever
    # inspected. Checked here — above every reasoning branch — so no
    # carrier shape can fall into a drop path (#82108 review finding).
    from agent.native_compaction import has_compaction_checkpoint

    if has_compaction_checkpoint(msg.get("codex_reasoning_items")):
        return False
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return True
    # reasoning_details list form
    rd = msg.get("reasoning_details")
    if isinstance(rd, list) and rd:
        return True
    # Codex Responses stores encrypted reasoning state under a separate
    # assistant-message key. Treat only real reasoning items as
    # thinking-only; empty/junk lists should fall through to the generic
    # empty-turn handling instead of being dropped here.
    codex_items = msg.get("codex_reasoning_items")
    if drop_codex_reasoning_items and isinstance(codex_items, list):
        return any(
            isinstance(item, dict) and item.get("type") == "reasoning"
            for item in codex_items
        )
    return False


def _cap_delegate_task_calls(tool_calls: list) -> list:
    """Truncate excess delegate_task calls to max_concurrent_children.

    The delegate_tool caps the task list inside a single call, but the
    model can emit multiple separate delegate_task tool_calls in one
    turn.  This truncates the excess, preserving all non-delegate calls.

    Returns the original list if no truncation was needed.
    """
    from tools.delegate_tool import _get_max_concurrent_children
    max_children = _get_max_concurrent_children()
    delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
    if delegate_count <= max_children:
        return tool_calls
    kept_delegates = 0
    truncated = []
    for tc in tool_calls:
        if tc.function.name == "delegate_task":
            if kept_delegates < max_children:
                truncated.append(tc)
                kept_delegates += 1
        else:
            truncated.append(tc)
    logger.warning(
        "Truncated %d excess delegate_task call(s) to enforce "
        "max_concurrent_children=%d limit",
        delegate_count - max_children, max_children,
    )
    return truncated

def _deduplicate_tool_calls(tool_calls: list) -> list:
    """Remove duplicate (tool_name, arguments) pairs within a single turn.

    Valid JSON arguments are canonicalized so equivalent objects do not
    evade deduplication merely because their keys or whitespace differ.
    Malformed arguments retain their raw representation rather than being
    repaired here. Only the first occurrence of each unique pair is kept.
    Returns the original list if no duplicates were found.
    """
    seen: set = set()
    unique: list = []
    for tc in tool_calls:
        arguments = tc.function.arguments
        try:
            arguments = json.dumps(
                json.loads(arguments), separators=(",", ":"), sort_keys=True
            )
        except (TypeError, ValueError):
            pass
        key = (tc.function.name, arguments)
        if key not in seen:
            seen.add(key)
            unique.append(tc)
        else:
            logger.warning("Removed duplicate tool call: %s", tc.function.name)
    return unique if len(unique) < len(tool_calls) else tool_calls

def _uniquify_tool_call_ids(tool_calls: list) -> list:
    """Ensure every tool call in a single assistant turn has a distinct id.

    Forwarder — policy owner is
    ``agent.message_sanitization.uniquify_tool_call_ids`` (audit F4).
    First occurrence keeps its id; later collisions get a deterministic
    ``<id>_d<n>`` suffix (never uuid4 — prompt-cache prefix stability).
    Mutates entries in place and returns the same list.
    """
    return _sanitize_uniquify_tool_call_ids(tool_calls)


def _invalidate_system_prompt(self):
    """Forwarder — see ``agent.system_prompt.invalidate_system_prompt``."""
    from agent.system_prompt import invalidate_system_prompt
    invalidate_system_prompt(self)

def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """Generate a deterministic call_id from tool call content.

    Used as a fallback when the API doesn't provide a call_id.
    Deterministic IDs prevent cache invalidation — random UUIDs would
    make every API call's prefix unique, breaking OpenAI's prompt cache.
    """
    return _codex_deterministic_call_id(fn_name, arguments, index)

def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a stored tool id into (call_id, response_item_id)."""
    return _codex_split_responses_tool_id(raw_id)

def _derive_responses_function_call_id(
    self,
    call_id: str,
    response_item_id: Optional[str] = None,
) -> str:
    """Build a valid Responses `function_call.id` (must start with `fc_`)."""
    return _codex_derive_responses_function_call_id(call_id, response_item_id)

def sanitize_tool_call_arguments(
    messages: list,
    *,
    logger=None,
    session_id: str = None,
    cursor: Optional[dict] = None,
) -> int:
    """Repair corrupted assistant tool-call argument JSON in-place.

    ``cursor`` (optional) is a caller-owned dict used to skip re-validating
    messages already validated on a previous call.  It stores, under
    ``"prefix"``, the exact message *objects* (strong references) validated
    last time, in order.  On the next call, the longest contiguous prefix of
    ``messages`` whose objects are ``is``-identical to the stored prefix is
    skipped; scanning starts at the first divergence (conservative: any
    reordering, truncation, compression rewrite, or mid-list insertion breaks
    identity at that index and everything from there is re-scanned).

    Safety argument for skipping: a message in the matched prefix was fully
    scanned before — every tool_call argument was either already valid JSON
    or was rewritten to ``"{}"`` (valid).  The only code paths that mutate
    ``function["arguments"]`` on live history dicts between calls are the
    surrogate / non-ASCII sanitizers, which substitute characters *inside*
    JSON string values and cannot invalidate JSON syntax.  Compression,
    repair, undo, and steer paths replace or reorder message dicts, which
    breaks the identity match and forces a re-scan.  Holding strong
    references (the objects themselves, not ``id()``s) makes address reuse
    aliasing (#50372-style) impossible.
    """
    log = logger or logging.getLogger(__name__)
    if not isinstance(messages, list):
        return 0

    start_index = 0
    if cursor is not None:
        prev_prefix = cursor.get("prefix")
        if isinstance(prev_prefix, list):
            limit = min(len(prev_prefix), len(messages))
            while start_index < limit and messages[start_index] is prev_prefix[start_index]:
                start_index += 1

    repaired = 0
    marker = TOOL_CALL_ARGUMENTS_CORRUPTION_MARKER

    def _prepend_marker(tool_msg: dict) -> None:
        existing = tool_msg.get("content")
        if isinstance(existing, str):
            if not existing:
                tool_msg["content"] = marker
            elif not existing.startswith(marker):
                tool_msg["content"] = f"{marker}\n{existing}"
            return
        if existing is None:
            tool_msg["content"] = marker
            return
        try:
            existing_text = json.dumps(existing)
        except TypeError:
            existing_text = str(existing)
        tool_msg["content"] = f"{marker}\n{existing_text}"

    message_index = start_index
    while message_index < len(messages):
        msg = messages[message_index]
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            message_index += 1
            continue

        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            message_index += 1
            continue

        insert_at = message_index + 1
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            arguments = function.get("arguments")
            if arguments is None or arguments == "":
                function["arguments"] = "{}"
                continue
            if isinstance(arguments, str) and not arguments.strip():
                function["arguments"] = "{}"
                continue
            if not isinstance(arguments, str):
                continue

            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                # Use the canonical ``call_id || id`` precedence so both the
                # scan for an existing tool result and any inserted stub key
                # on the same id the rest of the pipeline uses. Keying on bare
                # ``id`` here would fail to find a result built with ``call_id``
                # (Codex Responses format) and insert a duplicate stub that
                # itself becomes an orphan (#58168).
                tool_call_id = coalesce_tool_call_id(tool_call) or None
                function_name = function.get("name", "?")
                # Log the FULL original argument string (bounded), not an
                # 80-char preview: this branch is about to overwrite the
                # only copy of these bytes in the transcript with "{}", and
                # for a truncated write_file/patch call the destroyed
                # arguments contain real user content (#80498 — streamed
                # file content survived only as a log preview). A corrupted
                # call is rare, so the oversized WARNING is a fair price for
                # making the data recoverable from agent.log.
                preview = arguments[:_FULL_ARGS_LOG_BOUND]
                log.warning(
                    "Corrupted tool_call arguments repaired before request "
                    "(session=%s, message_index=%s, tool_call_id=%s, function=%s, "
                    "original_arguments=%r)",
                    session_id or "-",
                    message_index,
                    tool_call_id or "-",
                    function_name,
                    preview,
                )
                function["arguments"] = "{}"

                existing_tool_msg = None
                scan_index = message_index + 1
                while scan_index < len(messages):
                    candidate = messages[scan_index]
                    if not isinstance(candidate, dict) or candidate.get("role") != "tool":
                        break
                    if candidate.get("tool_call_id") == tool_call_id:
                        existing_tool_msg = candidate
                        break
                    scan_index += 1

                if existing_tool_msg is None:
                    messages.insert(
                        insert_at,
                        make_tool_result_message(
                            function_name if function_name != "?" else "",
                            marker,
                            tool_call_id,
                        ),
                    )
                    insert_at += 1
                else:
                    _prepend_marker(existing_tool_msg)

                repaired += 1

        message_index += 1

    if cursor is not None:
        # Strong references to the exact objects validated this call, in
        # order. Any future divergence (compression, undo, repair, steer)
        # breaks identity at the divergent index and re-scans from there.
        cursor["prefix"] = messages[:]

    return repaired

def strip_think_blocks(agent, content: str) -> str:
    """Remove reasoning/thinking blocks from content, returning only visible text.

    Handles four cases:
      1. Closed tag pairs (`` <think>… ``) — the common path when
         the provider emits complete reasoning blocks.
      2. Unterminated open tag at a block boundary (start of text or
         after a newline) — e.g. MiniMax M2.7 / NIM endpoints where the
         closing tag is dropped.  Everything from the open tag to end
         of string is stripped.  The block-boundary check mirrors
         ``gateway/stream_consumer.py``'s filter so models that mention
         `` <think>`` in prose aren't over-stripped.
      3. Stray orphan open/close tags that slip through.
      4. Tag variants: `` <think>``, ``<thinking>``, ``<reasoning>``,
         ``<REASONING_SCRATCHPAD>``, ``<thought>`` (Gemma 4), all
         case-insensitive.

    Additionally strips standalone tool-call XML blocks that some open
    models (notably Gemma variants on OpenRouter) emit inside assistant
    content instead of via the structured ``tool_calls`` field:
      * ``<tool_call>…</tool_call>``
      * ``<tool_calls>…</tool_calls>``
      * ``<tool_result>…</tool_result>``
      * ``<function_call>…</function_call>``
      * ``<function_calls>…</function_calls>``
      * ``<function name="…">…</function>`` (Gemma style)
    Ported from openclaw/openclaw#67318. The ``<function>`` variant is
    boundary-gated (only strips when the tag sits at start-of-line or
    after punctuation and carries a ``name="..."`` attribute) so prose
    mentions like "Use <function> in JavaScript" are preserved.
    """
    if not content:
        return ""
    # Coerce non-string content to text before any regex runs.  Providers
    # that return assistant ``content`` as a list of blocks (Anthropic via
    # OpenRouter emits ``[{"type":"text",...}, {"type":"thinking",...}]``) or
    # as a dict flow into this shared helper from several callers — most
    # notably ``_interim_assistant_visible_text`` reading a *stored* history
    # message whose content was persisted as a list.  A raw list/dict reaching
    # ``re.sub`` below raises ``TypeError: expected string or bytes-like
    # object, got 'list'``, which the outer conversation loop swallows and
    # retries forever (observed as an infinite "preparing terminal…" loop on
    # Anthropic models via OpenRouter).  Flatten here so every caller is safe.
    if not isinstance(content, str):
        if isinstance(content, list):
            _parts: list[str] = []
            for _part in content:
                if isinstance(_part, str):
                    _parts.append(_part)
                elif isinstance(_part, dict):
                    _ptype = str(_part.get("type") or "").strip().lower()
                    # Drop reasoning/thinking blocks outright — this function's
                    # whole job is to strip them, and their text lives under
                    # different keys ("thinking", "reasoning") per provider.
                    if _ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    _text = _part.get("text")
                    if isinstance(_text, str) and _text:
                        _parts.append(_text)
            content = "".join(_parts)
        elif isinstance(content, dict):
            content = str(content.get("text") or content.get("content") or "")
        else:
            content = str(content)
        if not content:
            return ""
    # 1. Closed tag pairs — case-insensitive for all variants so
    #    mixed-case tags (<THINK>, <Thinking>) don't slip through to
    #    the unterminated-tag pass and take trailing content with them.
    for _pattern in _REASONING_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1b. Tool-call XML blocks (openclaw/openclaw#67318). Handle the
    #     generic tag names first — they have no attribute gating since
    #     a literal <tool_call> in prose is already vanishingly rare.
    for _pattern in _TOOL_CALL_BLOCK_PATTERNS:
        content = _pattern.sub('', content)
    # 1c. <function name="...">...</function> — Gemma-style standalone
    #     tool call. Only strip when the tag sits at a block boundary
    #     (start of text, after a newline, or after sentence-ending
    #     punctuation) AND carries a name="..." attribute. This keeps
    #     prose mentions like "Use <function> to declare" safe.
    content = _NAMED_FUNCTION_BLOCK_PATTERN.sub('', content)
    # 2. Unterminated reasoning block — open tag at a block boundary
    #    (start of text, or after a newline) with no matching close.
    #    Strip from the tag to end of string.  Fixes #8878 / #9568
    #    (MiniMax M2.7 leaking raw reasoning into assistant content).
    content = _UNTERMINATED_REASONING_BLOCK_PATTERN.sub('', content)
    # 3. Stray orphan open/close tags that slipped through.
    content = _ORPHAN_REASONING_TAG_PATTERN.sub('', content)
    # 3b. Stray tool-call closers. (We do NOT strip bare <function> or
    #     unterminated <function name="..."> because a truncated tail
    #     during streaming may still be valuable to the user; matches
    #     OpenClaw's intentional asymmetry.)
    content = _STRAY_TOOL_CALL_CLOSER_PATTERN.sub('', content)
    return content

def drop_thinking_only_and_merge_users(
    messages: List[Dict[str, Any]],
    *,
    drop_codex_reasoning_items: bool = True,
) -> List[Dict[str, Any]]:
    """Drop thinking-only assistant turns; merge any adjacent user messages left behind.

    Runs on the per-call ``api_messages`` copy only. The stored
    conversation history (``agent.messages``) is never mutated, so the
    user still sees the thinking block in the CLI/gateway transcript and
    session persistence keeps the full trace. Only the wire copy sent to
    the provider is cleaned.

    Why drop-and-merge rather than inject stub text:
    - Fabricating ``"."`` / ``"(continued)"`` text lies in the history
      and makes future turns see model output the model didn't emit.
    - Dropping the turn preserves honesty; merging adjacent user messages
      preserves the provider's role-alternation invariant.
    - This is the pattern used by Claude Code's ``normalizeMessagesForAPI``
      (filterOrphanedThinkingOnlyMessages + mergeAdjacentUserMessages).
    """
    if not messages:
        return messages

    # Pass 1: drop thinking-only assistant turns.
    kept = [
        m for m in messages
        if not is_thinking_only_assistant(
            m,
            drop_codex_reasoning_items=drop_codex_reasoning_items,
        )
    ]
    dropped = len(messages) - len(kept)
    if dropped == 0:
        return messages

    # Pass 2: merge any newly-adjacent user messages.
    merged: List[Dict[str, Any]] = []
    merges = 0
    for m in kept:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.get("role") == "user"
            and m.get("role") == "user"
        ):
            prev_content = prev.get("content", "")
            cur_content = m.get("content", "")
            # Work on a copy of ``prev`` so the caller's input dicts are
            # never mutated. ``_sanitize_api_messages`` upstream already
            # hands us per-call copies, but staying pure here means we
            # can be called safely from anywhere (tests, other loops).
            prev_copy = dict(prev)
            # Only string-content merge is meaningful for role-alternation
            # purposes. If either side is a list (multimodal), append as a
            # separate block rather than collapsing.
            if isinstance(prev_content, str) and isinstance(cur_content, str):
                sep = "\n\n" if prev_content and cur_content else ""
                prev_copy["content"] = prev_content + sep + cur_content
            elif isinstance(prev_content, list) and isinstance(cur_content, list):
                prev_copy["content"] = list(prev_content) + list(cur_content)
            elif isinstance(prev_content, list) and isinstance(cur_content, str):
                if cur_content:
                    prev_copy["content"] = list(prev_content) + [
                        {"type": "text", "text": cur_content}
                    ]
                else:
                    prev_copy["content"] = list(prev_content)
            elif isinstance(prev_content, str) and isinstance(cur_content, list):
                new_blocks: List[Dict[str, Any]] = []
                if prev_content:
                    new_blocks.append({"type": "text", "text": prev_content})
                new_blocks.extend(cur_content)
                prev_copy["content"] = new_blocks
            else:
                # Unknown content shape — fall back to appending separately
                # (violates alternation, but safer than raising in a hot path).
                merged.append(m)
                continue
            merged[-1] = prev_copy
            merges += 1
        else:
            merged.append(m)

    logger.debug(
        "Pre-call sanitizer: dropped %d thinking-only assistant turn(s), "
        "merged %d adjacent user message(s)",
        dropped,
        merges,
    )
    return merged

def repair_tool_call(agent, tool_name: str) -> str | None:
    """Attempt to repair a mismatched tool name before aborting.

    Models sometimes emit variants of a tool name that differ only
    in casing, separators, or class-like suffixes. Normalize
    aggressively before falling back to fuzzy match:

    1. Lowercase direct match.
    2. Lowercase + hyphens/spaces -> underscores.
    3. CamelCase -> snake_case (TodoTool -> todo_tool).
    4. Strip trailing ``_tool`` / ``-tool`` / ``tool`` suffix that
       Claude-style models sometimes tack on (TodoTool_tool ->
       TodoTool -> Todo -> todo). Applied twice so double-tacked
       suffixes like ``TodoTool_tool`` reduce all the way.
    5. Fuzzy match (difflib, cutoff=0.7).

    See #14784 for the original reports (TodoTool_tool, Patch_tool,
    BrowserClick_tool were all returning "Unknown tool" before).

    Returns the repaired name if found in valid_tool_names, else None.
    """
    import re
    from difflib import get_close_matches

    if not tool_name:
        return None

    # VolcEngine api/plan workaround (issue #33007): the endpoint's
    # protocol-translation layer occasionally leaks raw XML attribute
    # fragments into tool_use.name, e.g.
    #   `terminal" parameter="command" string="true`
    #   `execute_code" parameter="code" string="true`
    #   `session_search" parameter="session_id" string="true`
    # We trim at the first unambiguous XML/quote character so the rest
    # of the repair pipeline (lowercase / snake_case / fuzzy match)
    # can resolve the cleaned name to a real tool.
    #
    # Crucially we DO NOT split on whitespace: legitimate inputs like
    # "write file" must keep flowing through ``_norm`` -> ``write_file``
    # (covered by test_space_to_underscore in
    # tests/run_agent/test_repair_tool_call_name.py).
    for _xml_sep in ('"', "'", "<", ">"):
        _idx = tool_name.find(_xml_sep)
        if _idx > 0:
            tool_name = tool_name[:_idx]
    if not tool_name:
        return None

    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "_")

    def _camel_snake(s: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    def _strip_tool_suffix(s: str) -> str | None:
        lc = s.lower()
        for suffix in ("_tool", "-tool", "tool"):
            if lc.endswith(suffix):
                return s[: -len(suffix)].rstrip("_-")
        return None

    # Cheap fast-paths first — these cover the common case.
    lowered = tool_name.lower()
    if lowered in agent.valid_tool_names:
        return lowered
    normalized = _norm(tool_name)
    if normalized in agent.valid_tool_names:
        return normalized

    # Build the full candidate set for class-like emissions.
    cands: set[str] = {tool_name, lowered, normalized, _camel_snake(tool_name)}
    # Strip trailing tool-suffix up to twice — TodoTool_tool needs it.
    for _ in range(2):
        extra: set[str] = set()
        for c in cands:
            stripped = _strip_tool_suffix(c)
            if stripped:
                extra.add(stripped)
                extra.add(_norm(stripped))
                extra.add(_camel_snake(stripped))
        cands |= extra

    for c in cands:
        if c and c in agent.valid_tool_names:
            return c

    # Fuzzy match as last resort.
    matches = get_close_matches(lowered, agent.valid_tool_names, n=1, cutoff=0.7)
    if matches:
        return matches[0]

    return None

def _msg_has_payload(msg: Dict[str, Any]) -> bool:
    """True if ``msg`` carries anything the API treats as non-empty content.

    Covers string content, non-empty multimodal content lists, tool_calls,
    tool_call_id linkage (tool results), and reasoning payloads. Mirrors the
    emptiness checks used by ``create_agent._is_thinking_only_assistant`` but is
    role-agnostic so it can vet user/assistant/tool turns uniformly.
    """
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            return True
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                # any typed block (text/image/tool_use/document/...) counts,
                # as long as a text block is not itself blank
                if block.get("type") == "text":
                    if isinstance(block.get("text"), str) and block["text"].strip():
                        return True
                    continue
                return True
            elif block:
                return True
    elif content not in (None, ""):
        return True
    # Structural payloads that make an "empty-content" message still valid.
    if msg.get("tool_calls"):
        return True
    if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"].strip():
        return True
    if msg.get("reasoning") or msg.get("reasoning_details"):
        return True
    # Codex Responses item carriers: a commentary-phase assistant turn
    # persists with content:"" by DESIGN — its text lives in
    # ``codex_message_items`` (delivered via the interim callback) and the
    # structured items are replayed for prefix-cache hits.  Same for
    # ``codex_reasoning_items``.  These turns are never wire-empty on any
    # api_mode: the codex transport replays the items, and the
    # chat-completions transport strips the carriers only after this repair
    # pass has already run.  Treat them as payload so the repair never
    # rewrites a designed-empty codex turn (July 2026: a write-time pad that
    # ignored this broke codex commentary replay in CI).
    if msg.get("codex_message_items") or msg.get("codex_reasoning_items"):
        return True
    return False

def repair_empty_non_final_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Heal empty-content non-final messages before they reach the provider.

    Root-cause context: a stream that dies with 0 recovered characters (peer
    reset, stall-kill) could persist an assistant turn with ``content=None``
    and no tool_calls. The Anthropic message schema — and the litellm/Bedrock
    proxies in front of it — reject ANY request whose transcript contains an
    empty non-final message:

        "all messages must have non-empty content except for the optional
         final assistant message"  (HTTP 400 INVALID_REQUEST_BODY)

    Once such a message lands mid-transcript it poisons EVERY subsequent turn
    of that session until it scrolls out of context. The write-time guard in
    ``chat_completion_helpers`` stops NEW stubs, but sessions already carrying
    one (persisted before the guard, or fed in from a host history) stay stuck
    and previously needed a manual DB edit + gateway restart to recover.

    This pass is the self-healing counterpart: it runs unconditionally on the
    per-call ``api_messages`` copy, so a poisoned transcript repairs itself
    IN MEMORY on the very next send — no restart, no DB surgery. The final
    message is left untouched (an empty final assistant turn is legal). The
    stored conversation history is never mutated; only the wire copy is
    repaired, so the UI/session trace stays faithful.

    Repair strategy is substitution, not deletion: dropping a mid-transcript
    turn can break role alternation and tool-call pairing, whereas an honest
    minimal placeholder keeps the sequence intact and reads correctly as an
    interrupted turn on replay.
    """
    if not messages or len(messages) < 2:
        return messages

    repaired: List[Dict[str, Any]] = []
    healed = 0
    last_idx = len(messages) - 1
    for idx, msg in enumerate(messages):
        if (
            idx != last_idx
            and isinstance(msg, dict)
            # tool results are validated by their own orphan/pairing pass; an
            # empty tool result is a separate (and rarer) concern.
            and msg.get("role") in ("assistant", "user")
            and not _msg_has_payload(msg)
        ):
            # Shallow-copy so stored history / prompt caching stays byte-stable.
            fixed = dict(msg)
            fixed["content"] = _INTERRUPTED_PLACEHOLDER
            repaired.append(fixed)
            healed += 1
        else:
            repaired.append(msg)

    if healed:
        logger.warning(
            "Pre-call sanitizer: healed %d empty non-final message(s) by "
            "substituting placeholder content — an empty-content turn was in "
            "the transcript and would 400 the request ('messages must have "
            "non-empty content' / INVALID_REQUEST_BODY). Self-recovering the "
            "poisoned transcript in memory; no restart needed.",
            healed,
        )
        return repaired
    return messages

def sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix orphaned tool_call / tool_result pairs before every LLM call.

    Runs unconditionally — not gated on whether the context compressor
    is present — so orphans from session loading or manual message
    manipulation are always caught.
    """
    # --- Role allowlist: drop messages with roles the API won't accept ---
    filtered = []
    for msg in messages:
        role = msg.get("role")
        if role not in VALID_API_ROLES:
            logger.debug(
                "Pre-call sanitizer: dropping message with invalid role %r",
                role,
            )
            continue
        filtered.append(msg)
    messages = filtered

    # --- Heal empty-content non-final messages (self-recovery) ---
    # A dead stream can leave an empty assistant stub (or an empty user turn)
    # mid-transcript; the provider then 400s EVERY subsequent request until it
    # scrolls out. Repair it here, on the per-call copy, so a poisoned session
    # recovers itself in memory on the next send — no restart, no DB edit.
    # Done first so a substituted turn participates normally in the tool-pair
    # and dedup passes below.
    messages = repair_empty_non_final_messages(messages)

    # --- Drop empty / malformed tool_calls arrays on assistant messages ---
    # An assistant message carrying ``tool_calls: []`` (an empty array) — or a
    # non-list value under the key — is semantically identical to an assistant
    # message with no tool calls, but strict OpenAI-compatible providers reject
    # the empty array outright: DeepSeek v4 returns HTTP 400 "Invalid
    # 'messages[N].tool_calls': empty array. Expected an array with minimum
    # length 1, but got an empty array instead." (#58755, follow-up to #56980).
    # Empty arrays reach here from session resume, host-fed histories, or the
    # consecutive-assistant merge in ``repair_message_sequence`` (which
    # preserves a pre-existing ``[]`` on the surviving turn). This is the final
    # pre-API chokepoint, so normalize defensively — and, per the #56980
    # review, do it HERE on the per-call copy rather than in
    # ``repair_message_sequence``, which would destructively rewrite the
    # persisted trajectory. Shallow-copy the message before dropping the key so
    # stored history (and prompt caching) stays byte-stable.
    normalized: List[Dict[str, Any]] = []
    dropped_empty_tool_calls = 0
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and "tool_calls" in msg
            and not (isinstance(msg["tool_calls"], list) and msg["tool_calls"])
        ):
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            dropped_empty_tool_calls += 1
        normalized.append(msg)
    if dropped_empty_tool_calls:
        messages = normalized
        logger.debug(
            "Pre-call sanitizer: dropped empty/invalid tool_calls on %d "
            "assistant message(s)",
            dropped_empty_tool_calls,
        )

    # --- Repair tool_calls whose function.name is empty/missing ---
    # Some providers (and partially-streamed responses) emit a tool_call with
    # id="call_xxx" but function.name="". Downstream Responses-API adapters
    # silently DROP such function_call items while still emitting the matching
    # function_call_output, producing the gateway's HTTP 400
    # "No tool call found for function call output with call_id ...".
    #
    # We do NOT drop the call: hermes' own dispatch loop intentionally keeps an
    # empty-name call paired with a synthesized anti-priming tool result
    # ("tool name was empty", see #47967) so weak models self-correct instead of
    # being fed the full tool catalog. Dropping the call here would (a) orphan
    # that result and strip the anti-priming signal, and (b) still leave any
    # provider-side orphan. Instead, rename the blank name to a non-empty
    # sentinel so the call and its result stay PAIRED — the adapter no longer
    # drops the function_call, so there is no orphaned output and no 400, while
    # the result content the model needs is preserved.
    _EMPTY_NAME_SENTINEL = "invalid_tool_call"
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls") or []
        if not tcs:
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                fn = tc.get("function")
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            else:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) if fn else None
            if isinstance(name, str) and name.strip():
                continue
            logger.warning(
                "Pre-call sanitizer: repairing tool_call with empty "
                "function.name -> %r (id=%s)",
                _EMPTY_NAME_SENTINEL,
                coalesce_tool_call_id(tc),
            )
            if isinstance(fn, dict):
                fn["name"] = _EMPTY_NAME_SENTINEL
            elif fn is not None and hasattr(fn, "name"):
                try:
                    fn.name = _EMPTY_NAME_SENTINEL
                except Exception:
                    pass
            elif isinstance(tc, dict):
                tc["function"] = {"name": _EMPTY_NAME_SENTINEL, "arguments": "{}"}

    surviving_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                cid = coalesce_tool_call_id(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = (msg.get("tool_call_id") or "").strip()
            if cid:
                result_call_ids.add(cid)

    # 1. Drop tool results with no matching assistant call
    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and (m.get("tool_call_id") or "").strip() in orphaned_results)
        ]
        logger.debug(
            "Pre-call sanitizer: removed %d orphaned tool result(s)",
            len(orphaned_results),
        )

    # 2. Inject stub results for calls whose result was dropped
    missing_results = surviving_call_ids - result_call_ids
    if missing_results:
        patched: List[Dict[str, Any]] = []
        for msg in messages:
            patched.append(msg)
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = coalesce_tool_call_id(tc)
                    if cid in missing_results:
                        patched.append({
                            "role": "tool",
                            "name": get_tool_call_name(tc),
                            "content": "[Result unavailable — see context summary above]",
                            "tool_call_id": cid,
                        })
        messages = patched
        logger.debug(
            "Pre-call sanitizer: added %d stub tool result(s)",
            len(missing_results),
        )

    # 3. Deduplicate tool_call_ids. Strict providers (DeepSeek) reject a
    # payload where the same tool_call_id appears more than once with HTTP 400
    # "Duplicate value for 'tool_call_id'" (#58327). Duplicates can arise from
    # retries, crash/resume glitches, or a compression window that re-emits a
    # tool result. This is the final pre-API chokepoint, so dedup defensively
    # here even though repair_message_sequence also consumes matched ids.
    #   (a) collapse duplicate tool_calls WITHIN an assistant message
    #   (b) drop tool results that answer no OUTSTANDING tool call
    #
    # (b) tracks outstanding calls rather than every id ever seen, because
    # ``tool_call_id`` is NOT globally unique in practice: llama.cpp emits a
    # single constant id for every tool call it ever returns (verified: three
    # separate completions from one server all carry the same id). A
    # seen-once-drop-forever rule reads the SECOND legitimate tool result of
    # such a session as a duplicate and deletes it, so from the second tool
    # call onward the model never sees any result — it announces its next
    # action and the turn dies with the work unfinished. Outstanding-call
    # semantics keep both protections intact: a re-emitted result still
    # answers no pending call and is still dropped, while a genuine new call
    # that reuses the id re-arms that id first.
    seen_assistant_call_ids: set = set()
    outstanding_call_ids: set = set()
    deduped: List[Dict[str, Any]] = []
    removed_dupes = 0
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            kept_tcs = []
            for tc in msg.get("tool_calls") or []:
                cid = coalesce_tool_call_id(tc)
                if cid and cid in seen_assistant_call_ids:
                    removed_dupes += 1
                    continue
                if cid:
                    seen_assistant_call_ids.add(cid)
                    outstanding_call_ids.add(cid)
                kept_tcs.append(tc)
            if kept_tcs:
                msg = {**msg, "tool_calls": kept_tcs}
            elif len(kept_tcs) != len(msg.get("tool_calls") or []):
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            deduped.append(msg)
        elif role == "tool":
            cid = (msg.get("tool_call_id") or "").strip()
            if cid and cid not in outstanding_call_ids:
                removed_dupes += 1
                continue
            if cid:
                # Answered: this id is no longer outstanding, so a second
                # result replaying it is still caught above.
                outstanding_call_ids.discard(cid)
                # A reused id must be re-armable by the next assistant call.
                seen_assistant_call_ids.discard(cid)
            deduped.append(msg)
        else:
            deduped.append(msg)
    if removed_dupes:
        messages = deduped
        logger.debug(
            "Pre-call sanitizer: removed %d duplicate tool_call_id reference(s)",
            removed_dupes,
        )
    return messages
