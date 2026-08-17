"""Prompt-toolkit voice shortcut normalization and display helpers."""

from __future__ import annotations

import sys
from typing import Any

# ``super``/``win``/``windows`` are intentionally absent: prompt_toolkit
# has no super/meta modifier for the Cmd key. The normalizer returns the
# documented default (``c-b``) for those spellings.
_VOICE_MOD_ALIASES = {
    "ctrl": "c-",
    "control": "c-",
    "alt": "a-",
    "option": "a-",
    "opt": "a-",
}

# Named keys prompt_toolkit accepts in ``c-<name>`` / ``a-<name>`` form.
# Aliases collapse to prompt_toolkit's canonical spelling so the same
# config value binds identically in both runtimes (Copilot round-10 on
# #19835).
_VOICE_NAMED_KEYS = {
    "space": "space",
    "spc": "space",
    "enter": "enter",
    "return": "enter",
    "ret": "enter",
    "tab": "tab",
    "escape": "escape",
    "esc": "escape",
    "backspace": "backspace",
    "bs": "backspace",
    "delete": "delete",
    "del": "delete",
}

# The CLI reserves these chords for interrupt, quit, and clear-screen.
_VOICE_RESERVED_CTRL_CHARS = frozenset({"c", "d", "l"})

# On macOS the classic CLI's prompt_toolkit bindings for copy / exit /
# clear also claim ``a-c`` / ``a-d`` / ``a-l`` via the action-modifier
# lookup, so reserve those shortcuts too.
_VOICE_RESERVED_ALT_CHARS_MAC = frozenset({"c", "d", "l"})

_DEFAULT_PT_KEY = "c-b"


def voice_record_key_from_config(cfg: Any) -> Any:
    """Shape-safe ``cfg.voice.record_key`` lookup.

    ``load_config()`` deep-merges raw YAML and preserves scalar
    overrides, so a hand-edited ``voice: true`` / ``voice: cmd+b``
    leaves ``cfg["voice"]`` as a bool/str instead of a dict, and the
    naive ``.get("voice", {}).get("record_key")`` chain raises
    AttributeError before voice can even start (Copilot round-11 on
    #19835). Return ``None`` for malformed shapes so call sites can
    feed the result straight into the normalizer/formatter and get
    the documented default.
    """
    if not isinstance(cfg, dict):
        return None

    voice = cfg.get("voice")
    if not isinstance(voice, dict):
        return None

    return voice.get("record_key")


def normalize_voice_record_key_for_prompt_toolkit(raw: Any) -> str:
    """Coerce ``voice.record_key`` into prompt_toolkit's ``c-x`` / ``a-x`` format.

    * non-string / empty / typo'd / bare-char / multi-modifier / reserved
      ``ctrl+c|d|l`` → documented default ``c-b``
    * single-char keys: ``ctrl+o`` → ``c-o``
    * named keys: ``ctrl+space`` → ``c-space`` (aliases collapse:
      ``ctrl+return`` → ``c-enter``)
    * ``super`` / ``win`` / ``windows`` → ``c-b`` because prompt_toolkit has
      no super modifier
    """
    if not isinstance(raw, str):
        return _DEFAULT_PT_KEY

    lowered = raw.strip().lower()
    if not lowered:
        return _DEFAULT_PT_KEY

    parts = [p.strip() for p in lowered.split("+") if p.strip()]
    if not parts:
        return _DEFAULT_PT_KEY

    # Multi-modifier chords like ``ctrl+alt+r`` bind different shortcuts
    # in prompt_toolkit (a-c-r form); collapse to the documented default.
    if len(parts) > 2:
        return _DEFAULT_PT_KEY

    # Bare char / bare named key (no explicit modifier) — the CLI's
    # prompt_toolkit binds the raw key without a modifier; require an explicit
    # modifier so configuration stays predictable.
    if len(parts) == 1:
        return _DEFAULT_PT_KEY

    modifier_token, key_token = parts

    # prompt_toolkit has no super modifier, so fall back to the default.
    if modifier_token in {"super", "win", "windows"}:
        return _DEFAULT_PT_KEY

    normalized_mod = _VOICE_MOD_ALIASES.get(modifier_token)
    if not normalized_mod:
        return _DEFAULT_PT_KEY

    # Single-char key: reject reserved CLI chords.
    if len(key_token) == 1:
        if normalized_mod == "c-" and key_token in _VOICE_RESERVED_CTRL_CHARS:
            return _DEFAULT_PT_KEY
        if (
            normalized_mod == "a-"
            and sys.platform == "darwin"
            and key_token in _VOICE_RESERVED_ALT_CHARS_MAC
        ):
            return _DEFAULT_PT_KEY
        return f"{normalized_mod}{key_token}"

    # Multi-char key token must be a known named key; typos like
    # ``ctrl+spcae`` fall back to the default rather than being passed
    # through as ``c-spcae`` (which prompt_toolkit would reject).
    named = _VOICE_NAMED_KEYS.get(key_token)
    if not named:
        return _DEFAULT_PT_KEY

    return f"{normalized_mod}{named}"


def pt_key_to_sequence(pt_key: str) -> tuple[str, ...]:
    """Convert a prompt_toolkit key specifier (e.g. 'c-b' or 'a-v') to a sequence tuple.

    prompt_toolkit's ``@kb.add`` rejects 'a-x' strings directly (raises ValueError),
    expecting ('escape', 'x') instead for Alt-modifier shortcuts.
    """
    if isinstance(pt_key, str) and pt_key.startswith("a-"):
        return ("escape", pt_key[2:])
    return (pt_key,)


def format_voice_record_key_for_status(raw: Any) -> str:
    """Render ``voice.record_key`` for ``/voice status`` in CLI-friendly form.

    Returns ``Ctrl+B`` / ``Alt+Space`` / ``Ctrl+Enter``. Malformed configs surface as the
    documented default so status never advertises a shortcut that
    won't bind (Copilot round-10 on #19835).
    """
    normalized = normalize_voice_record_key_for_prompt_toolkit(raw)

    if normalized.startswith("c-"):
        prefix, key = "Ctrl+", normalized[2:]
    elif normalized.startswith("a-"):
        prefix, key = "Alt+", normalized[2:]
    elif "+" in normalized:
        # ``super+<key>`` / ``win+<key>`` — CLI won't bind them, but
        # render in title case so status output is still readable.
        mod, key = normalized.split("+", 1)
        prefix = mod[0].upper() + mod[1:] + "+"
    else:
        return "Ctrl+B"

    if not key:
        return prefix.rstrip("+")

    if len(key) == 1:
        return prefix + key.upper()

    return prefix + key[0].upper() + key[1:]
