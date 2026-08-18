"""Resolve gateway ``terminal.cwd`` placeholder values to ``TERMINAL_CWD``.

When ``terminal.cwd`` is unset or a placeholder (``.``, ``auto``, ``cwd``),
the gateway must not blindly map host ``Path.home()`` into container backends.
Docker with workspace mounting requires an absolute ``terminal.cwd``; a
placeholder leaves the backend on its sandbox default.
"""

from __future__ import annotations

CWD_PLACEHOLDERS = frozenset({".", "auto", "cwd"})


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def resolve_placeholder_terminal_cwd(
    *,
    configured_cwd: str,
    terminal_backend: str,
    home_fallback: str,
) -> str | None:
    """Return the ``TERMINAL_CWD`` value to set, or ``None`` to leave it unset.

    Cases:
      - **local** + placeholder → ``home_fallback``
      - **docker** + placeholder + mount on → ``None`` until an absolute
        ``terminal.cwd`` is configured
      - **docker** + placeholder + mount off → ``None`` (sandbox default)
      - other non-local backends + placeholder → ``None``
    """
    if configured_cwd and configured_cwd not in CWD_PLACEHOLDERS:
        return configured_cwd

    backend = (terminal_backend or "local").strip().lower()
    if backend == "local":
        return home_fallback

    return None
