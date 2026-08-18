"""Process-local ownership of the active gateway runner."""

from __future__ import annotations

import weakref
from typing import Any


_active_runner: weakref.ReferenceType[Any] | None = None


def register_runner(runner: Any) -> None:
    """Register *runner* without extending its lifetime."""
    global _active_runner
    _active_runner = weakref.ref(runner)


def get_runner() -> Any | None:
    """Return the active gateway runner when it is still alive."""
    return _active_runner() if _active_runner is not None else None


def clear_runner(runner: Any) -> None:
    """Clear the registry only when *runner* still owns it."""
    global _active_runner
    if get_runner() is runner:
        _active_runner = None
