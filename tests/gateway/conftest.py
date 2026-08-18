"""Shared fixtures for retained gateway tests."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def make_async_session_db(sync_mock=None):
    from hermes_state import AsyncSessionDB

    sync_mock = sync_mock if sync_mock is not None else MagicMock()
    return AsyncSessionDB(sync_mock), sync_mock


class _FakeEnumMember(str):
    def __new__(cls, enum_name: str, member_name: str, value: str):
        obj = str.__new__(cls, value)
        obj._qualname = f"{enum_name}.{member_name}"
        return obj

    def __repr__(self) -> str:
        return f"<{self._qualname}: {str.__repr__(self)}>"


def _fake_str_enum(enum_name: str, **members: str):
    return SimpleNamespace(
        **{
            name: _FakeEnumMember(enum_name, name, value)
            for name, value in members.items()
        }
    )


def _ensure_telegram_mock() -> None:
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    module = MagicMock()
    module.ext.ContextTypes.DEFAULT_TYPE = type(None)
    parse_mode = _fake_str_enum(
        "ParseMode", MARKDOWN="Markdown", MARKDOWN_V2="MarkdownV2", HTML="HTML"
    )
    chat_type = _fake_str_enum(
        "ChatType",
        PRIVATE="private",
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
    )
    module.ParseMode = parse_mode
    module.constants.ParseMode = parse_mode
    module.ChatType = chat_type
    module.constants.ChatType = chat_type
    module.error.TelegramError = type("TelegramError", (Exception,), {})
    module.error.NetworkError = type(
        "NetworkError", (module.error.TelegramError,), {}
    )
    module.error.TimedOut = type("TimedOut", (module.error.NetworkError,), {})
    module.error.BadRequest = type("BadRequest", (module.error.NetworkError,), {})
    module.error.Forbidden = type("Forbidden", (module.error.TelegramError,), {})
    module.error.InvalidToken = type("InvalidToken", (module.error.TelegramError,), {})

    class RetryAfter(module.error.TelegramError):
        def __init__(self, retry_after=1):
            self.retry_after = retry_after

    module.error.RetryAfter = RetryAfter
    module.error.Conflict = type("Conflict", (module.error.TelegramError,), {})
    module.Update.ALL_TYPES = []
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules[name] = module
    sys.modules["telegram.error"] = module.error


_ensure_telegram_mock()
