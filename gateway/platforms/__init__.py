"""Retained messaging platform adapters."""

from .base import BasePlatformAdapter, MessageEvent, SendResult
from .mattermost import MattermostAdapter
from .telegram.adapter import TelegramAdapter

__all__ = [
    "BasePlatformAdapter",
    "MattermostAdapter",
    "MessageEvent",
    "SendResult",
    "TelegramAdapter",
]
