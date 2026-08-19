"""Messaging platform metadata used by CLI configuration displays."""

from collections import OrderedDict
from typing import NamedTuple


class PlatformInfo(NamedTuple):
    label: str
    default_toolset: str


PLATFORMS: OrderedDict[str, PlatformInfo] = OrderedDict([
    ("cli", PlatformInfo(label="🖥️  CLI", default_toolset="hermes-cli")),
    ("telegram", PlatformInfo(label="📱 Telegram", default_toolset="hermes-telegram")),
    ("mattermost", PlatformInfo(label="💬 Mattermost", default_toolset="hermes-mattermost")),
    ("cron", PlatformInfo(label="⏰ Cron", default_toolset="hermes-cron")),
])


def platform_label(key: str, default: str = "") -> str:
    info = PLATFORMS.get(key)
    return info.label if info is not None else default


def get_all_platforms() -> "OrderedDict[str, PlatformInfo]":
    return OrderedDict(PLATFORMS)
