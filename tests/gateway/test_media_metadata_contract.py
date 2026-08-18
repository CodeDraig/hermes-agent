"""Retained adapters accept shared media-delivery metadata."""

import inspect

import pytest

from gateway.platforms.mattermost import MattermostAdapter
from gateway.platforms.telegram.adapter import TelegramAdapter


def _accepts_metadata(method) -> bool:
    params = inspect.signature(method).parameters
    return "metadata" in params or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )


@pytest.mark.parametrize("adapter_cls", [TelegramAdapter, MattermostAdapter])
def test_send_image_accepts_metadata(adapter_cls):
    assert _accepts_metadata(adapter_cls.send_image)


@pytest.mark.parametrize("adapter_cls", [TelegramAdapter, MattermostAdapter])
def test_edit_message_accepts_finalize(adapter_cls):
    assert "finalize" in inspect.signature(adapter_cls.edit_message).parameters
