"""Outbound delivery for the retained Telegram and Mattermost gateway transports."""

import asyncio
import json
import logging
import os
import re

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_TELEGRAM_TOPIC_TARGET_RE = re.compile(r"^\s*(-?\d+)(?::(\d+))?\s*$")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".m2a", ".wav", ".m4a", ".flac"}
_VOICE_EXTS = {".ogg", ".opus"}
_TELEGRAM_SEND_AUDIO_EXTS = {".mp3", ".m4a"}
_CAPTIONABLE_EXTS = _IMAGE_EXTS | _VIDEO_EXTS | {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".csv", ".xlsx", ".zip",
}
_TELEGRAM_CAPTION_LIMIT = 1024
_URL_SECRET_QUERY_RE = re.compile(
    r"([?&](?:access_token|api[_-]?key|auth[_-]?token|token|signature|sig)=)([^&#\\s]+)",
    re.IGNORECASE,
)
_GENERIC_SECRET_ASSIGN_RE = re.compile(
    r"\\b(access_token|api[_-]?key|auth[_-]?token|signature|sig)\\s*=\\s*([^\\s,;]+)",
    re.IGNORECASE,
)


def _media_caption_split(text, media_files, *, max_caption_len):
    """Return a native media caption when one captionable attachment can carry it."""
    stripped = (text or "").strip()
    media = media_files or []
    if not stripped or len(media) != 1:
        return None, text
    media_path, is_voice = media[0]
    if is_voice or os.path.splitext(media_path)[1].lower() not in _CAPTIONABLE_EXTS:
        return None, text
    if len(stripped) > max_caption_len:
        return None, text
    return stripped, ""


def _sanitize_error_text(text) -> str:
    redacted = redact_sensitive_text(str(text))
    redacted = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}***", redacted)
    return _GENERIC_SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=***", redacted)


def _error(message: str) -> dict:
    return {"error": _sanitize_error_text(message)}


def _telegram_retry_delay(exc: Exception, attempt: int) -> float | None:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return 1.0
    text = str(exc).lower()
    if "timed out" in text or "timeout" in text:
        return None
    if any(marker in text for marker in (
        "bad gateway", "502", "too many requests", "429",
        "service unavailable", "503", "gateway timeout", "504",
    )):
        return float(2 ** attempt)
    return None


async def _send_telegram_message_with_retry(bot, *, attempts: int = 3, **kwargs):
    for attempt in range(attempts):
        try:
            return await bot.send_message(**kwargs)
        except Exception as exc:
            delay = _telegram_retry_delay(exc, attempt)
            if delay is None or attempt >= attempts - 1:
                raise
            logger.warning(
                "Transient Telegram send failure (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, attempts, delay, _sanitize_error_text(exc),
            )
            await asyncio.sleep(delay)


SEND_MESSAGE_SCHEMA = {
    "name": "send_message",
    "description": "Send through Telegram or Mattermost, or list available targets.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["send", "list", "react", "unreact"],
                "description": "Send, list targets, add a reaction, or remove a reaction.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Target as 'telegram', 'telegram:chat_id[:topic_id]', "
                    "'mattermost', or 'mattermost:channel_id'."
                ),
            },
            "message": {
                "type": "string",
                "description": "Text to send. MEDIA:<local_path> attaches a local file.",
            },
            "emoji": {"type": "string", "description": "Reaction emoji."},
            "message_id": {"type": "string", "description": "Message id to react to."},
        },
        "required": [],
    },
}


def send_message_tool(args, **kw):
    action = args.get("action", "send")
    if action == "list":
        return _handle_list()
    if action == "react":
        return _handle_react(args)
    if action == "unreact":
        return _handle_react(args, remove=True)
    return _handle_send(args)


def _handle_list():
    try:
        from gateway.channel_directory import format_directory_for_display
        return json.dumps({"targets": format_directory_for_display()})
    except Exception as exc:
        return json.dumps(_error(f"Failed to load channel directory: {exc}"))


def _load_platform(platform_name):
    from gateway.config import Platform
    try:
        return Platform(platform_name)
    except (ValueError, KeyError):
        return None


def _handle_react(args, remove=False):
    target = args.get("target", "")
    emoji = (args.get("emoji") or "").strip()
    message_id = (args.get("message_id") or "").strip() or None
    if not target or (not remove and not emoji):
        return tool_error("'target' and 'emoji' are required" if not remove else "'target' is required")
    platform_name, _, target_ref = target.partition(":")
    platform_name = platform_name.strip().lower()
    platform = _load_platform(platform_name)
    if platform is None:
        return tool_error(f"Unknown platform: {platform_name}")
    chat_id = None
    if target_ref:
        chat_id, _, error = resolve_send_target(
            platform_name, target_ref.strip(), pass_unresolved_references=True
        )
        if error:
            return tool_error(error)
    from gateway.config import load_gateway_config
    config = load_gateway_config()
    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = home.chat_id
    if not chat_id:
        return tool_error(f"No chat specified and no home channel set for {platform_name}.")
    try:
        from gateway.runtime_registry import get_runner
        runner = get_runner()
    except Exception:
        runner = None
    adapter = runner.adapters.get(platform) if runner is not None else None
    if adapter is None:
        return tool_error(f"Reactions require a live {platform_name} adapter.")
    react_fn = getattr(adapter, "remove_reaction" if remove else "add_reaction", None)
    if not callable(react_fn):
        return tool_error(f"Platform '{platform_name}' does not support message reactions.")
    try:
        from model_tools import _run_async
        kwargs = {"chat_id": chat_id, "message_id": message_id}
        if not remove:
            kwargs["emoji"] = emoji
        result = _run_async(react_fn(**kwargs))
    except Exception as exc:
        return json.dumps(_error(f"Reaction failed: {exc}"))
    return json.dumps(result if isinstance(result, dict) else {"success": bool(result)})


def _handle_send(args):
    target = args.get("target", "")
    message = args.get("message", "")
    if not target or not message:
        return tool_error("Both 'target' and 'message' are required when action='send'")
    platform_name, _, target_ref = target.partition(":")
    platform_name = platform_name.strip().lower()
    platform = _load_platform(platform_name)
    if platform is None or platform_name == "api_server":
        return tool_error(f"Unknown delivery platform: {platform_name}")
    chat_id = thread_id = None
    if target_ref:
        chat_id, thread_id, error = resolve_send_target(platform_name, target_ref.strip())
        if error:
            return tool_error(error)
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return tool_error("Interrupted")
    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
    except Exception as exc:
        return json.dumps(_error(f"Failed to load gateway config: {exc}"))
    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        return tool_error(f"Platform '{platform_name}' is not configured.")
    from gateway.platforms.base import BasePlatformAdapter
    force_document = "[[as_document]]" in message
    media_files, cleaned_message = BasePlatformAdapter.extract_media(message)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    mirror_text = cleaned_message.strip() or _describe_media_for_mirror(media_files)
    used_home = False
    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = home.chat_id
            used_home = True
    if not chat_id:
        return tool_error(f"No home channel set for {platform_name}; specify a target or configure one.")
    duplicate = _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id)
    if duplicate:
        return json.dumps(duplicate)
    try:
        from model_tools import _run_async
        result = _run_async(_send_to_platform(
            platform, pconfig, chat_id, cleaned_message,
            thread_id=thread_id, media_files=media_files, force_document=force_document,
        ))
        if used_home and isinstance(result, dict) and result.get("success"):
            result["note"] = f"Sent to {platform_name} home channel (chat_id: {chat_id})"
        if isinstance(result, dict) and result.get("success") and mirror_text:
            try:
                from gateway.mirror import mirror_to_session
                from gateway.session_context import get_session_env
                if mirror_to_session(
                    platform_name, chat_id, mirror_text,
                    source_label=get_session_env("HERMES_SESSION_PLATFORM", "cli"),
                    thread_id=thread_id,
                    user_id=get_session_env("HERMES_SESSION_USER_ID", "") or None,
                ):
                    result["mirrored"] = True
            except Exception:
                pass
        if isinstance(result, dict) and "error" in result:
            result["error"] = _sanitize_error_text(result["error"])
        return json.dumps(result)
    except Exception as exc:
        return json.dumps(_error(f"Send failed: {exc}"))


def _parse_target_ref(platform_name: str, target_ref: str):
    target_ref = target_ref.strip()
    if platform_name == "telegram":
        match = _TELEGRAM_TOPIC_TARGET_RE.fullmatch(target_ref)
        if match:
            return match.group(1), match.group(2), True
        from gateway.platforms.telegram.telegram_ids import parse_telegram_username_target
        username = parse_telegram_username_target(target_ref)
        if username:
            return username, None, True
    if platform_name == "mattermost" and target_ref and not any(c.isspace() for c in target_ref):
        return target_ref, None, True
    return None, None, False


def resolve_send_target(platform_name, target_ref, *, pass_unresolved_references=False):
    platform = _load_platform(platform_name)
    if platform is None or platform_name == "api_server":
        return None, None, f"Unknown platform: {platform_name}"
    chat_id, thread_id, explicit = _parse_target_ref(platform_name, target_ref)
    if explicit:
        return chat_id, thread_id, None
    try:
        from gateway.channel_directory import resolve_channel_name
        resolved = resolve_channel_name(platform_name, target_ref)
    except Exception:
        resolved = None
    if resolved:
        return resolved, None, None
    if pass_unresolved_references:
        return target_ref, None, None
    return None, None, (
        f"Could not resolve '{target_ref}' on {platform_name}. "
        "Use send_message(action='list') to see available targets."
    )


def _describe_media_for_mirror(media_files):
    if not media_files:
        return ""
    if len(media_files) == 1:
        media_path, is_voice = media_files[0]
        ext = os.path.splitext(media_path)[1].lower()
        if is_voice and ext in _VOICE_EXTS:
            return "[Sent voice message]"
        if ext in _IMAGE_EXTS:
            return "[Sent image attachment]"
        if ext in _VIDEO_EXTS:
            return "[Sent video attachment]"
        if ext in _AUDIO_EXTS:
            return "[Sent audio attachment]"
        return "[Sent document attachment]"
    return f"[Sent {len(media_files)} media attachments]"


def _get_cron_auto_delivery_target():
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip().lower()
    chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
    if not platform or not chat_id:
        return None
    return {
        "platform": platform,
        "chat_id": chat_id,
        "thread_id": get_session_env("HERMES_CRON_AUTO_DELIVER_THREAD_ID", "").strip() or None,
    }


def _maybe_skip_cron_duplicate_send(platform_name, chat_id, thread_id):
    target = _get_cron_auto_delivery_target()
    if not target or target != {
        "platform": platform_name, "chat_id": str(chat_id), "thread_id": thread_id,
    }:
        return None
    label = f"{platform_name}:{chat_id}" + (f":{thread_id}" if thread_id else "")
    return {
        "success": True, "skipped": True,
        "reason": "cron_auto_delivery_duplicate_target", "target": label,
        "note": f"Skipped send_message to {label}; cron will auto-deliver there.",
    }


async def _send_via_adapter(
    platform, pconfig, chat_id, chunk, *, thread_id=None,
    media_files=None, force_document=False,
):
    """Use a live retained adapter, falling back to its internal standalone sender."""
    try:
        from gateway.runtime_registry import get_runner
        runner = get_runner()
    except Exception:
        runner = None
    adapter = runner.adapters.get(platform) if runner is not None else None
    if adapter is not None:
        metadata = {"thread_id": thread_id} if thread_id else None
        result = await adapter.send(chat_id=chat_id, content=chunk, metadata=metadata)
        return (
            {"success": True, "message_id": result.message_id}
            if result.success else {"error": f"Adapter send failed: {result.error}"}
        )
    from gateway.config import Platform
    if platform == Platform.TELEGRAM:
        from gateway.platforms.telegram.adapter import standalone_send
    elif platform == Platform.MATTERMOST:
        from gateway.platforms.mattermost import standalone_send
    else:
        return {"error": f"No sender for platform '{platform.value}'"}
    return await standalone_send(
        pconfig, chat_id, chunk, thread_id=thread_id,
        media_files=media_files, force_document=force_document,
    )


async def _send_to_platform(
    platform, pconfig, chat_id, message, thread_id=None,
    media_files=None, force_document=False, args=None,
):
    from gateway.config import Platform
    media_files = media_files or []
    if platform == Platform.TELEGRAM:
        return await _send_via_adapter(
            platform, pconfig, chat_id, message, thread_id=thread_id,
            media_files=media_files, force_document=force_document,
        )
    if platform != Platform.MATTERMOST:
        return {"error": f"Unsupported delivery platform: {platform.value}"}
    from gateway.platforms.base import BasePlatformAdapter
    chunks = BasePlatformAdapter.truncate_message(message, 4000)
    last_result = None
    for index, chunk in enumerate(chunks):
        last_result = await _send_via_adapter(
            platform, pconfig, chat_id, chunk, thread_id=thread_id,
            media_files=media_files if index == len(chunks) - 1 else [],
            force_document=force_document,
        )
        if isinstance(last_result, dict) and last_result.get("error"):
            return last_result
    return last_result


def _is_telegram_thread_not_found(error: Exception) -> bool:
    return "thread not found" in str(error).lower()


async def _send_telegram(token, chat_id, message, media_files=None, thread_id=None, disable_link_previews=False, force_document=False):
    """Send via Telegram Bot API (one-shot, no polling needed).

    Applies markdown→MarkdownV2 formatting (same as the gateway adapter)
    so that bold, links, and headers render correctly.  If the message
    already contains HTML tags, it is sent with ``parse_mode='HTML'``
    instead, bypassing MarkdownV2 conversion.
    """
    try:
        from telegram import Bot
        from telegram.constants import ParseMode

        # Auto-detect HTML tags — if present, skip MarkdownV2 and send as HTML.
        # Inspired by github.com/ashaney — PR #1568.
        _has_html = bool(re.search(r'<[a-zA-Z/][^>]*>', message))

        if _has_html:
            formatted = message
            send_parse_mode = ParseMode.HTML
        else:
            # Reuse the gateway adapter's format_message for markdown→MarkdownV2
            try:
                from gateway.platforms.telegram.adapter import TelegramAdapter
                _adapter = TelegramAdapter.__new__(TelegramAdapter)
                formatted = _adapter.format_message(message)
            except Exception:
                # Fallback: send as-is if formatting unavailable
                formatted = message
            send_parse_mode = ParseMode.MARKDOWN_V2

        # Honour a configured proxy (telegram.proxy_url in config.yaml, exported
        # as TELEGRAM_PROXY env var by load_gateway_config). Without this, the
        # standalone send path bypasses the proxy and times out in regions
        # where api.telegram.org is blocked. The in-gateway adapter does the
        # same thing in gateway/platforms/telegram.py.
        try:
            from gateway.platforms.base import resolve_proxy_url
            _tg_proxy = resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org"])
        except Exception:
            _tg_proxy = None
        if _tg_proxy:
            try:
                from telegram.request import HTTPXRequest
                logger.info("send_message: standalone Telegram send routed through proxy %s", _tg_proxy)
                bot = Bot(
                    token=token,
                    request=HTTPXRequest(proxy=_tg_proxy),
                    get_updates_request=HTTPXRequest(proxy=_tg_proxy),
                )
            except Exception as _proxy_err:
                logger.warning("send_message: failed to attach Telegram proxy (%s), falling back to direct connection", _proxy_err)
                bot = Bot(token=token)
        else:
            bot = Bot(token=token)
        from gateway.platforms.telegram.telegram_ids import (
            normalize_telegram_chat_id,
        )

        # Telegram accepts a numeric chat_id OR an @username string; normalize
        # rather than force-int so username home channels don't crash (#13206).
        int_chat_id = normalize_telegram_chat_id(chat_id)
        media_files = media_files or []
        thread_kwargs = {}
        if thread_id is not None:
            # Reuse the gateway adapter's General-topic mapping: in Telegram
            # forum supergroups, the General topic is addressed as
            # message_thread_id="1" on incoming updates, but Bot API
            # sendMessage rejects message_thread_id=1 with "Message thread
            # not found". The adapter's helper maps "1" to None for that
            # reason; the send_message tool needs the same mapping or a
            # send to a forum group's General topic always errors out
            # (see issue #22267).
            try:
                from gateway.platforms.telegram.adapter import TelegramAdapter
                effective_thread_id = TelegramAdapter._message_thread_id_for_send(
                    str(thread_id)
                )
            except Exception:
                # Fallback: explicit mapping in case the adapter import
                # fails (e.g. python-telegram-bot missing in this venv).
                effective_thread_id = (
                    None if str(thread_id) == "1" else int(thread_id)
                )
            if effective_thread_id is not None:
                thread_kwargs["message_thread_id"] = effective_thread_id
        # disable_web_page_preview is only valid for send_message, not
        # send_photo/send_video/etc.  Keep it separate so media sends
        # don't inherit an invalid parameter (issue #27012).
        text_kwargs = dict(thread_kwargs)
        if disable_link_previews:
            text_kwargs["disable_web_page_preview"] = True

        last_msg = None
        warnings = []

        # MEDIA:<path> caption: when a single captionable file is accompanied
        # by short text, attach the text to the media bubble as its native
        # caption instead of sending it as a separate message beforehand
        # (single enforced decision in _media_caption_split). Caption with the
        # *formatted* text so MarkdownV2/HTML styling is preserved, but guard
        # the formatted length against Telegram's 1024 cap — formatting can
        # inflate a raw-<1024 string past it, in which case fall back to a
        # separate body message.
        _tg_caption = None
        from gateway.platforms.base import utf16_len as _utf16_len
        _cap, _ = _media_caption_split(
            message, media_files, max_caption_len=_TELEGRAM_CAPTION_LIMIT
        )
        if _cap is not None and _utf16_len(formatted) <= _TELEGRAM_CAPTION_LIMIT:
            _tg_caption = formatted
            formatted = ""  # suppress the separate text send below

        if formatted.strip():
            # Chunk *after* formatting: MarkdownV2/HTML escaping inflates the
            # text (each escaped char like `!`/`.`/`-` becomes `\!`/`\.`/`\-`),
            # so a message that fit under 4096 UTF-16 units raw can exceed the
            # Telegram limit once formatted and get rejected as "Message is too
            # long". Sizing on the formatted text in UTF-16 units guarantees
            # every chunk is deliverable. (issue #28557)
            from gateway.platforms.base import BasePlatformAdapter, utf16_len

            text_chunks = BasePlatformAdapter.truncate_message(
                formatted, 4096, len_fn=utf16_len
            )
            for chunk in text_chunks:
                try:
                    last_msg = await _send_telegram_message_with_retry(
                        bot,
                        chat_id=int_chat_id, text=chunk,
                        parse_mode=send_parse_mode, **text_kwargs
                    )
                except Exception as md_error:
                    # Thread not found — retry without message_thread_id so the
                    # message still delivers (matching the gateway adapter's
                    # fallback behaviour, issue #27012).
                    if _is_telegram_thread_not_found(md_error) and text_kwargs.get("message_thread_id") is not None:
                        logger.warning(
                            "Thread %s not found in _send_telegram, retrying without message_thread_id",
                            text_kwargs.get("message_thread_id"),
                        )
                        text_kwargs.pop("message_thread_id", None)
                        last_msg = await _send_telegram_message_with_retry(
                            bot,
                            chat_id=int_chat_id, text=chunk,
                            parse_mode=send_parse_mode, **text_kwargs
                        )
                    elif "parse" in str(md_error).lower() or "markdown" in str(md_error).lower() or "html" in str(md_error).lower():
                        logger.warning(
                            "Parse mode %s failed in _send_telegram, falling back to plain text: %s",
                            send_parse_mode,
                            _sanitize_error_text(md_error),
                        )
                        if not _has_html:
                            try:
                                from gateway.platforms.telegram.adapter import _strip_mdv2
                                plain = _strip_mdv2(chunk)
                            except Exception:
                                plain = chunk
                        else:
                            plain = chunk
                        last_msg = await _send_telegram_message_with_retry(
                            bot,
                            chat_id=int_chat_id, text=plain,
                            parse_mode=None, **text_kwargs
                        )
                    else:
                        raise

        for media_path, is_voice in media_files:
            if not os.path.exists(media_path):
                warning = f"Media file not found, skipping: {media_path}"
                logger.warning(warning)
                warnings.append(warning)
                # Caption mode suppressed the separate text send; if the file
                # it was meant to caption is gone, deliver the caption text on
                # its own so the words aren't silently lost.
                if _tg_caption is not None and last_msg is None:
                    try:
                        last_msg = await _send_telegram_message_with_retry(
                            bot, chat_id=int_chat_id, text=_tg_caption,
                            parse_mode=send_parse_mode, **text_kwargs
                        )
                        _tg_caption = None  # delivered — don't re-caption a later file
                    except Exception as _cap_err:
                        logger.warning(
                            "Telegram caption-fallback send failed for missing media: %s",
                            _sanitize_error_text(_cap_err),
                        )
                continue

            ext = os.path.splitext(media_path)[1].lower()
            try:
                with open(media_path, "rb") as f:
                    media_kwargs = dict(thread_kwargs)
                    # Attach the MEDIA:<path> caption to the bubble itself for
                    # captionable kinds (photo/video/document). _tg_caption is
                    # only set for a single captionable file, so this never
                    # double-captions a multi-file send or a voice note.
                    if _tg_caption is not None and not (ext in _VOICE_EXTS and is_voice):
                        media_kwargs["caption"] = _tg_caption
                        media_kwargs["parse_mode"] = send_parse_mode
                    if (ext in _VOICE_EXTS and is_voice) or ext in _TELEGRAM_SEND_AUDIO_EXTS:
                        try:
                            from gateway.platforms.telegram.adapter import _probe_voice_duration_seconds
                            duration = await asyncio.to_thread(_probe_voice_duration_seconds, media_path)
                            if duration is not None:
                                media_kwargs["duration"] = duration
                        except Exception:
                            pass
                    try:
                        if ext in _IMAGE_EXTS and not force_document:
                            last_msg = await bot.send_photo(
                                chat_id=int_chat_id, photo=f, **media_kwargs
                            )
                        elif ext in _VIDEO_EXTS:
                            last_msg = await bot.send_video(
                                chat_id=int_chat_id, video=f, **media_kwargs
                            )
                        elif ext in _VOICE_EXTS and is_voice:
                            last_msg = await bot.send_voice(
                                chat_id=int_chat_id, voice=f, **media_kwargs
                            )
                        elif ext in _TELEGRAM_SEND_AUDIO_EXTS:
                            last_msg = await bot.send_audio(
                                chat_id=int_chat_id, audio=f, **media_kwargs
                            )
                        else:
                            last_msg = await bot.send_document(
                                chat_id=int_chat_id, document=f, **media_kwargs
                            )
                    except Exception as media_err:
                        if _is_telegram_thread_not_found(media_err) and media_kwargs.get("message_thread_id"):
                            # Thread not found for media — retry without
                            # message_thread_id (issue #27012).
                            logger.warning(
                                "Thread %s not found for media send, retrying without message_thread_id",
                                media_kwargs["message_thread_id"],
                            )
                            # Re-seek the file since the first attempt consumed it
                            f.seek(0)
                            media_kwargs.pop("message_thread_id", None)
                            if ext in _IMAGE_EXTS and not force_document:
                                last_msg = await bot.send_photo(
                                    chat_id=int_chat_id, photo=f, **media_kwargs
                                )
                            elif ext in _VIDEO_EXTS:
                                last_msg = await bot.send_video(
                                    chat_id=int_chat_id, video=f, **media_kwargs
                                )
                            elif ext in _VOICE_EXTS and is_voice:
                                last_msg = await bot.send_voice(
                                    chat_id=int_chat_id, voice=f, **media_kwargs
                                )
                            elif ext in _TELEGRAM_SEND_AUDIO_EXTS:
                                last_msg = await bot.send_audio(
                                    chat_id=int_chat_id, audio=f, **media_kwargs
                                )
                            else:
                                last_msg = await bot.send_document(
                                    chat_id=int_chat_id, document=f, **media_kwargs
                                )
                        elif media_kwargs.get("parse_mode") and (
                            "parse" in str(media_err).lower()
                            or "caption" in str(media_err).lower()
                        ):
                            # Caption failed to parse as MarkdownV2/HTML —
                            # retry with a plain-text caption so the media
                            # (and its caption) still deliver.
                            logger.warning(
                                "Caption parse failed for media send, retrying plain: %s",
                                _sanitize_error_text(media_err),
                            )
                            f.seek(0)
                            media_kwargs.pop("parse_mode", None)
                            if not _has_html and media_kwargs.get("caption"):
                                try:
                                    from gateway.platforms.telegram.adapter import _strip_mdv2
                                    media_kwargs["caption"] = _strip_mdv2(media_kwargs["caption"])
                                except Exception:
                                    pass
                            if ext in _IMAGE_EXTS and not force_document:
                                last_msg = await bot.send_photo(
                                    chat_id=int_chat_id, photo=f, **media_kwargs
                                )
                            elif ext in _VIDEO_EXTS:
                                last_msg = await bot.send_video(
                                    chat_id=int_chat_id, video=f, **media_kwargs
                                )
                            else:
                                last_msg = await bot.send_document(
                                    chat_id=int_chat_id, document=f, **media_kwargs
                                )
                        else:
                            raise
            except Exception as e:
                warning = _sanitize_error_text(f"Failed to send media {media_path}: {e}")
                logger.error(warning)
                warnings.append(warning)

        if last_msg is None:
            error = "No deliverable text or media remained after processing MEDIA tags"
            if warnings:
                return {"error": error, "warnings": warnings}
            return {"error": error}

        result = {
            "success": True,
            "platform": "telegram",
            "chat_id": chat_id,
            "message_id": str(last_msg.message_id),
        }
        if warnings:
            result["warnings"] = warnings
        return result
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: pip install python-telegram-bot"}
    except Exception as e:
        return _error(f"Telegram send failed: {e}")


def _check_send_message():
    if os.environ.get("HERMES_KANBAN_TASK"):
        return True
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if platform and platform != "local":
        return True
    try:
        from gateway.status import is_gateway_running
        return is_gateway_running()
    except Exception:
        return False


from tools.registry import tool_error

# send_message is intentionally not registered as a model-callable tool. Cron,
# the hermes send command, and the opt-in MCP server call this transport directly.
