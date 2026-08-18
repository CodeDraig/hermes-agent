"""Contract tests for the retained shared media-cache dispatch."""

import mimetypes

import pytest

from gateway.platforms.media_cache import (
    DEFAULT_EXT_TO_MIME,
    DEFAULT_MIME_TO_EXT,
    cache_media_bytes,
    ext_for_mime,
    mime_for_ext,
)


# ---------------------------------------------------------------------------
# Shared table contract
# ---------------------------------------------------------------------------

class TestSharedTable:
    def test_defaults_resolve(self):
        for mime, ext in DEFAULT_MIME_TO_EXT.items():
            assert ext_for_mime(mime) == ext


    def test_stage_gating(self):
        # use_defaults=False skips the shared table.
        assert ext_for_mime(
            "audio/ogg", use_defaults=False, use_mimetypes=False, fallback=".x"
        ) == ".x"
        # use_mimetypes=False skips the mimetypes fallback.
        assert ext_for_mime("image/bmp", use_mimetypes=False) is None


    def test_mime_for_ext_fallback_and_case(self):
        assert mime_for_ext(".JPG") == "image/jpeg"
        assert mime_for_ext(".unknown") == "application/octet-stream"
        assert mime_for_ext(".unknown", fallback="x/y") == "x/y"
        assert mime_for_ext(".pdf", overrides={".pdf": "custom/pdf"}) == "custom/pdf"


# ---------------------------------------------------------------------------
# cache_media_bytes dispatch
# ---------------------------------------------------------------------------

class TestCacheMediaBytes:
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    def test_image_dispatch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(self.PNG, "image/png")
        assert path.endswith(".png")


    def test_document_dispatch_uses_filename_hint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "gateway.platforms.base.get_document_cache_dir", lambda: tmp_path
        )
        path = cache_media_bytes(b"%PDF-1.4", "application/pdf",
                                 filename_hint="report.pdf")
        assert path.endswith("_report.pdf")
