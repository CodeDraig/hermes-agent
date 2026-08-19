"""Stdlib document-to-text extraction for ``read_file``.

Supports Jupyter notebooks, DOCX, and XLSX without adding hard dependencies.
Malformed documents raise :class:`ExtractionError`; callers can then fall back to
normal text/binary handling.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

__all__ = [
    "EXTRACTABLE_EXTENSIONS",
    "ExtractionError",
    "extract_document_bytes",
    "extract_document_text",
    "is_extractable_document",
]

EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx"})
MAX_XLSX_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_MAX_XLSX_ROWS_PER_SHEET = 5000
_MAX_XLSX_COLS = 256

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class ExtractionError(Exception):
    """Raised when a supported-looking document cannot be rendered as text."""


def _extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    return ext if ext in EXTRACTABLE_EXTENSIONS else ""


def is_extractable_document(path: str) -> bool:
    return bool(_extension(path))


def extract_document_text(path: str) -> str:
    ext = _extension(path)
    if ext == ".ipynb":
        return _extract_notebook(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".xlsx":
        return _extract_xlsx(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def extract_document_bytes(data: bytes, path: str) -> str:
    """Extract a document already fetched across a file backend boundary."""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, limit is {MAX_DOCUMENT_BYTES:,})"
        )
    ext = _extension(path)
    if ext not in EXTRACTABLE_EXTENSIONS:
        raise ExtractionError(f"Unsupported document type: {path!r}")

    # The stdlib extractors are path-oriented. Materialize backend bytes in a
    # private host temp file, then remove it even when parsing fails.
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
            fh.write(data)
            temp_path = fh.name
        return extract_document_text(temp_path)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _source_text(source) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def _human_size(n_bytes: int) -> str:
    return f"{round(n_bytes / 1024)} KB" if n_bytes >= 1024 else f"{n_bytes} B"


def _base64_bytes(payload: str) -> int:
    """Approximate decoded size of a base64 payload (whitespace ignored)."""
    clean = re.sub(r"[^0-9+/=A-Za-z]", "", payload)
    padding = min(2, len(clean) - len(clean.rstrip("=")))
    return max(0, (len(clean) * 3) // 4 - padding)


def _clean_stream_text(text: str) -> str:
    """Strip ANSI escapes and collapse ``\\r`` progress-bar rewrites.

    tqdm and friends redraw the same line via carriage returns; Jupyter
    renders only the final frame, so keeping the text after the last ``\\r``
    of each line reproduces what the notebook displays without the invisible
    intermediate frames.
    """
    from tools.ansi_strip import strip_ansi

    cleaned = strip_ansi(text).replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        frames = [frame for frame in line.split("\r") if frame]
        lines.append(frames[-1] if frames else "")
    return "\n".join(lines)


# Notebook outputs longer than this are tail-truncated per output block so a
# single runaway training log cannot flood the extracted text.
_MAX_OUTPUT_CHARS = 20_000


def _notebook_output_text(output: Any) -> str:
    """Render one notebook output as compact text.

    Keeps stream text, error tracebacks, and textual results; replaces
    token-heavy payloads (base64 images, HTML, widget state) with short
    sized placeholders. Handles both nbformat v4 output shapes and the
    legacy v3 ones (``pyout``/``pyerr``; data flat on the output dict).
    """
    if not isinstance(output, dict):
        return ""
    otype = output.get("output_type")

    if otype == "stream":
        body = _clean_stream_text(_source_text(output.get("text", "")))
        return body if body.strip() else ""

    if otype in {"error", "pyerr"}:
        traceback = output.get("traceback")
        tb_text = ""
        if isinstance(traceback, list):
            tb_text = _clean_stream_text(
                "\n".join(line for line in traceback if isinstance(line, str))
            )
        header = f"Error: {output.get('ename', '')}: {output.get('evalue', '')}".rstrip(": ")
        return f"{header}\n{tb_text}".rstrip()

    if otype in {"execute_result", "display_data", "pyout"}:
        data = output.get("data")
        if not isinstance(data, dict):
            # nbformat v3 stores mime data flat on the output dict.
            data = {}
            if isinstance(output.get("text"), (str, list)):
                data["text/plain"] = output["text"]
            for v3_key, mime in (("png", "image/png"), ("jpeg", "image/jpeg"),
                                 ("svg", "image/svg+xml"), ("html", "text/html")):
                if v3_key in output:
                    data[mime] = output[v3_key]

        if "application/vnd.jupyter.widget-view+json" in data:
            return "[interactive widget — omitted]"

        # Prefer readable text: models consume text/plain (e.g. the pandas
        # twin of an HTML table) far better than markup.
        for mime in ("text/plain", "text/markdown"):
            if mime in data:
                body = _clean_stream_text(_source_text(data[mime]))
                if body.strip():
                    return body

        for mime, value in data.items():
            if isinstance(mime, str) and mime.startswith("image/"):
                size = _base64_bytes(_source_text(value))
                return f"[{mime} output — {_human_size(size)}, omitted]"

        if "text/html" in data:
            html = _source_text(data["text/html"])
            return f"[text/html output — {len(html):,} chars, omitted]"

        mimes = ", ".join(str(m) for m in data) or "unknown"
        return f"[{mimes} output — omitted]"

    return ""


def _notebook_outputs(cell: dict, jq_pointer: str = "", filename: str = "") -> str:
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return ""
    blocks = [text for text in (_notebook_output_text(o) for o in outputs) if text]
    if not blocks:
        return ""
    joined = "\n".join(blocks)
    if len(joined) > _MAX_OUTPUT_CHARS:
        omitted = len(joined) - _MAX_OUTPUT_CHARS
        hint = ""
        if jq_pointer and filename:
            hint = f" — full output: jq -r '{jq_pointer}' {filename}"
        joined = joined[:_MAX_OUTPUT_CHARS] + f"\n… [{omitted:,} output chars truncated{hint}]"
    return joined


def _extract_notebook(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            nb = json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    if not isinstance(nb, dict):
        raise ExtractionError("Notebook root is not an object")

    raw_cells = nb.get("cells")
    if isinstance(raw_cells, list):
        cells = [(f".cells[{i}].outputs", cell) for i, cell in enumerate(raw_cells)]
    else:
        cells = [
            (f".worksheets[{wi}].cells[{ci}].outputs", cell)
            for wi, ws in enumerate(nb.get("worksheets", []))
            if isinstance(ws, dict)
            for ci, cell in enumerate(ws.get("cells", []))
        ]
    if not cells:
        raise ExtractionError("Notebook contains no cells")

    nb_name = os.path.basename(path)
    counts = {"markdown": 0, "code": 0, "raw": 0}
    labels = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}
    out: list[str] = []
    for jq_pointer, cell in cells:
        if not isinstance(cell, dict):
            continue
        typ = cell.get("cell_type")
        if typ not in labels:
            continue
        counts[typ] += 1
        suffix = f" {counts[typ]}" if typ != "raw" else ""
        out.extend((f"# ── {labels[typ]} cell{suffix} ──", _source_text(cell.get("source", "")).rstrip("\n"), ""))
        if typ == "code":
            rendered = _notebook_outputs(cell, jq_pointer, nb_name)
            if rendered:
                out.extend((f"# ── Output (cell {counts[typ]}) ──", rendered.rstrip("\n"), ""))
    if not out:
        raise ExtractionError("Notebook contains no readable cells")
    return "\n".join(out).rstrip("\n") + "\n"


def _zip_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        return ET.fromstring(zf.read(name))
    except KeyError as exc:
        raise ExtractionError(f"Missing {name}") from exc
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {name}: {exc}") from exc


def _extract_docx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            root = _zip_xml(zf, "word/document.xml")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid DOCX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc

    w = f"{{{_NS_W}}}"
    lines: list[str] = []
    for para in root.iter(f"{w}p"):
        buf: list[str] = []
        for node in para.iter():
            if node.tag == f"{w}t":
                buf.append(node.text or "")
            elif node.tag == f"{w}tab":
                buf.append("\t")
            elif node.tag in {f"{w}br", f"{w}cr"}:
                buf.append("\n")
        lines.extend("".join(buf).split("\n"))
    if not any(line.strip() for line in lines):
        raise ExtractionError("DOCX contains no extractable text")
    return "\n".join(lines).rstrip("\n") + "\n"


def _extract_xlsx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            shared = _shared_strings(zf, names)
            sheets = _workbook_sheets(zf)
            rels = _workbook_rels(zf, names)
            out: list[str] = []
            for name, state, rid in sheets:
                if state in {"hidden", "veryHidden"}:
                    continue
                part = _sheet_part(rels.get(rid, ""))
                if part not in names:
                    continue
                try:
                    rows = _sheet_rows(zf.read(part), shared)
                except ET.ParseError:
                    continue
                out.append(f"# ── Sheet: {name} ──")
                out.extend("\t".join(row) for row in rows)
                if not rows:
                    out.append("(empty)")
                out.append("")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid XLSX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc

    if not out:
        raise ExtractionError("XLSX has no visible sheets with content")
    return "\n".join(out).rstrip("\n") + "\n"


def _shared_strings(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except ET.ParseError:
        return []
    s = f"{{{_NS_S}}}"
    return ["".join(t.text or "" for t in item.iter(f"{s}t")) for item in root.iter(f"{s}si")]


def _workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    root = _zip_xml(zf, "xl/workbook.xml")
    s, r = f"{{{_NS_S}}}", f"{{{_NS_REL}}}"
    return [
        (sheet.get("name", "Sheet"), sheet.get("state", "visible"), sheet.get(f"{r}id", ""))
        for sheet in root.iter(f"{s}sheet")
    ]


def _workbook_rels(zf: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    rels_path = "xl/_rels/workbook.xml.rels"
    if rels_path not in names:
        return {}
    try:
        root = ET.fromstring(zf.read(rels_path))
    except ET.ParseError:
        return {}
    rel_tag = f"{{{_NS_PKG_REL}}}Relationship"
    return {rel.get("Id", ""): rel.get("Target", "") for rel in root.iter(rel_tag) if rel.get("Id")}


def _sheet_part(target: str) -> str:
    target = target.lstrip("/")
    return posixpath.normpath(target if target.startswith("xl/") else f"xl/{target}")


def _col_index(ref: str) -> int:
    idx = 0
    for ch in ref:
        if not ch.isalpha():
            break
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return max(idx - 1, 0)


def _sheet_rows(xml_bytes: bytes, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(xml_bytes)
    s = f"{{{_NS_S}}}"
    rows: list[list[str]] = []
    for row in root.iter(f"{s}row"):
        if len(rows) >= _MAX_XLSX_ROWS_PER_SHEET:
            break
        cells: dict[int, str] = {}
        max_col = -1
        for cell in row.iter(f"{s}c"):
            col = _col_index(cell.get("r", "")) if cell.get("r") else max_col + 1
            if col >= _MAX_XLSX_COLS:
                continue
            cells[col] = _cell_value(cell, shared, s)
            max_col = max(max_col, col)
        rows.append([cells.get(i, "") for i in range(max_col + 1)] if max_col >= 0 else [])
    while rows and not any(value.strip() for value in rows[-1]):
        rows.pop()
    return rows


def _cell_value(cell: ET.Element, shared: list[str], s: str) -> str:
    value = cell.findtext(f"{s}v") or ""
    typ = cell.get("t", "")
    if typ == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return ""
    if typ == "inlineStr":
        inline = cell.find(f"{s}is")
        return "" if inline is None else "".join(t.text or "" for t in inline.iter(f"{s}t"))
    if typ == "b":
        return "TRUE" if value.strip() in {"1", "true", "TRUE"} else "FALSE"
    if typ == "e":
        return value or "#ERROR"
    return value
