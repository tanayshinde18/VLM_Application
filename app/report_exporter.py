from __future__ import annotations

from collections import Counter
from datetime import datetime
from textwrap import wrap
from typing import Dict, Iterable, List, Sequence, Tuple


LineItem = Tuple[str, int, str]


def build_incident_pdf_report(
    logs: Sequence[Dict[str, str]],
    title: str = "Incident Surveillance Report",
) -> bytes:
    """
    Build a clean, text-first PDF report from detection logs.
    Uses a dependency-free PDF writer for portability.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line_items = _build_report_lines(logs=logs, title=title, generated_at=generated_at)
    pages = _paginate_lines(line_items)
    return _render_pdf(pages)


def _build_report_lines(logs: Sequence[Dict[str, str]], title: str, generated_at: str) -> List[LineItem]:
    risk_counts = Counter(
        log.get("Risk Level", log.get("risk_level", "UNKNOWN")) for log in logs
    )

    lines: List[LineItem] = [
        ("F2", 18, title),
        ("F1", 10, f"Generated At: {generated_at}"),
        ("F1", 10, f"Total Events: {len(logs)}"),
        ("F1", 10, f"DANGEROUS: {risk_counts.get('DANGEROUS', 0)}"),
        ("F1", 10, f"SUSPICIOUS: {risk_counts.get('SUSPICIOUS', 0)}"),
        ("F1", 10, f"SAFE: {risk_counts.get('SAFE', 0)}"),
        ("F1", 10, ""),
        ("F2", 12, "Event Log"),
        ("F1", 10, "-" * 88),
        ("F2", 10, "Time       Risk         Caption"),
        ("F1", 10, "-" * 88),
    ]

    for idx, log in enumerate(logs, start=1):
        timestamp = str(log.get("Timestamp", log.get("timestamp", "N/A")))
        risk = str(log.get("Risk Level", log.get("risk_level", "UNKNOWN")))
        caption = str(log.get("Caption Snippet", log.get("caption", "")))
        sentiment = str(log.get("sentiment", "")).upper()
        unsafe = str(log.get("unsafe", ""))
        reason = str(log.get("reason", ""))

        caption_text = caption
        if sentiment:
            caption_text += f" | Sentiment: {sentiment}"
        if unsafe:
            caption_text += f" | Unsafe: {unsafe}"
        if reason:
            caption_text += f" | {reason}"

        caption_parts = wrap(caption_text, width=58) or [""]
        first_line_prefix = f"{timestamp:<10} {risk:<12} "
        lines.append(("F1", 10, f"{idx:>3}. {first_line_prefix}{caption_parts[0]}"))
        for continuation in caption_parts[1:]:
            lines.append(("F1", 10, f"     {'':<10} {'':<12} {continuation}"))
        lines.append(("F1", 10, "-" * 88))

    return lines


def _paginate_lines(lines: Sequence[LineItem], max_lines_per_page: int = 44) -> List[List[LineItem]]:
    pages: List[List[LineItem]] = []
    current_page: List[LineItem] = []

    for line in lines:
        current_page.append(line)
        if len(current_page) >= max_lines_per_page:
            pages.append(current_page)
            current_page = []

    if current_page:
        pages.append(current_page)

    if not pages:
        pages = [[("F2", 14, "No events found.")]]

    return pages


def _render_pdf(pages: Iterable[Sequence[LineItem]]) -> bytes:
    page_list = list(pages)

    objects: Dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
    objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

    page_object_ids: List[int] = []
    next_obj_id = 5

    for line_items in page_list:
        page_id = next_obj_id
        content_id = next_obj_id + 1
        next_obj_id += 2
        page_object_ids.append(page_id)

        stream = _build_content_stream(line_items)
        stream_header = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
        objects[content_id] = stream_header + stream + b"\nendstream"

        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")

    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[2] = f"<< /Type /Pages /Count {len(page_object_ids)} /Kids [{kids}] >>".encode("ascii")

    max_obj_id = max(objects.keys())
    pdf_bytes = bytearray()
    pdf_bytes.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: Dict[int, int] = {}
    for obj_id in range(1, max_obj_id + 1):
        obj_data = objects[obj_id]
        offsets[obj_id] = len(pdf_bytes)
        pdf_bytes.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        pdf_bytes.extend(obj_data)
        pdf_bytes.extend(b"\nendobj\n")

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {max_obj_id + 1}\n".encode("ascii"))
    pdf_bytes.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, max_obj_id + 1):
        pdf_bytes.extend(f"{offsets[obj_id]:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<< /Size {max_obj_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF"
    )
    pdf_bytes.extend(trailer.encode("ascii"))
    return bytes(pdf_bytes)


def _build_content_stream(line_items: Sequence[LineItem]) -> bytes:
    commands: List[str] = ["BT"]
    y_position = 750
    line_height = 16

    for font, size, text in line_items:
        escaped = _escape_pdf_text(text)
        commands.append(f"/{font} {size} Tf")
        commands.append(f"1 0 0 1 50 {y_position} Tm")
        commands.append(f"({escaped}) Tj")
        y_position -= line_height

    commands.append("ET")
    stream_text = "\n".join(commands)
    return stream_text.encode("latin-1", errors="replace")


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
