from dataclasses import dataclass
import re
from typing import Iterable

from bs4 import BeautifulSoup
import pymupdf

from app.errors import AppError


@dataclass(frozen=True)
class TextBlock:
    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None = None
    section: str | None = None


SUPPORTED_EXTENSIONS = {".pdf", ".htm", ".html", ".txt", ".md"}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


def parse_document(filename: str, content: bytes) -> list[TextBlock]:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        return _parse_pdf(content)
    if extension in {"html", "htm"}:
        return _parse_html(content)
    if extension in {"txt", "md"}:
        return _parse_text(content)
    raise AppError(
        "Unsupported file type. Upload a PDF, HTML, Markdown, or plain-text document."
    )


def _parse_pdf(content: bytes) -> list[TextBlock]:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            blocks = [
                TextBlock(page.get_text("text"), page=index + 1)
                for index, page in enumerate(document)
            ]
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise AppError("The uploaded PDF could not be read.") from exc
    if not any(block.text.strip() for block in blocks):
        raise AppError("The PDF contains no extractable text. OCR scanned PDFs first.")
    return blocks


def _parse_html(content: bytes) -> list[TextBlock]:
    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()
    body = soup.body or soup
    blocks: list[TextBlock] = []
    section = "Document"

    for element in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if element.find_parent("table") and element.name != "table":
            continue
        if element.name == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = [
                    _clean_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                if cells:
                    rows.append(" | ".join(cells))
            text = "\n".join(rows)
        else:
            text = _clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            section = text[:200]
        else:
            blocks.append(TextBlock(text, section=section))

    if not blocks:
        fallback = _clean_text(body.get_text(" ", strip=True))
        if fallback:
            blocks.append(TextBlock(fallback, section="Document"))
    if not blocks:
        raise AppError("The uploaded HTML document contains no extractable text.")
    return blocks


def _parse_text(content: bytes) -> list[TextBlock]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AppError("Text documents must use UTF-8 encoding.") from exc
    text = _clean_text(text)
    if not text:
        raise AppError("The uploaded document contains no extractable text.")
    return [TextBlock(text, section="Document")]


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def chunk_blocks(
    blocks: Iterable[TextBlock],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    grouped_text: list[tuple[int | None, str | None, list[str]]] = []
    for block in blocks:
        if (
            grouped_text
            and grouped_text[-1][0] == block.page
            and grouped_text[-1][1] == block.section
        ):
            grouped_text[-1][2].append(block.text)
        else:
            grouped_text.append((block.page, block.section, [block.text]))

    chunks: list[Chunk] = []
    for page, section, texts in grouped_text:
        block = TextBlock("\n".join(texts), page=page, section=section)
        text = _clean_text(block.text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(piece, page=block.page, section=block.section))
            if end >= len(text):
                break
            next_start = max(start + 1, end - overlap)
            start = next_start
    return chunks
