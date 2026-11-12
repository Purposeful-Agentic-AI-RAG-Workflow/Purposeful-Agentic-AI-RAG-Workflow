"""Read, clean, and deterministically chunk citizen-feedback text files."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

from models import Chunk


class IngestionError(Exception):
    """Raised when feedback files cannot be read or produce no usable text."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Ingestion failed for {path}: {reason}")


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "pre",
        "section",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


_REPEATED_CHARACTERS = re.compile(r"([^\s])\1{2,}", flags=re.IGNORECASE)
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff", "\ufe0e", "\ufe0f"}


def _remove_emoji_and_noise_characters(text: str) -> str:
    cleaned: list[str] = []
    for character in text:
        codepoint = ord(character)
        if character in _ZERO_WIDTH:
            continue
        if unicodedata.category(character) == "So":
            continue
        if 0x1F000 <= codepoint <= 0x1FAFF or 0x1FC00 <= codepoint <= 0x1FFFF:
            continue
        if 0x1F1E6 <= codepoint <= 0x1F1FF or 0x2600 <= codepoint <= 0x27BF:
            continue
        cleaned.append(character)
    return "".join(cleaned)


def clean_text(text: str) -> str:
    """Remove common markup and noisy characters without editing the source file."""
    extractor = _TextExtractor()
    extractor.feed(text)
    extractor.close()
    text = "".join(extractor.parts)
    text = _remove_emoji_and_noise_characters(text)
    text = _REPEATED_CHARACTERS.sub(lambda match: match.group(1) * 2, text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        if current and current_length + 1 + len(word) > max_chars:
            parts.append(" ".join(current))
            current = []
            current_length = 0
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)

    if current:
        parts.append(" ".join(current))
    return parts


def _paragraph_chunks(text: str, max_chars: int) -> Iterable[tuple[str, str]]:
    current_heading: str | None = None
    paragraph_number = 0

    for raw_paragraph in re.split(r"\n\s*\n+", text):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue

        lines = paragraph.splitlines()
        if lines and re.match(r"^\s{0,3}#{1,6}\s+", lines[0]):
            current_heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", lines.pop(0)).strip()
            paragraph = "\n".join(lines).strip()
            if not paragraph:
                continue
            section = current_heading or "general"
        elif current_heading:
            section = current_heading
        else:
            paragraph_number += 1
            section = f"paragraph {paragraph_number}"

        normalized = re.sub(r"\s+", " ", paragraph).strip()
        for part in _split_long_paragraph(normalized, max_chars):
            yield section, part


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise IngestionError(path, f"unreadable file ({error})") from error


def _ingest_file(path: Path, doc_id: str, max_chars: int) -> list[Chunk]:
    source_text = _read_file(path)
    if not source_text.strip():
        raise IngestionError(path, "file is empty")

    cleaned = clean_text(source_text)
    if not cleaned:
        raise IngestionError(path, "file contains no usable text after cleaning")

    chunks: list[Chunk] = []
    duplicate_counts: defaultdict[str, int] = defaultdict(int)
    for section, chunk_text in _paragraph_chunks(cleaned, max_chars):
        digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
        occurrence = duplicate_counts[digest]
        duplicate_counts[digest] += 1
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"chunk-{digest}-{occurrence:02d}",
                section=section,
                text=chunk_text,
            )
        )

    if not chunks:
        raise IngestionError(path, "file contains no usable paragraphs")
    return chunks


def ingest_files(paths: Iterable[str | Path], max_chars: int = 1200) -> list[Chunk]:
    """Ingest an explicit set of .txt/.md files without reading unrelated files."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    files = sorted({Path(path).resolve() for path in paths}, key=lambda path: str(path).casefold())
    if not files:
        raise IngestionError(Path("."), "no files were provided")

    for path in files:
        if path.suffix.lower() not in {".txt", ".md"}:
            raise IngestionError(path, "only .txt and .md files are supported")
        if not path.is_file():
            raise IngestionError(path, "path does not refer to a file")

    common_root = Path(os.path.commonpath([str(path.parent) for path in files]))
    chunks: list[Chunk] = []
    for path in files:
        doc_id = path.relative_to(common_root).as_posix()
        chunks.extend(_ingest_file(path, doc_id, max_chars))
    return chunks


def ingest_directory(data_dir: str | Path = "data", max_chars: int = 1200) -> list[Chunk]:
    """Ingest all .txt/.md files under a directory into stable paragraph chunks.

    A file that cannot be read, is empty, or contains no usable text fails ingestion
    explicitly instead of being silently skipped. Source files are read-only.
    """
    root = Path(data_dir)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if not root.exists() or not root.is_dir():
        raise IngestionError(root, "data directory does not exist or is not a directory")

    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not files:
        raise IngestionError(root, "no .txt or .md files found")

    chunks: list[Chunk] = []
    for path in files:
        doc_id = path.relative_to(root).as_posix()
        chunks.extend(_ingest_file(path, doc_id, max_chars))

    return chunks
