"""
Document processing engine for the investigation simulator.

Responsibilities
----------------
* Walk ``data/public_records/`` (and optionally hidden records).
* Extract raw text from PDF, plain-text, and ``.eml`` files.
* Stamp a stable Evidence ID (``EVID-001``, ``EVID-002``, ...) on each file.
* Return a structured mapping that preserves the source file name and ID.

The ID sequence is deterministic: files are sorted by relative path so reruns
produce the same identifiers. Hidden files continue the sequence after public
files when both directories are processed together.
"""

from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Iterable

from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "data" / "public_records"
HIDDEN_DIR = ROOT / "data" / "hidden_records"

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".eml", ".md"}


class DocumentProcessingError(RuntimeError):
    """Raised when a case file cannot be read or decoded."""


def extract_text_from_pdf(path: Path) -> str:
    """Extract concatenated page text from a PDF using PyPDF2."""
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise DocumentProcessingError(f"Unable to open PDF {path.name}: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise DocumentProcessingError(
                f"Unable to extract text from {path.name} page {index}: {exc}"
            ) from exc
        pages.append(page_text.strip())
    return "\n\n".join(part for part in pages if part).strip()


def extract_text_from_email(path: Path) -> str:
    """
    Parse a ``.eml`` file and return headers plus the plain-text body.

    Falls back to raw UTF-8 if the file is not a valid RFC 5322 message.
    """
    raw = path.read_bytes()
    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:
        return raw.decode("utf-8", errors="replace").strip()

    headers = [
        f"From: {message.get('From', '')}",
        f"To: {message.get('To', '')}",
        f"Cc: {message.get('Cc', '')}",
        f"Date: {message.get('Date', '')}",
        f"Subject: {message.get('Subject', '')}",
    ]
    body = message.get_body(preferencelist=("plain", "html"))
    payload = ""
    if body is not None:
        payload = str(body.get_content())
    else:
        payload = message.get_payload(decode=True)
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        payload = payload or ""
    return "\n".join(headers + ["", payload.strip()]).strip()


def extract_text_from_file(path: Path) -> str:
    """Dispatch text extraction based on file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".eml":
        return extract_text_from_email(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    raise DocumentProcessingError(f"Unsupported file type: {path.name}")


def _iter_supported_files(directory: Path) -> Iterable[Path]:
    """Yield supported files in a directory, sorted by name for stable IDs."""
    if not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda item: item.name.lower(),
    )


def make_evidence_id(index: int) -> str:
    """Return a zero-padded evidence identifier such as ``EVID-001``."""
    if index < 1:
        raise ValueError("Evidence index must be >= 1")
    return f"EVID-{index:03d}"


def build_record(
    path: Path,
    evidence_id: str,
    access: str,
    *,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    """Build one structured evidence dictionary for a source file."""
    text = extract_text_from_file(path)
    if not text:
        raise DocumentProcessingError(f"No extractable text in {path.name}")
    try:
        relative_name = str(path.relative_to(source_root))
    except ValueError:
        relative_name = path.name
    return {
        "evidence_id": evidence_id,
        "source_file": path.name,
        "relative_path": relative_name.replace("\\", "/"),
        "absolute_path": str(path.resolve()),
        "access": access,
        "text": text,
        "char_count": len(text),
    }


def process_directory(
    directory: Path,
    access: str,
    *,
    start_index: int = 1,
    source_root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """
    Extract every supported file in ``directory`` and assign sequential IDs.

    Parameters
    ----------
    directory:
        Folder to scan (non-recursive).
    access:
        Visibility label stored on each record (``public`` or ``hidden``).
    start_index:
        First numeric ID to assign. Hidden records typically continue after
        the last public ID.
    source_root:
        Root used to compute ``relative_path``.

    Returns
    -------
    dict
        Mapping of evidence ID to a structured record.
    """
    records: dict[str, dict[str, Any]] = {}
    index = start_index
    for path in _iter_supported_files(directory):
        evidence_id = make_evidence_id(index)
        records[evidence_id] = build_record(
            path,
            evidence_id,
            access,
            source_root=source_root,
        )
        index += 1
    return records


def load_public_evidence() -> dict[str, dict[str, Any]]:
    """Process only the investigator's starting pack."""
    if not PUBLIC_DIR.exists():
        raise DocumentProcessingError(
            f"Public records folder is missing: {PUBLIC_DIR}. "
            "Run generate_case_files.py first."
        )
    return process_directory(PUBLIC_DIR, access="public", start_index=1)


def load_all_evidence() -> dict[str, dict[str, Any]]:
    """
    Process public then hidden records with a single ID sequence.

    Hidden text is extracted here so later unlock logic can reveal an already
    tagged record instead of re-parsing files at runtime.
    """
    public = load_public_evidence()
    next_index = len(public) + 1
    hidden = process_directory(HIDDEN_DIR, access="hidden", start_index=next_index)
    merged = dict(public)
    merged.update(hidden)
    return merged


def evidence_summary(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ID, file name, access, and character count without full text."""
    return [
        {
            "evidence_id": record["evidence_id"],
            "source_file": record["source_file"],
            "access": record["access"],
            "char_count": record["char_count"],
        }
        for record in records.values()
    ]


def main() -> None:
    """CLI check: print a clean JSON summary of public evidence."""
    records = load_public_evidence()
    print(json.dumps(evidence_summary(records), indent=2, ensure_ascii=False))
    print()
    for evidence_id, record in records.items():
        preview = " ".join(record["text"].split())[:160]
        print(f"{evidence_id}  {record['source_file']}")
        print(f"    {preview}...")
        print()


if __name__ == "__main__":
    main()
