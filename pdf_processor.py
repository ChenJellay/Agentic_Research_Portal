"""
PDF processing module — Phase 2.

Handles extraction and structuring of text from PDF files, with optional
section-aware parsing for academic papers.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from logger_config import setup_logger

logger = setup_logger(__name__)


class PDFProcessingError(Exception):
    """Exception raised for PDF processing errors."""
    pass


# ---------------------------------------------------------------------------
# Section-aware extraction (new for Phase 2)
# ---------------------------------------------------------------------------

# Common academic section headings (case-insensitive regex patterns)
_SECTION_PATTERNS = [
    r"^(?:abstract)\b",
    r"^(?:\d+\.?\s+)?introduction\b",
    r"^(?:\d+\.?\s+)?background\b",
    r"^(?:\d+\.?\s+)?related\s+work\b",
    r"^(?:\d+\.?\s+)?literature\s+review\b",
    r"^(?:\d+\.?\s+)?methodology?\b",
    r"^(?:\d+\.?\s+)?method(?:s)?\b",
    r"^(?:\d+\.?\s+)?approach\b",
    r"^(?:\d+\.?\s+)?experiment(?:s|al)?\b",
    r"^(?:\d+\.?\s+)?result(?:s)?\b",
    r"^(?:\d+\.?\s+)?evaluation\b",
    r"^(?:\d+\.?\s+)?discussion\b",
    r"^(?:\d+\.?\s+)?conclusion(?:s)?\b",
    r"^(?:\d+\.?\s+)?future\s+work\b",
    r"^(?:\d+\.?\s+)?acknowledge?ments?\b",
    r"^(?:\d+\.?\s+)?references?\b",
    r"^(?:\d+\.?\s+)?appendix\b",
    r"^(?:\d+\.?\s+)?(?:[A-Z][a-z]+(?:\s+[A-Za-z]+){0,4})$",  # generic numbered heading
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SECTION_PATTERNS]


def _is_section_heading(line: str) -> bool:
    """Heuristic: check if a line looks like an academic section heading."""
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 100:
        return False
    # Must match one of the known patterns
    for pat in _COMPILED_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _detect_heading_by_font(page: Any) -> List[Tuple[str, int]]:
    """
    Use pdfplumber char-level data to detect headings by font size.
    Returns list of (heading_text, page_number).
    """
    headings: List[Tuple[str, int]] = []
    try:
        chars = page.chars
        if not chars:
            return headings

        # Compute median font size
        sizes = [c.get("size", 0) for c in chars if c.get("size")]
        if not sizes:
            return headings
        sizes_sorted = sorted(sizes)
        median_size = sizes_sorted[len(sizes_sorted) // 2]

        # Group chars into lines (by similar top coordinate)
        lines: Dict[float, List[Any]] = {}
        for c in chars:
            top = round(c.get("top", 0), 1)
            lines.setdefault(top, []).append(c)

        for top in sorted(lines.keys()):
            line_chars = sorted(lines[top], key=lambda c: c.get("x0", 0))
            avg_size = sum(c.get("size", 0) for c in line_chars) / max(len(line_chars), 1)
            text = "".join(c.get("text", "") for c in line_chars).strip()

            # Heading heuristic: font ≥ 1.15× median AND short line
            if avg_size >= median_size * 1.15 and 3 <= len(text) <= 120:
                headings.append((text, page.page_number))
    except Exception:
        pass  # fall back gracefully
    return headings


def extract_sections_from_pdf(filepath: Path) -> List[Dict[str, Any]]:
    """
    Extract section-structured content from a PDF.

    Returns a list of section dicts::

        [
            {
                "section": "Abstract",
                "text": "...",
                "page_start": 1,
                "page_end": 1
            },
            ...
        ]

    Falls back to flat extraction if section detection fails.
    """
    try:
        all_page_texts: List[Tuple[int, str]] = []  # (page_num, text)
        font_headings: List[Tuple[str, int]] = []

        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                text = page.extract_text() or ""
                all_page_texts.append((page_num, text))
                font_headings.extend(_detect_heading_by_font(page))

        if not all_page_texts:
            raise PDFProcessingError(f"No text extracted from {filepath}")

        # Build full text with page markers
        lines_with_page: List[Tuple[str, int]] = []
        for page_num, text in all_page_texts:
            for line in text.split("\n"):
                lines_with_page.append((line, page_num))

        # Identify section boundaries using both font and text heuristics
        font_heading_texts = {h[0].lower().strip() for h in font_headings}
        sections: List[Dict[str, Any]] = []
        current_section = "Preamble"
        current_lines: List[str] = []
        current_page_start = 1

        for line, page_num in lines_with_page:
            is_heading = False
            heading_text = line.strip()

            # Check text-pattern heuristic
            if _is_section_heading(heading_text):
                is_heading = True
            # Check font-size heuristic
            elif heading_text.lower().strip() in font_heading_texts:
                is_heading = True

            if is_heading and current_lines:
                # Save previous section
                sections.append({
                    "section": current_section,
                    "text": _clean_extracted_text("\n".join(current_lines)),
                    "page_start": current_page_start,
                    "page_end": page_num,
                })
                current_section = heading_text
                current_lines = []
                current_page_start = page_num
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            last_page = lines_with_page[-1][1] if lines_with_page else current_page_start
            sections.append({
                "section": current_section,
                "text": _clean_extracted_text("\n".join(current_lines)),
                "page_start": current_page_start,
                "page_end": last_page,
            })

        # If we only got one section ("Preamble"), section detection basically failed
        if len(sections) <= 1:
            logger.warning(f"Section detection yielded ≤1 section for {filepath.name}; using flat text.")
            flat_text = "\n\n".join(text for _, text in all_page_texts)
            return [{
                "section": "Full Document",
                "text": _clean_extracted_text(flat_text),
                "page_start": 1,
                "page_end": all_page_texts[-1][0],
            }]

        logger.info(
            f"Extracted {len(sections)} sections from {filepath.name} "
            f"(pages 1–{all_page_texts[-1][0]})"
        )
        return sections

    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(f"Error processing PDF {filepath}: {e}") from e


# ---------------------------------------------------------------------------
# Flat extraction (carried forward from Phase 1)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: Path) -> str:
    """
    Extract text from a single PDF file (flat, non-sectioned).

    Args:
        filepath: Path to the PDF file.

    Returns:
        Extracted text content.

    Raises:
        PDFProcessingError: If PDF cannot be read or processed.
    """
    try:
        logger.debug(f"Extracting text from PDF: {filepath}")
        text_parts: List[str] = []

        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    logger.debug(f"Extracted text from page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

        if not text_parts:
            raise PDFProcessingError(f"No text could be extracted from {filepath}")

        combined_text = "\n\n".join(text_parts)
        cleaned_text = _clean_extracted_text(combined_text)

        logger.info(f"Successfully extracted {len(cleaned_text)} characters from {filepath.name}")
        return cleaned_text

    except FileNotFoundError:
        raise PDFProcessingError(f"PDF file not found: {filepath}")
    except Exception as e:
        raise PDFProcessingError(f"Error processing PDF {filepath}: {str(e)}") from e


def save_processed_source(
    source_id: str,
    sections: List[Dict[str, Any]],
    dest_dir: Path,
) -> Path:
    """Persist extracted sections to a JSON file in data/processed/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{source_id}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"source_id": source_id, "sections": sections}, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved processed source → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_sources_for_task(
    task_id: int, source_map: Dict[str, str], base_dir: Path
) -> Dict[str, str]:
    """Load multiple PDF sources for a task (flat text)."""
    logger.info(f"Loading sources for task {task_id}")
    sources: Dict[str, str] = {}
    failed: List[str] = []

    for source_id, filename in source_map.items():
        filepath = base_dir / filename
        try:
            text = extract_text_from_pdf(filepath)
            sources[source_id] = text
        except PDFProcessingError as e:
            logger.error(f"Failed to load source {source_id}: {e}")
            failed.append(source_id)

    if not sources:
        raise PDFProcessingError(f"No sources loaded for task {task_id}")
    return sources


def combine_source_texts(sources: Dict[str, str]) -> str:
    parts: List[str] = []
    for source_id, text in sources.items():
        sep = f"\n\n{'='*80}\nSOURCE: {source_id}\n{'='*80}\n\n"
        parts.append(sep + text)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    lines = text.split("\n")
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
        elif cleaned and cleaned[-1]:
            cleaned.append("")
    while cleaned and not cleaned[-1]:
        cleaned.pop()
    return "\n".join(cleaned)
