"""Zero-GPU office document text extraction routines for local semantic search."""

import contextlib
from pathlib import Path
from typing import ClassVar


class PdfExtractor:
    """Extract plain text from PDF documents without discrete GPU acceleration."""

    supported_extensions: ClassVar[set[str]] = {".pdf"}

    @classmethod
    def extract_text(cls, path: Path | str) -> list[tuple[int, str]]:
        """Extract text sections as (page_number, text_content).

        Returns empty list if the file is unreadable, corrupt, empty, or encrypted.
        """
        target = Path(path)
        try:
            if not target.is_file() or target.stat().st_size == 0:
                return []
        except OSError:
            return []

        try:
            import pypdf
        except ImportError:
            return []

        try:
            reader = pypdf.PdfReader(str(target))
            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                    if decrypted == 0:
                        return []
                except Exception:  # noqa: BLE001 - corrupted or unsupported encryption keys
                    return []

            sections: list[tuple[int, str]] = []
            for page_num, page in enumerate(reader.pages, start=1):
                with contextlib.suppress(Exception):
                    raw_text = page.extract_text() or ""
                    cleaned = raw_text.strip()
                    if cleaned:
                        sections.append((page_num, cleaned))
            return sections
        except Exception:  # noqa: BLE001 - arbitrary parser errors on malformed PDF streams
            return []

    @classmethod
    def extract_full_text(cls, path: Path | str) -> str:
        """Extract all text concatenated with double newlines."""
        return "\n\n".join(text for _, text in cls.extract_text(path))


class DocxExtractor:
    """Extract plain text from Microsoft Word (.docx) documents."""

    supported_extensions: ClassVar[set[str]] = {".docx"}

    @classmethod
    def extract_text(cls, path: Path | str) -> list[tuple[int, str]]:
        """Extract text sections as (section_or_page_number, text_content).

        Returns empty list if the file is unreadable, corrupt, or empty.
        """
        target = Path(path)
        try:
            if not target.is_file() or target.stat().st_size == 0:
                return []
        except OSError:
            return []

        try:
            import docx
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError:
            return []

        try:
            doc = docx.Document(str(target))
            sections: list[tuple[int, str]] = []
            current_lines: list[str] = []
            section_num = 1

            def flush() -> None:
                nonlocal current_lines, section_num
                if current_lines:
                    sections.append((section_num, "\n".join(current_lines)))
                    current_lines = []
                    section_num += 1

            for child in doc.element.body:
                if child.tag.endswith("p"):
                    p = Paragraph(child, doc)
                    text = p.text.strip()
                    if text:
                        current_lines.append(text)
                    if p._p.pPr is not None and p._p.pPr.sectPr is not None:
                        flush()
                elif child.tag.endswith("tbl"):
                    tbl = Table(child, doc)
                    for row in tbl.rows:
                        row_cells = [
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        ]
                        if row_cells:
                            deduped: list[str] = []
                            for cell in row_cells:
                                if not deduped or deduped[-1] != cell:
                                    deduped.append(cell)
                            current_lines.append(" | ".join(deduped))

            flush()
            return sections
        except Exception:  # noqa: BLE001 - corrupted docx/zip archives or missing parts
            return []

    @classmethod
    def extract_full_text(cls, path: Path | str) -> str:
        """Extract all text concatenated with double newlines."""
        return "\n\n".join(text for _, text in cls.extract_text(path))


class PptxExtractor:
    """Extract plain text from Microsoft PowerPoint (.pptx) presentations."""

    supported_extensions: ClassVar[set[str]] = {".pptx"}

    @classmethod
    def extract_text(cls, path: Path | str) -> list[tuple[int, str]]:
        """Extract text sections as (slide_number, text_content).

        Returns empty list if the file is unreadable, corrupt, or empty.
        """
        target = Path(path)
        try:
            if not target.is_file() or target.stat().st_size == 0:
                return []
        except OSError:
            return []

        try:
            import pptx
        except ImportError:
            return []

        try:
            prs = pptx.Presentation(str(target))
            sections: list[tuple[int, str]] = []
            for slide_num, slide in enumerate(prs.slides, start=1):
                slide_lines: list[str] = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            line = paragraph.text.strip()
                            if line:
                                slide_lines.append(line)
                if slide.has_notes_slide:
                    notes_slide = slide.notes_slide
                    if notes_slide and notes_slide.notes_text_frame:
                        for paragraph in notes_slide.notes_text_frame.paragraphs:
                            note_line = paragraph.text.strip()
                            if note_line:
                                slide_lines.append(note_line)
                text = "\n".join(slide_lines).strip()
                if text:
                    sections.append((slide_num, text))
            return sections
        except Exception:  # noqa: BLE001 - corrupted pptx/zip archives or missing slides
            return []

    @classmethod
    def extract_full_text(cls, path: Path | str) -> str:
        """Extract all text concatenated with double newlines."""
        return "\n\n".join(text for _, text in cls.extract_text(path))


class DocumentExtractor:
    """Unified dispatcher for zero-GPU document text extraction."""

    supported_extensions: ClassVar[set[str]] = {".pdf", ".docx", ".pptx"}

    _extractors: ClassVar[
        dict[str, type[PdfExtractor | DocxExtractor | PptxExtractor]]
    ] = {
        ".pdf": PdfExtractor,
        ".docx": DocxExtractor,
        ".pptx": PptxExtractor,
    }

    @classmethod
    def get_extractor(
        cls, path: Path | str
    ) -> type[PdfExtractor | DocxExtractor | PptxExtractor] | None:
        """Return the extractor class appropriate for the given file extension."""
        suffix = Path(path).suffix.lower()
        return cls._extractors.get(suffix)

    @classmethod
    def extract_text(cls, path: Path | str) -> list[tuple[int, str]]:
        """Extract text sections from supported document format as (section_or_page, text)."""
        extractor = cls.get_extractor(path)
        if extractor is None:
            return []
        return extractor.extract_text(path)

    @classmethod
    def extract_full_text(cls, path: Path | str) -> str:
        """Extract all text from document as a single normalized string."""
        extractor = cls.get_extractor(path)
        if extractor is None:
            return ""
        return extractor.extract_full_text(path)
