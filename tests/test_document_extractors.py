"""Unit tests for zero-GPU document text extractors (PDF, DOCX, PPTX)."""

from pathlib import Path

from conftest import create_docx, create_pdf, create_pptx

from intel_npu_tools.extractors import (
    DocumentExtractor,
    DocxExtractor,
    PdfExtractor,
    PptxExtractor,
)


def test_pdf_extractor_single_and_multipage(tmp_path: Path):
    single_pdf = tmp_path / "single.pdf"
    single_pdf.write_bytes(create_pdf(["First page text"]))

    sections = PdfExtractor.extract_text(single_pdf)
    assert len(sections) == 1
    assert sections[0] == (1, "First page text")

    multi_pdf = tmp_path / "multi.pdf"
    multi_pdf.write_bytes(create_pdf(["Page 1 content", "Page 2 content"]))

    multi_sections = PdfExtractor.extract_text(multi_pdf)
    assert len(multi_sections) == 2
    assert multi_sections[0] == (1, "Page 1 content")
    assert multi_sections[1] == (2, "Page 2 content")
    assert (
        PdfExtractor.extract_full_text(multi_pdf) == "Page 1 content\n\nPage 2 content"
    )


def test_pdf_extractor_blank_or_scanned(tmp_path: Path):
    blank_pdf = tmp_path / "blank.pdf"
    blank_pdf.write_bytes(create_pdf([], blank_pages=2))

    sections = PdfExtractor.extract_text(blank_pdf)
    assert sections == []
    assert PdfExtractor.extract_full_text(blank_pdf) == ""


def test_pdf_extractor_encrypted_handling(tmp_path: Path):
    enc_pdf = tmp_path / "encrypted.pdf"
    enc_pdf.write_bytes(
        create_pdf(["Confidential text"], password="super-secret-password")
    )

    # Must return empty list gracefully without throwing an exception
    sections = PdfExtractor.extract_text(enc_pdf)
    assert sections == []
    assert PdfExtractor.extract_full_text(enc_pdf) == ""


def test_docx_extractor_paragraphs_and_tables(tmp_path: Path):
    sample_docx = tmp_path / "sample.docx"
    sample_docx.write_bytes(
        create_docx(
            paragraphs=[
                "Quarterly Report Overview",
                "Key performance metrics exceed targets.",
            ],
            table_rows=[["Metric", "Value"], ["Revenue", "$12.5M"]],
        )
    )

    sections = DocxExtractor.extract_text(sample_docx)
    assert len(sections) >= 1
    full_text = DocxExtractor.extract_full_text(sample_docx)
    assert "Quarterly Report Overview" in full_text
    assert "Key performance metrics exceed targets." in full_text
    assert "Revenue | $12.5M" in full_text


def test_docx_extractor_sections(tmp_path: Path):
    sectioned_docx = tmp_path / "sections.docx"
    sectioned_docx.write_bytes(
        create_docx(
            paragraphs=["First section text"],
            with_sections=True,
        )
    )

    sections = DocxExtractor.extract_text(sectioned_docx)
    assert len(sections) == 2
    assert sections[0][0] == 1
    assert "First section text" in sections[0][1]
    assert sections[1][0] == 2
    assert "Second section paragraph" in sections[1][1]


def test_pptx_extractor_slides_and_notes(tmp_path: Path):
    sample_pptx = tmp_path / "presentation.pptx"
    sample_pptx.write_bytes(
        create_pptx(
            [
                {
                    "title": "Introduction to AI",
                    "body": "Edge AI with Intel NPU",
                    "note": "Welcome attendees",
                },
                {
                    "title": "Performance Benchmarks",
                    "body": "Sub-100ms inference on Arrow Lake",
                    "note": "Highlight power efficiency",
                },
            ]
        )
    )

    sections = PptxExtractor.extract_text(sample_pptx)
    assert len(sections) == 2
    assert sections[0][0] == 1
    assert "Introduction to AI" in sections[0][1]
    assert "Edge AI with Intel NPU" in sections[0][1]
    assert "Welcome attendees" in sections[0][1]

    assert sections[1][0] == 2
    assert "Performance Benchmarks" in sections[1][1]
    assert "Sub-100ms inference on Arrow Lake" in sections[1][1]
    assert "Highlight power efficiency" in sections[1][1]

    full_text = PptxExtractor.extract_full_text(sample_pptx)
    assert "Welcome attendees" in full_text
    assert "Performance Benchmarks" in full_text


def test_corrupt_and_empty_files_resilience(tmp_path: Path):
    for ext, extractor in [
        (".pdf", PdfExtractor),
        (".docx", DocxExtractor),
        (".pptx", PptxExtractor),
    ]:
        zero_file = tmp_path / f"zero{ext}"
        zero_file.write_bytes(b"")
        assert extractor.extract_text(zero_file) == []
        assert extractor.extract_full_text(zero_file) == ""

        corrupt_file = tmp_path / f"corrupt{ext}"
        corrupt_file.write_bytes(b"not-a-valid-archive-or-binary-format")
        assert extractor.extract_text(corrupt_file) == []
        assert extractor.extract_full_text(corrupt_file) == ""

    # Non-existent file
    missing = tmp_path / "missing.pdf"
    assert PdfExtractor.extract_text(missing) == []
    assert DocumentExtractor.extract_text(missing) == []


def test_document_extractor_unified_dispatcher(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(create_pdf(["Unified PDF text"]))

    docx_file = tmp_path / "doc.docx"
    docx_file.write_bytes(create_docx(["Unified DOCX text"]))

    pptx_file = tmp_path / "doc.pptx"
    pptx_file.write_bytes(create_pptx([{"title": "Unified PPTX text"}]))

    unsupported = tmp_path / "doc.unknown"
    unsupported.write_text("Hello")

    assert DocumentExtractor.get_extractor(pdf) is PdfExtractor
    assert DocumentExtractor.get_extractor(docx_file) is DocxExtractor
    assert DocumentExtractor.get_extractor(pptx_file) is PptxExtractor
    assert DocumentExtractor.get_extractor(unsupported) is None

    assert "Unified PDF text" in DocumentExtractor.extract_full_text(pdf)
    assert "Unified DOCX text" in DocumentExtractor.extract_full_text(docx_file)
    assert "Unified PPTX text" in DocumentExtractor.extract_full_text(pptx_file)
    assert DocumentExtractor.extract_full_text(unsupported) == ""
    assert DocumentExtractor.extract_text(unsupported) == []
