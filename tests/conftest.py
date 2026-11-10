"""Keep every test away from the developer's own settings file.

Settings resolve from the environment first and the stored file second, so
without this a real `settings.json` under the user's data directory would leak
into the suite: a machine with `model_cache` enabled would fail the tests that
assert it is off by default, and the failure would look like a code defect
rather than local state.
"""

import io

import docx
import pptx
import pypdf
import pytest

from intel_npu_tools import config


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    for variable in config.ENVIRONMENT.values():
        monkeypatch.delenv(variable, raising=False)
    yield


def create_pdf(
    texts: list[str], password: str | None = None, blank_pages: int = 0
) -> bytes:
    """Helper to generate in-memory PDF fixtures with text or encryption."""
    writer = pypdf.PdfWriter()
    for text in texts:
        page_stream = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj
4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
5 0 obj << /Length {len(text) + 33} >> stream
BT /F1 24 Tf 100 700 Td ({text}) Tj ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000319 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
450
%%EOF"""
        reader = pypdf.PdfReader(io.BytesIO(page_stream.encode("latin1")))
        writer.add_page(reader.pages[0])

    for _ in range(blank_pages):
        writer.add_blank_page(width=100, height=100)

    if password:
        writer.encrypt(password)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_docx(
    paragraphs: list[str],
    table_rows: list[list[str]] | None = None,
    with_sections: bool = False,
) -> bytes:
    """Helper to generate in-memory Word document fixtures."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if with_sections:
        doc.add_section()
        doc.add_paragraph("Second section paragraph")
    if table_rows:
        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r_idx, row in enumerate(table_rows):
            for c_idx, cell_value in enumerate(row):
                table.cell(r_idx, c_idx).text = cell_value
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def create_pptx(slides_data: list[dict[str, str]]) -> bytes:
    """Helper to generate in-memory PowerPoint presentation fixtures."""
    prs = pptx.Presentation()
    for slide_info in slides_data:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        if "title" in slide_info:
            slide.shapes.title.text = slide_info["title"]
        if "body" in slide_info and len(slide.placeholders) > 1:
            slide.placeholders[1].text = slide_info["body"]
        if "note" in slide_info:
            slide.notes_slide.notes_text_frame.text = slide_info["note"]
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
