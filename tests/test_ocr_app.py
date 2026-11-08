"""Tests for intel-npu-ocr CLI and batch document OCR pipeline."""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from intel_npu_tools import ocr, ocr_app
from intel_npu_tools.layout import WordToken

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def make_row(level, block, par, line, word, left, top, width, height, conf, text="", page=1):
    return "\t".join(
        str(part)
        for part in (level, page, block, par, line, word, left, top, width, height, conf, text)
    )


LEASE_TSV = "\n".join([
    HEADER,
    make_row(1, 0, 0, 0, 0, 0, 0, 800, 1000, -1),
    # Heading
    make_row(5, 1, 1, 1, 1, 200, 50, 140, 20, 95.0, "COMMERCIAL"),
    make_row(5, 1, 1, 1, 2, 350, 50, 60, 20, 95.0, "LEASE"),
    make_row(5, 1, 1, 1, 3, 420, 50, 130, 20, 95.0, "AGREEMENT"),
    # Key Values
    make_row(5, 1, 1, 2, 1, 50, 100, 70, 15, 92.0, "Lessor:"),
    make_row(5, 1, 1, 2, 2, 130, 100, 80, 15, 92.0, "BuildingQR"),
    make_row(5, 1, 1, 3, 1, 50, 125, 60, 15, 92.0, "Lessee:"),
    make_row(5, 1, 1, 3, 2, 120, 125, 75, 15, 92.0, "TechCorp"),
    # Multi-Column Table: Unit | Term | Monthly Rent | Deposit
    make_row(5, 1, 2, 1, 1, 50, 180, 40, 16, 95.0, "Unit"),
    make_row(5, 1, 2, 1, 2, 200, 180, 45, 16, 95.0, "Term"),
    make_row(5, 1, 2, 1, 3, 400, 180, 60, 16, 95.0, "Monthly"),
    make_row(5, 1, 2, 1, 4, 465, 180, 35, 16, 95.0, "Rent"),
    make_row(5, 1, 2, 1, 5, 600, 180, 60, 16, 95.0, "Deposit"),
    # Row 1: Suite 400 | 36 Months | $4,500.00 | $4,500.00
    make_row(5, 1, 2, 2, 1, 50, 210, 35, 15, 94.0, "Suite"),
    make_row(5, 1, 2, 2, 2, 90, 210, 25, 15, 94.0, "400"),
    make_row(5, 1, 2, 2, 3, 200, 210, 20, 15, 94.0, "36"),
    make_row(5, 1, 2, 2, 4, 225, 210, 50, 15, 94.0, "Months"),
    make_row(5, 1, 2, 2, 5, 400, 210, 70, 15, 94.0, "$4,500.00"),
    make_row(5, 1, 2, 2, 6, 600, 210, 70, 15, 94.0, "$4,500.00"),
    # Row 2: Suite 402 | 12 Months | $2,200.00 | $2,200.00
    make_row(5, 1, 2, 3, 1, 50, 235, 35, 15, 93.0, "Suite"),
    make_row(5, 1, 2, 3, 2, 90, 235, 25, 15, 93.0, "402"),
    make_row(5, 1, 2, 3, 3, 200, 235, 20, 15, 93.0, "12"),
    make_row(5, 1, 2, 3, 4, 225, 235, 50, 15, 93.0, "Months"),
    make_row(5, 1, 2, 3, 5, 400, 235, 70, 15, 93.0, "$2,200.00"),
    make_row(5, 1, 2, 3, 6, 600, 235, 70, 15, 93.0, "$2,200.00"),
])


def test_cli_argument_parsing():
    args = ocr_app.parse_args(["lease.pdf", "-o", "out.md", "--benchmark", "--format", "tables"])
    assert args.files == ["lease.pdf"]
    assert args.output == "out.md"
    assert args.benchmark is True
    assert args.format == "tables"


def test_ocr_document_to_markdown_with_pdf(tmp_path, monkeypatch):
    """Verify that a sample PDF file executes under 2 seconds per page and extracts tables."""
    # Create sample PDF using ps2pdf
    ps_file = tmp_path / "sample.ps"
    pdf_file = tmp_path / "sample_lease.pdf"
    ps_file.write_text(
        "%!PS\n"
        "/Helvetica-Bold findfont 14 scalefont setfont\n"
        "100 750 moveto (COMMERCIAL LEASE AGREEMENT) show\n"
        "/Helvetica findfont 10 scalefont setfont\n"
        "100 700 moveto (Unit    Term    Rent) show\n"
        "100 680 moveto (Suite 1 12 Mo   $1000) show\n"
        "showpage\n"
    )
    subprocess.run(["ps2pdf", str(ps_file), str(pdf_file)], check=True)

    # Monkeypatch tesseract_tsv to return LEASE_TSV fixture
    monkeypatch.setattr(ocr, "tesseract_tsv", lambda path: LEASE_TSV)

    start = time.perf_counter()
    md, meta = ocr.ocr_document_to_markdown(pdf_file)
    elapsed = time.perf_counter() - start

    # Acceptance Criteria Verification
    assert meta["pages"] == 1
    assert meta["seconds_per_page"] < 2.0, f"Execution took {meta['seconds_per_page']}s, expected < 2.0s"
    assert elapsed < 2.0, f"Total execution took {elapsed}s, expected < 2.0s"
    assert meta["tables"] >= 1
    assert "# COMMERCIAL LEASE AGREEMENT" in md
    assert "**Lessor:** BuildingQR" in md
    assert "| Unit      | Term      | Monthly Rent | Deposit   |" in md
    assert "| Suite 400 | 36 Months | $4,500.00    | $4,500.00 |" in md
    assert "| Suite 402 | 12 Months | $2,200.00    | $2,200.00 |" in md


def test_real_pdf_end_to_end_without_monkeypatch(tmp_path):
    """Verify end-to-end execution on real PDF file with table extraction in under 2s."""
    ps_file = tmp_path / "lease.ps"
    pdf_file = tmp_path / "real_lease.pdf"
    ps_file.write_text(
        "%!PS\n"
        "/Helvetica-Bold 14 selectfont\n"
        "100 750 moveto (COMMERCIAL LEASE AGREEMENT) show\n"
        "/Helvetica 10 selectfont\n"
        "100 700 moveto (Landlord: Acme Realty) show\n"
        "100 680 moveto (Tenant: John Smith) show\n"
        "100 640 moveto (Unit) show 180 640 moveto (Lease Term) show 300 640 moveto (Monthly Rent) show 450 640 moveto (Deposit) show\n"
        "100 620 moveto (Apt 4B) show 180 620 moveto (12 Months) show 300 620 moveto ($2,500.00) show 450 620 moveto ($2,500.00) show\n"
        "100 600 moveto (Apt 4C) show 180 600 moveto (24 Months) show 300 600 moveto ($2,800.00) show 450 600 moveto ($2,800.00) show\n"
        "showpage\n"
    )
    subprocess.run(["ps2pdf", str(ps_file), str(pdf_file)], check=True)

    start = time.perf_counter()
    md, meta = ocr.ocr_document_to_markdown(pdf_file)
    elapsed = time.perf_counter() - start

    assert meta["pages"] == 1
    assert meta["seconds_per_page"] < 2.0, f"Expected < 2s/page, got {meta['seconds_per_page']}s"
    assert meta["tables"] >= 1
    assert "# COMMERCIAL LEASE AGREEMENT" in md
    assert "**Landlord:** Acme Realty" in md
    assert "**Tenant:** John Smith" in md
    assert "| Unit   | Lease Term | Monthly Rent | Deposit" in md
    assert "| Apt 4B | 12 Months  | $2,500.00    | $2,500.00 |" in md
    assert "| Apt 4C | 24 Months  | $2,800.00    | $2,800.00 |" in md


def test_cli_main_batch_execution(tmp_path, monkeypatch, capsys):
    """Verify batch CLI execution on multiple files with output file generation."""
    # Create dummy images
    img1 = tmp_path / "lease1.png"
    img2 = tmp_path / "lease2.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")
    img2.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")

    monkeypatch.setattr(ocr, "tesseract_tsv", lambda path: LEASE_TSV)

    out_file = tmp_path / "output.md"
    ocr_app.main([str(img1), str(img2), "-o", str(out_file), "-b"])

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# COMMERCIAL LEASE AGREEMENT" in content
    assert "| Suite 400 | 36 Months | $4,500.00    | $4,500.00 |" in content


def test_cli_main_tables_only_format(tmp_path, monkeypatch, capsys):
    """Verify --format tables flag extracts exclusively markdown tables."""
    img = tmp_path / "doc.png"
    img.write_bytes(b"dummy")

    monkeypatch.setattr(ocr, "tesseract_tsv", lambda path: LEASE_TSV)

    ocr_app.main([str(img), "--format", "tables", "-q"])
    captured = capsys.readouterr()

    assert "| Unit      | Term      | Monthly Rent | Deposit   |" in captured.out
    assert "| Suite 400 | 36 Months | $4,500.00    | $4,500.00 |" in captured.out
    # Heading and key values should be excluded in tables-only mode
    assert "# COMMERCIAL LEASE AGREEMENT" not in captured.out
    assert "**Lessor:**" not in captured.out


def test_cli_main_json_format(tmp_path, monkeypatch, capsys):
    """Verify --format json outputs structured metadata and content."""
    img = tmp_path / "doc.png"
    img.write_bytes(b"dummy")

    monkeypatch.setattr(ocr, "tesseract_tsv", lambda path: LEASE_TSV)

    ocr_app.main([str(img), "--format", "json", "-q"])
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert "documents" in data
    assert len(data["documents"]) == 1
    assert data["documents"][0]["tables"] >= 1
    assert "# COMMERCIAL LEASE AGREEMENT" in data["content"]
