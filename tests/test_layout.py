"""Tests for document layout detection and multi-column table extraction."""

import pytest
from intel_npu_tools.layout import (
    WordToken,
    group_words_into_lines,
    segment_line_cells,
    detect_layout_blocks,
    render_markdown,
    tsv_to_markdown,
)

HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def make_row(level, block, par, line, word, left, top, width, height, conf, text="", page=1):
    return "\t".join(
        str(part)
        for part in (level, page, block, par, line, word, left, top, width, height, conf, text)
    )


def test_lease_agreement_table_extraction():
    """Verify accurate extraction of multi-column tables from a sample residential lease agreement."""
    rows = [
        HEADER,
        make_row(1, 0, 0, 0, 0, 0, 0, 800, 1000, -1),
        # Heading: RESIDENTIAL LEASE AGREEMENT (top=50)
        make_row(5, 1, 1, 1, 1, 200, 50, 140, 20, 95.0, "RESIDENTIAL"),
        make_row(5, 1, 1, 1, 2, 350, 50, 60, 20, 95.0, "LEASE"),
        make_row(5, 1, 1, 1, 3, 420, 50, 130, 20, 95.0, "AGREEMENT"),
        # Key-Value: Landlord: Acme Properties (top=100)
        make_row(5, 1, 1, 2, 1, 50, 100, 70, 15, 92.0, "Landlord:"),
        make_row(5, 1, 1, 2, 2, 130, 100, 45, 15, 92.0, "Acme"),
        make_row(5, 1, 1, 2, 3, 180, 100, 80, 15, 92.0, "Properties"),
        # Key-Value: Tenant: John Doe (top=125)
        make_row(5, 1, 1, 3, 1, 50, 125, 60, 15, 92.0, "Tenant:"),
        make_row(5, 1, 1, 3, 2, 120, 125, 40, 15, 92.0, "John"),
        make_row(5, 1, 1, 3, 3, 165, 125, 35, 15, 92.0, "Doe"),
        # Table Header: Unit No. | Lease Term | Monthly Rent | Deposit (top=180)
        # Col 0: 50..120, Col 1: 200..300, Col 2: 400..510, Col 3: 600..680
        make_row(5, 1, 2, 1, 1, 50, 180, 40, 16, 95.0, "Unit"),
        make_row(5, 1, 2, 1, 2, 95, 180, 25, 16, 95.0, "No."),
        make_row(5, 1, 2, 1, 3, 200, 180, 45, 16, 95.0, "Lease"),
        make_row(5, 1, 2, 1, 4, 250, 180, 40, 16, 95.0, "Term"),
        make_row(5, 1, 2, 1, 5, 400, 180, 60, 16, 95.0, "Monthly"),
        make_row(5, 1, 2, 1, 6, 465, 180, 35, 16, 95.0, "Rent"),
        make_row(5, 1, 2, 1, 7, 600, 180, 60, 16, 95.0, "Deposit"),
        # Table Row 1: Apt 101 | 12 Months | $1,500.00 | $1,500.00 (top=210)
        make_row(5, 1, 2, 2, 1, 50, 210, 30, 15, 94.0, "Apt"),
        make_row(5, 1, 2, 2, 2, 85, 210, 25, 15, 94.0, "101"),
        make_row(5, 1, 2, 2, 3, 200, 210, 20, 15, 94.0, "12"),
        make_row(5, 1, 2, 2, 4, 225, 210, 50, 15, 94.0, "Months"),
        make_row(5, 1, 2, 2, 5, 400, 210, 70, 15, 94.0, "$1,500.00"),
        make_row(5, 1, 2, 2, 6, 600, 210, 70, 15, 94.0, "$1,500.00"),
        # Table Row 2: Apt 102 | 24 Months | $1,800.00 | $1,800.00 (top=235)
        make_row(5, 1, 2, 3, 1, 50, 235, 30, 15, 93.0, "Apt"),
        make_row(5, 1, 2, 3, 2, 85, 235, 25, 15, 93.0, "102"),
        make_row(5, 1, 2, 3, 3, 200, 235, 20, 15, 93.0, "24"),
        make_row(5, 1, 2, 3, 4, 225, 235, 50, 15, 93.0, "Months"),
        make_row(5, 1, 2, 3, 5, 400, 235, 70, 15, 93.0, "$1,800.00"),
        make_row(5, 1, 2, 3, 6, 600, 235, 70, 15, 93.0, "$1,800.00"),
        # Clause Heading: 1. Payment Terms (top=320)
        make_row(5, 1, 3, 1, 1, 50, 320, 15, 15, 90.0, "1."),
        make_row(5, 1, 3, 1, 2, 70, 320, 65, 15, 90.0, "Payment"),
        make_row(5, 1, 3, 1, 3, 140, 320, 45, 15, 90.0, "Terms"),
        # Paragraph text (top=350)
        make_row(5, 1, 3, 2, 1, 50, 350, 35, 14, 91.0, "Rent"),
        make_row(5, 1, 3, 2, 2, 90, 350, 15, 14, 91.0, "is"),
        make_row(5, 1, 3, 2, 3, 110, 350, 25, 14, 91.0, "due"),
        make_row(5, 1, 3, 2, 4, 140, 350, 20, 14, 91.0, "on"),
        make_row(5, 1, 3, 2, 5, 165, 350, 25, 14, 91.0, "the"),
        make_row(5, 1, 3, 2, 6, 195, 350, 35, 14, 91.0, "first"),
        make_row(5, 1, 3, 2, 7, 235, 350, 40, 14, 91.0, "day."),
    ]
    tsv = "\n".join(rows)
    md = tsv_to_markdown(tsv)

    assert "# RESIDENTIAL LEASE AGREEMENT" in md
    assert "**Landlord:** Acme Properties" in md
    assert "**Tenant:** John Doe" in md
    assert "| Unit No. | Lease Term | Monthly Rent | Deposit   |" in md
    assert "| Apt 101  | 12 Months  | $1,500.00    | $1,500.00 |" in md
    assert "| Apt 102  | 24 Months  | $1,800.00    | $1,800.00 |" in md
    assert "## 1. Payment Terms" in md
    assert "Rent is due on the first day." in md


def test_payment_receipt_table_extraction():
    """Verify receipt extraction with financial line items and subtotal."""
    rows = [
        HEADER,
        make_row(1, 0, 0, 0, 0, 0, 0, 600, 800, -1),
        # Heading
        make_row(5, 1, 1, 1, 1, 150, 40, 70, 18, 98.0, "OFFICIAL"),
        make_row(5, 1, 1, 1, 2, 230, 40, 75, 18, 98.0, "RECEIPT"),
        # Key values
        make_row(5, 1, 1, 2, 1, 50, 80, 60, 14, 95.0, "Receipt:"),
        make_row(5, 1, 1, 2, 2, 120, 80, 80, 14, 95.0, "REC-1049"),
        # Table Header: Item Description | Qty | Rate | Amount (top=140)
        make_row(5, 1, 2, 1, 1, 50, 140, 35, 15, 95.0, "Item"),
        make_row(5, 1, 2, 1, 2, 90, 140, 80, 15, 95.0, "Description"),
        make_row(5, 1, 2, 1, 3, 260, 140, 25, 15, 95.0, "Qty"),
        make_row(5, 1, 2, 1, 4, 340, 140, 30, 15, 95.0, "Rate"),
        make_row(5, 1, 2, 1, 5, 450, 140, 50, 15, 95.0, "Amount"),
        # Row 1: Water Bill | 1 | $45.00 | $45.00 (top=170)
        make_row(5, 1, 2, 2, 1, 50, 170, 40, 14, 94.0, "Water"),
        make_row(5, 1, 2, 2, 2, 95, 170, 25, 14, 94.0, "Bill"),
        make_row(5, 1, 2, 2, 3, 260, 170, 10, 14, 94.0, "1"),
        make_row(5, 1, 2, 2, 4, 340, 170, 45, 14, 94.0, "$45.00"),
        make_row(5, 1, 2, 2, 5, 450, 170, 45, 14, 94.0, "$45.00"),
        # Row 2: Maintenance Fee | 1 | $120.00 | $120.00 (top=195)
        make_row(5, 1, 2, 3, 1, 50, 195, 85, 14, 94.0, "Maintenance"),
        make_row(5, 1, 2, 3, 2, 140, 195, 25, 14, 94.0, "Fee"),
        make_row(5, 1, 2, 3, 3, 260, 195, 10, 14, 94.0, "1"),
        make_row(5, 1, 2, 3, 4, 340, 195, 55, 14, 94.0, "$120.00"),
        make_row(5, 1, 2, 3, 5, 450, 195, 55, 14, 94.0, "$120.00"),
    ]
    tsv = "\n".join(rows)
    md = tsv_to_markdown(tsv)

    assert "# OFFICIAL RECEIPT" in md
    assert "**Receipt:** REC-1049" in md
    assert "| Item Description | Qty | Rate    | Amount  |" in md
    assert "| Water Bill       | 1   | $45.00  | $45.00  |" in md
    assert "| Maintenance Fee  | 1   | $120.00 | $120.00 |" in md


def test_empty_tsv_returns_empty_string():
    assert tsv_to_markdown("") == ""
    assert tsv_to_markdown(HEADER) == ""
