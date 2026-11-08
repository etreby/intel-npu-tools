# User Story: Batch Document OCR & Markdown Table Extraction

- **Task**: ETR-1425
- **Date**: 2026-11-08
- **Author**: Senior Full-Stack & UX Engineer
- **Component**: `intel-npu-tools` (`intel-npu-ocr`)

## User & Goal
- **User**: Property administrator / office manager
- **Goal**: Scan PDF documents (lease agreements, invoices, payment receipts) and images into structured GitHub-flavored markdown with automated multi-column table extraction using local hardware/NPU acceleration.

## User Flow
1. **Command Execution**:
   The user runs `intel-npu-ocr sample_lease.pdf -o lease_summary.md` or batch scans multiple receipts:
   ```bash
   intel-npu-ocr receipt1.pdf receipt2.png -o receipts_digest.md
   ```
2. **Page & Layout Ingestion**:
   - The CLI renders PDF pages in tens of milliseconds per page using poppler `pdftoppm`.
   - The OCR engine processes layout coordinates using Tesseract or OpenVINO NPU detection backbone.
3. **Table & Layout Recognition**:
   - Words are grouped into reading-order lines.
   - Column boundaries are identified via inter-word spacing analysis.
   - Multi-column tables (e.g., Unit / Term / Rent / Deposit) are detected and aligned into clean GitHub-flavored markdown tables with padded columns.
   - Headings and key-value fields (e.g. `**Landlord:** Acme Properties`) are structured cleanly.
4. **Output Formats**:
   - Standard markdown (`--format markdown`)
   - Tables only (`--format tables`)
   - Structured JSON (`--format json`)

## Verification & Acceptance Evidence
- **Acceptance Criteria 1**: Accurately extracts multi-column tables from sample lease agreements.
  - Verified by `tests/test_layout.py::test_lease_agreement_table_extraction` and `tests/test_ocr_app.py::test_real_pdf_end_to_end_without_monkeypatch`.
- **Acceptance Criteria 2**: CLI command `intel-npu-ocr <file.pdf>` executes in under 2 seconds per page.
  - Verified benchmark output: 1 page in 0.41s - 0.57s (< 2.0s per page).
- **Automated Test Results**: 162 unit and integration tests passing in 1.98s.
