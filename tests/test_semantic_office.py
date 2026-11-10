"""Integration tests for semantic indexing of office documents (.pdf, .docx, .pptx)."""

from pathlib import Path

import numpy as np
from conftest import create_docx, create_pdf, create_pptx

from intel_npu_tools.semantic import SemanticIndex


class FakePipeline:
    """Mock OpenVINO TextEmbeddingPipeline for zero-hardware testing."""

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = np.array(
            [
                lowered.count("financial"),
                lowered.count("presentation"),
                lowered.count("budget"),
                lowered.count("contract"),
                1.0,
            ],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        return (vector / (norm if norm > 0 else 1.0)).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_office_index(db_path: Path) -> SemanticIndex:
    return SemanticIndex(db_path, pipeline_factory=FakePipeline)


def test_office_document_indexing_and_search(tmp_path: Path):
    docs_dir = tmp_path / "office_docs"
    docs_dir.mkdir()

    # 1. PDF file
    pdf_path = docs_dir / "annual_report.pdf"
    pdf_path.write_bytes(create_pdf(["Annual financial review and balance sheet."]))

    # 2. DOCX file
    docx_path = docs_dir / "service_agreement.docx"
    docx_path.write_bytes(
        create_docx(["Approved budget for upcoming software development contract."])
    )

    # 3. PPTX file
    pptx_path = docs_dir / "strategy.pptx"
    pptx_path.write_bytes(
        create_pptx(
            [
                {
                    "title": "Keynote Presentation",
                    "body": "Strategic presentation roadmap for 2026.",
                }
            ]
        )
    )

    # 4. Standard text file
    txt_path = docs_dir / "notes.txt"
    txt_path.write_text("General non-office notes.")

    index = build_office_index(tmp_path / "index.sqlite3")

    # Index with include_office=True (default)
    result = index.index(str(docs_dir), include_office=True)
    assert result["files_seen"] == 4
    assert result["files_updated"] == 4
    assert result["chunks_added"] >= 4

    # Search for terms in PDF
    hits_pdf = index.search("financial", limit=2)
    assert len(hits_pdf) >= 1
    assert any("annual_report.pdf" in hit["path"] for hit in hits_pdf)

    # Search for terms in DOCX
    hits_docx = index.search("contract", limit=2)
    assert len(hits_docx) >= 1
    assert any("service_agreement.docx" in hit["path"] for hit in hits_docx)

    # Search for terms in PPTX
    hits_pptx = index.search("presentation", limit=2)
    assert len(hits_pptx) >= 1
    assert any("strategy.pptx" in hit["path"] for hit in hits_pptx)


def test_office_document_indexing_disabled(tmp_path: Path):
    docs_dir = tmp_path / "mixed_docs"
    docs_dir.mkdir()

    (docs_dir / "doc.pdf").write_bytes(create_pdf(["Financial statement"]))
    (docs_dir / "doc.docx").write_bytes(create_docx(["Budget contract"]))
    (docs_dir / "plain.txt").write_text("Financial plain text")

    index = build_office_index(tmp_path / "index_no_office.sqlite3")

    # Index with include_office=False
    result = index.index(str(docs_dir), include_office=False)
    assert result["files_seen"] == 1
    assert result["files_updated"] == 1

    hits = index.search("financial")
    assert len(hits) == 1
    assert hits[0]["path"].endswith("plain.txt")


def test_office_document_corrupt_resilience_during_index(tmp_path: Path):
    docs_dir = tmp_path / "corrupt_docs"
    docs_dir.mkdir()

    (docs_dir / "good.pdf").write_bytes(create_pdf(["Solid financial overview"]))
    (docs_dir / "broken.docx").write_bytes(b"garbage-content-not-docx")
    (docs_dir / "zero.pptx").write_bytes(b"")

    index = build_office_index(tmp_path / "index_resilience.sqlite3")

    # Indexing must not crash when encountering broken files
    result = index.index(str(docs_dir))
    assert result["files_seen"] == 3
    # Only good.pdf successfully chunks and updates
    assert result["files_updated"] == 1

    hits = index.search("financial")
    assert len(hits) == 1
    assert hits[0]["path"].endswith("good.pdf")
