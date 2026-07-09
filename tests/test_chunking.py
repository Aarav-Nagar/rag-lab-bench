import pytest

from rag_lab_bench import (
    ChunkingStrategy,
    SourceDocument,
    chunk_document,
    chunk_text,
    compare_strategies,
    ingest_documents,
)
from rag_lab_bench.sample_data import load_sample_documents


def test_chunk_text_uses_requested_size() -> None:
    assert chunk_text("one two three four five", 2) == ["one two", "three four", "five"]


def test_ingest_documents_rejects_duplicate_ids() -> None:
    records = [
        {"doc_id": "faq", "title": "FAQ", "text": "First record."},
        {"doc_id": "faq", "title": "FAQ copy", "text": "Second record."},
    ]

    with pytest.raises(ValueError, match="duplicate doc_id"):
        ingest_documents(records)


def test_chunk_document_keeps_overlap_offsets() -> None:
    document = SourceDocument(
        doc_id="guide",
        title="Guide",
        text="one two three four five six seven eight nine ten",
        tags=("demo",),
    )
    strategy = ChunkingStrategy(name="small", target_words=4, overlap_words=1)

    chunks = chunk_document(document, strategy)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert [(chunk.start_word, chunk.end_word) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]
    assert chunks[0].chunk_id == "guide:small:001"
    assert chunks[0].tags == ("demo",)


def test_compare_strategies_reports_all_defaults() -> None:
    reports = compare_strategies(load_sample_documents())

    assert [report.strategy.name for report in reports] == [
        "tight-context",
        "balanced",
        "wide-context",
    ]
    assert all(report.chunk_count > 0 for report in reports)
    assert reports[0].chunk_count >= reports[-1].chunk_count
