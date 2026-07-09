from __future__ import annotations

from collections.abc import Iterable

from .models import Chunk, ChunkingStrategy, SourceDocument, StrategyReport


DEFAULT_STRATEGIES: tuple[ChunkingStrategy, ...] = (
    ChunkingStrategy(name="tight-context", target_words=24, overlap_words=4),
    ChunkingStrategy(name="balanced", target_words=40, overlap_words=8),
    ChunkingStrategy(name="wide-context", target_words=72, overlap_words=12),
)


def ingest_documents(records: Iterable[dict[str, object]]) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    seen_ids: set[str] = set()

    for record in records:
        doc_id = str(record.get("doc_id", "")).strip()
        if doc_id in seen_ids:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        seen_ids.add(doc_id)

        raw_tags = record.get("tags", ())
        if isinstance(raw_tags, str):
            tags = (raw_tags,)
        else:
            tags = tuple(str(tag) for tag in raw_tags)

        documents.append(
            SourceDocument(
                doc_id=doc_id,
                title=str(record.get("title", "")).strip(),
                text=str(record.get("text", "")).strip(),
                tags=tags,
            )
        )

    return documents


def chunk_document(document: SourceDocument, strategy: ChunkingStrategy) -> list[Chunk]:
    words = document.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = strategy.target_words - strategy.overlap_words
    start = 0

    while start < len(words):
        end = min(start + strategy.target_words, len(words))
        chunk_words = words[start:end]
        chunk_number = len(chunks) + 1
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{strategy.name}:{chunk_number:03d}",
                doc_id=document.doc_id,
                strategy=strategy.name,
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
                tags=document.tags,
            )
        )
        if end == len(words):
            break
        start += step

    return chunks


def chunk_corpus(
    documents: Iterable[SourceDocument],
    strategy: ChunkingStrategy,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy))
    return chunks


def compare_strategies(
    documents: Iterable[SourceDocument],
    strategies: Iterable[ChunkingStrategy] = DEFAULT_STRATEGIES,
) -> list[StrategyReport]:
    document_list = list(documents)
    reports: list[StrategyReport] = []

    for strategy in strategies:
        chunks = chunk_corpus(document_list, strategy)
        counts = [chunk.word_count for chunk in chunks]
        if not counts:
            reports.append(StrategyReport(strategy, 0, 0.0, 0, 0))
            continue
        reports.append(
            StrategyReport(
                strategy=strategy,
                chunk_count=len(chunks),
                average_words=sum(counts) / len(counts),
                min_words=min(counts),
                max_words=max(counts),
            )
        )

    return reports


def chunk_text(text: str, size: int) -> list[str]:
    document = SourceDocument(doc_id="inline", title="Inline text", text=text)
    strategy = ChunkingStrategy(name="fixed", target_words=size)
    return [chunk.text for chunk in chunk_document(document, strategy)]
