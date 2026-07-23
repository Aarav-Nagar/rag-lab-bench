from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceDocument:
    """A small source record ready for deterministic RAG experiments."""

    doc_id: str
    title: str
    text: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("doc_id is required")
        if not self.title.strip():
            raise ValueError("title is required")
        if not self.text.strip():
            raise ValueError("text is required")

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    target_words: int
    overlap_words: int = 0

    def __post_init__(self) -> None:
        if self.target_words < 1:
            raise ValueError("target_words must be at least 1")
        if self.overlap_words < 0:
            raise ValueError("overlap_words cannot be negative")
        if self.overlap_words >= self.target_words:
            raise ValueError("overlap_words must be smaller than target_words")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    start_word: int
    end_word: int
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class StrategyReport:
    strategy: ChunkingStrategy
    chunk_count: int
    average_words: float
    min_words: int
    max_words: int

    def as_row(self) -> dict[str, str]:
        return {
            "strategy": self.strategy.name,
            "target": str(self.strategy.target_words),
            "overlap": str(self.strategy.overlap_words),
            "chunks": str(self.chunk_count),
            "avg_words": f"{self.average_words:.1f}",
            "min_words": str(self.min_words),
            "max_words": str(self.max_words),
        }


@dataclass(frozen=True)
class QueryFixture:
    """A deterministic retrieval question with expected supporting documents."""

    query_id: str
    question: str
    expected_doc_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id is required")
        if not self.question.strip():
            raise ValueError("question is required")
        if not self.expected_doc_ids:
            raise ValueError("expected_doc_ids must include at least one document id")
        if any(not doc_id.strip() for doc_id in self.expected_doc_ids):
            raise ValueError("expected_doc_ids cannot include empty document ids")


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...]

    def as_row(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "score": f"{self.score:.1f}",
            "matches": ",".join(self.matched_terms),
        }


@dataclass(frozen=True)
class RetrievalCheck:
    fixture: QueryFixture
    strategy: str
    top_hits: tuple[RetrievalHit, ...]
    top_k: int

    @property
    def first_expected_rank(self) -> int | None:
        expected = set(self.fixture.expected_doc_ids)
        for index, hit in enumerate(self.top_hits, start=1):
            if hit.chunk.doc_id in expected:
                return index
        return None

    @property
    def expected_found(self) -> bool:
        return self.first_expected_rank is not None

    def as_row(self) -> dict[str, str]:
        top_hit = self.top_hits[0] if self.top_hits else None
        return {
            "query": self.fixture.query_id,
            "strategy": self.strategy,
            "expected": ",".join(self.fixture.expected_doc_ids),
            "top_doc": top_hit.chunk.doc_id if top_hit else "none",
            "found": "yes" if self.expected_found else "no",
            "rank": str(self.first_expected_rank or "-"),
            "score": f"{top_hit.score:.1f}" if top_hit else "0.0",
        }
