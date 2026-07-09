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
