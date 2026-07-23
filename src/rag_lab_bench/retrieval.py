from __future__ import annotations

import re
from collections.abc import Iterable

from .chunking import DEFAULT_STRATEGIES, chunk_corpus
from .models import Chunk, ChunkingStrategy, QueryFixture, RetrievalCheck, RetrievalHit, SourceDocument


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "or",
    "should",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def content_terms(text: str) -> tuple[str, ...]:
    return tuple(term for term in tokenize(text) if term not in STOPWORDS)


def score_chunk(question: str, chunk: Chunk) -> RetrievalHit:
    query_terms = set(content_terms(question))
    chunk_terms = set(content_terms(chunk.text))
    tag_terms = set(content_terms(" ".join(chunk.tags)))
    id_terms = set(content_terms(chunk.doc_id.replace("-", " ")))

    matched_body_terms = query_terms & chunk_terms
    matched_metadata_terms = query_terms & (tag_terms | id_terms)
    matched_terms = tuple(sorted(matched_body_terms | matched_metadata_terms))
    score = (len(matched_body_terms) * 2.0) + (len(matched_metadata_terms) * 0.5)

    return RetrievalHit(chunk=chunk, score=score, matched_terms=matched_terms)


def retrieve_chunks(
    question: str,
    chunks: Iterable[Chunk],
    top_k: int = 3,
) -> list[RetrievalHit]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    hits = [score_chunk(question, chunk) for chunk in chunks]
    ranked = [hit for hit in hits if hit.score > 0]
    ranked.sort(
        key=lambda hit: (
            -hit.score,
            hit.chunk.doc_id,
            hit.chunk.start_word,
            hit.chunk.chunk_id,
        )
    )
    return ranked[:top_k]


def run_retrieval_checks(
    documents: Iterable[SourceDocument],
    fixtures: Iterable[QueryFixture],
    strategy: ChunkingStrategy = DEFAULT_STRATEGIES[1],
    top_k: int = 3,
) -> list[RetrievalCheck]:
    chunks = chunk_corpus(documents, strategy)
    return [
        RetrievalCheck(
            fixture=fixture,
            strategy=strategy.name,
            top_hits=tuple(retrieve_chunks(fixture.question, chunks, top_k=top_k)),
            top_k=top_k,
        )
        for fixture in fixtures
    ]
