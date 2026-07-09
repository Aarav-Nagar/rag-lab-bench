# RAG Lab Bench

RAG Lab Bench is a small, deterministic playground for comparing retrieval-augmented generation setup choices before wiring in an LLM. It starts with the pieces that usually decide whether a RAG answer can be trusted: source ingestion, chunk size, overlap, and visible evidence boundaries.

## What is included

- A typed `SourceDocument` ingestion model for repeatable benchmark fixtures.
- Overlap-aware chunking strategies with stable chunk identifiers.
- A built-in campus support sample corpus and a matching JSONL file in `samples/`.
- A CLI demo that compares chunk counts and word distributions across strategies.
- Unit tests for ingestion validation, chunk windows, and report generation.

## Quick Start

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m rag_lab_bench --demo
```

Example output:

```text
RAG Lab Bench sample corpus
- campus-ai-policy: 48 words, tags=policy,education
- library-retrieval-notes: 50 words, tags=retrieval,library
- student-advising-faq: 46 words, tags=faq,advising

Chunking strategy comparison
strategy      | target | overlap | chunks | avg_words | min_words | max_words
------------- | ------ | ------- | ------ | --------- | --------- | ---------
tight-context | 24     | 4       | 9      | 18.7      | 6         | 24
balanced      | 40     | 8       | 6      | 28.0      | 14        | 40
wide-context  | 72     | 12      | 3      | 48.0      | 46        | 50
```

## Ingestion Model

Each fixture is represented as a `SourceDocument`:

```python
from rag_lab_bench import ingest_documents

documents = ingest_documents([
    {
        "doc_id": "policy-001",
        "title": "AI Use Policy",
        "text": "Students must cite meaningful AI assistance in final work.",
        "tags": ["policy"],
    }
])
```

The ingestion helper rejects missing identifiers, empty text, and duplicate `doc_id` values so benchmark runs cannot silently collapse documents.

## Chunking Comparison Design

The first benchmark compares three deterministic strategies:

| Strategy | Target words | Overlap words | Best for |
| --- | ---: | ---: | --- |
| `tight-context` | 24 | 4 | Precise retrieval for narrow questions. |
| `balanced` | 40 | 8 | General-purpose retrieval with some context carryover. |
| `wide-context` | 72 | 12 | Definition-heavy sources where surrounding caveats matter. |

Chunks preserve document ids, strategy names, start/end word offsets, tags, and stable chunk ids. That metadata is the contract for later retrieval, groundedness, and citation-hit metrics.

## Sample Dataset

The sample corpus models a small campus-support knowledge base:

- `campus-ai-policy`: acceptable AI use and authorship checks.
- `library-retrieval-notes`: how passages are ranked and cited.
- `student-advising-faq`: prerequisite, waitlist, and petition guidance.

The same records are available in `samples/campus_support_corpus.jsonl` for future loader and retrieval tests.

## Development

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m pytest -q
```
