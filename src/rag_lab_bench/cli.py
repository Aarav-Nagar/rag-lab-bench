from __future__ import annotations

import argparse

from .chunking import DEFAULT_STRATEGIES, chunk_text, compare_strategies
from .retrieval import run_retrieval_checks
from .sample_data import load_sample_documents, load_sample_query_fixtures


def _format_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""

    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = " | ".join("-" * widths[header] for header in headers)
    body = [
        " | ".join(row[header].ljust(widths[header]) for header in headers)
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def build_demo_report() -> str:
    documents = load_sample_documents()
    rows = [report.as_row() for report in compare_strategies(documents, DEFAULT_STRATEGIES)]
    retrieval_rows = [
        check.as_row()
        for check in run_retrieval_checks(
            documents,
            load_sample_query_fixtures(),
            strategy=DEFAULT_STRATEGIES[1],
            top_k=3,
        )
    ]
    document_summary = "\n".join(
        f"- {document.doc_id}: {document.word_count} words, tags={','.join(document.tags)}"
        for document in documents
    )
    return (
        "RAG Lab Bench sample corpus\n"
        f"{document_summary}\n\n"
        "Chunking strategy comparison\n"
        f"{_format_table(rows)}\n\n"
        "Deterministic retrieval fixture checks\n"
        f"{_format_table(retrieval_rows)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare deterministic chunking strategies for RAG experiments."
    )
    parser.add_argument("--demo", action="store_true", help="Run the sample benchmark.")
    parser.add_argument("--text", help="Chunk inline text instead of the sample corpus.")
    parser.add_argument("--size", type=int, default=40, help="Word chunk size for --text.")
    args = parser.parse_args()

    if args.text:
        for chunk in chunk_text(args.text, size=args.size):
            print(chunk)
        return

    if args.demo:
        print(build_demo_report())
        return

    parser.print_help()


if __name__ == "__main__":
    main()
