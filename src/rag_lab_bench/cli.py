from __future__ import annotations

import argparse


def chunk_text(text: str, size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[index:index + size]) for index in range(0, len(words), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run a small chunking demo.")
    args = parser.parse_args()

    if args.demo:
        sample = "RAG systems work best when retrieval evidence is visible and measurable."
        print(chunk_text(sample, size=5))


if __name__ == "__main__":
    main()

