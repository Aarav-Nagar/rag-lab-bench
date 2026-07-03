from rag_lab_bench.cli import chunk_text


def test_chunk_text_uses_requested_size() -> None:
    assert chunk_text("one two three four five", 2) == ["one two", "three four", "five"]

