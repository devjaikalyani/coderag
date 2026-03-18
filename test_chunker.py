"""
test_chunker.py
---------------
Unit tests for the code-aware chunker.
Run with: pytest tests/test_chunker.py -v
"""

import pytest
from src.ingestion.chunker import chunk_text, Chunk

PYTHON_CODE = '''
import os
from pathlib import Path

class FileManager:
    """Manages file operations."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def read_file(self, filename: str) -> str:
        path = self.base_path / filename
        return path.read_text(encoding="utf-8")

    def write_file(self, filename: str, content: str) -> None:
        path = self.base_path / filename
        path.write_text(content, encoding="utf-8")


def process_directory(path: str) -> list:
    result = []
    for root, dirs, files in os.walk(path):
        for f in files:
            result.append(os.path.join(root, f))
    return result
'''

MARKDOWN_DOC = """
# Introduction

This is the introduction section.

## Installation

Install the package using pip:

```bash
pip install mypackage
```

## Usage

Here is how to use the package.
"""


class TestChunker:

    def test_basic_python_chunking(self):
        chunks = chunk_text(PYTHON_CODE, source="test.py")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_required_fields(self):
        chunks = chunk_text(PYTHON_CODE, source="test.py")
        for c in chunks:
            assert c.text
            assert c.source == "test.py"
            assert c.chunk_id.startswith("test.py::")
            assert c.language == "python"

    def test_chunk_text_not_empty(self):
        chunks = chunk_text(PYTHON_CODE, source="test.py")
        for c in chunks:
            assert c.text.strip()

    def test_markdown_chunking(self):
        chunks = chunk_text(MARKDOWN_DOC, source="README.md")
        assert len(chunks) > 0
        assert all(c.language == "markdown" for c in chunks)

    def test_language_detection(self):
        cases = [
            ("code.py", "python"),
            ("app.js", "javascript"),
            ("main.go", "go"),
            ("lib.rs", "rust"),
            ("README.md", "markdown"),
        ]
        for source, expected_lang in cases:
            chunks = chunk_text("def foo(): pass", source=source)
            assert chunks[0].language == expected_lang

    def test_chunk_ids_unique(self):
        chunks = chunk_text(PYTHON_CODE, source="test.py")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_single_line_text(self):
        chunks = chunk_text("print('hello world')", source="hello.py")
        assert len(chunks) >= 1
        assert "hello world" in chunks[0].text

    def test_max_tokens_respected(self):
        large_text = "\n".join([f"def function_{i}():\n    return {i}\n" for i in range(200)])
        chunks = chunk_text(large_text, source="large.py", max_tokens=100)
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        for c in chunks:
            assert len(enc.encode(c.text)) <= 150
