"""
chunker.py
----------
Splits source code / docs into overlapping chunks.
Strategy per file type:
  - Python/JS/TS/Go/Rust: split at class/function boundaries
  - HTML/CSS/EJS: keep whole file if small, sliding window if large
  - Markdown: split at headings
  - Everything else: sliding window with overlap

Key fix: HTML/CSS files are NEVER split at tag boundaries.
They are kept whole or split on blank lines only.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import tiktoken
from loguru import logger


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str
    language: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

def _token_len(text: str) -> int:
    return len(_TOKENIZER.encode(text))


_EXT_TO_LANG = {
    # Python
    ".py": "python", ".pyw": "python", ".pyx": "python",
    # JS/TS
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    # Frontend — important: these get whole-file treatment
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "css", ".sass": "css", ".less": "css",
    ".vue": "vue", ".svelte": "svelte",
    # Templates
    ".ejs": "html", ".hbs": "html", ".handlebars": "html",
    ".jinja": "html", ".jinja2": "html", ".pug": "html",
    # Backend
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust",
    ".cpp": "cpp", ".c": "c", ".h": "c", ".cc": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".cs": "csharp",
    # Shell
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    # Data/Config
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".xml": "xml", ".sql": "sql",
    ".graphql": "graphql", ".prisma": "prisma",
    # Docs
    ".md": "markdown", ".mdx": "markdown",
    ".rst": "rst", ".txt": "text",
    # Other
    ".r": "r", ".lua": "lua", ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir",
}

# Languages that should NEVER be split at structural boundaries
# — keep whole file or use pure sliding window only
_WHOLE_FILE_LANGS = {"html", "css", "vue", "svelte", "xml", "json"}


def _detect_language(path: str) -> str:
    name = Path(path).name.lower()
    ext  = Path(path).suffix.lower()
    # Extensionless special files
    specials = {
        "dockerfile": "dockerfile", "makefile": "makefile",
        "procfile": "text", "gemfile": "ruby",
    }
    if name in specials:
        return specials[name]
    return _EXT_TO_LANG.get(ext, "text")


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

_SPLIT_PATTERNS = {
    "python":     re.compile(r"^(class |def )", re.MULTILINE),
    "javascript": re.compile(r"^(function |class |const |let |var |async function )", re.MULTILINE),
    "typescript": re.compile(r"^(function |class |interface |type |const |async function )", re.MULTILINE),
    "java":       re.compile(r"^(public |private |protected |class )", re.MULTILINE),
    "go":         re.compile(r"^(func |type |var |const )", re.MULTILINE),
    "rust":       re.compile(r"^(fn |pub |struct |impl |trait |enum )", re.MULTILINE),
    "ruby":       re.compile(r"^(def |class |module )", re.MULTILINE),
    "php":        re.compile(r"^(function |class |namespace )", re.MULTILINE),
    "csharp":     re.compile(r"^(public |private |protected |class |namespace )", re.MULTILINE),
    "markdown":   re.compile(r"^#{1,3} .+", re.MULTILINE),
    "sql":        re.compile(r"^(CREATE |SELECT |INSERT |UPDATE |DELETE |DROP )", re.MULTILINE | re.IGNORECASE),
}


def _split_by_structure(text: str, language: str) -> List[str]:
    """Split at semantic boundaries. Returns [text] unchanged for HTML/CSS."""
    # Never structurally split these — return whole text
    if language in _WHOLE_FILE_LANGS:
        return [text]

    pattern = _SPLIT_PATTERNS.get(language)
    if not pattern:
        # Fallback: split on double newlines (paragraph-style)
        parts = re.split(r"\n{3,}", text)
        return [p for p in parts if p.strip()]

    segments = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            segments.append(text[last:m.start()])
        last = m.start()
    segments.append(text[last:])
    return [s for s in segments if s.strip()]


def _sliding_window(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Split long text into overlapping windows by token count."""
    lines = text.splitlines(keepends=True)
    chunks = []
    current = []
    current_tokens = 0

    for line in lines:
        line_tokens = _token_len(line)
        if current_tokens + line_tokens > max_tokens and current:
            chunks.append("".join(current))
            # Keep last `overlap` tokens worth of lines for context
            overlap_lines = []
            overlap_count = 0
            for l in reversed(current):
                t = _token_len(l)
                if overlap_count + t > overlap:
                    break
                overlap_lines.insert(0, l)
                overlap_count += t
            current = overlap_lines
            current_tokens = overlap_count
        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append("".join(current))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str,
    max_tokens: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    if not text.strip():
        return []

    language = _detect_language(source)
    lines = text.splitlines()
    total_tokens = _token_len(text)

    # Small files: keep whole — never split files under max_tokens
    if total_tokens <= max_tokens:
        return [Chunk(
            text=text.strip(),
            source=source,
            chunk_id=f"{source}::0",
            language=language,
            start_line=0,
            end_line=len(lines) - 1,
            metadata={"language": language, "token_count": total_tokens},
        )]

    # Large files: split then window if needed
    structural_segments = _split_by_structure(text, language)

    raw_chunks: List[str] = []
    for seg in structural_segments:
        if _token_len(seg) <= max_tokens:
            raw_chunks.append(seg)
        else:
            raw_chunks.extend(_sliding_window(seg, max_tokens, overlap))

    # Filter empty
    raw_chunks = [r for r in raw_chunks if r.strip()]
    if not raw_chunks:
        raw_chunks = [text]

    # Build Chunk objects with approximate line numbers
    chunks = []
    line_cursor = 0
    for idx, raw in enumerate(raw_chunks):
        raw_lines = raw.count("\n") + 1
        start_line = line_cursor
        end_line = min(line_cursor + raw_lines - 1, len(lines) - 1)
        line_cursor = end_line + 1
        chunks.append(Chunk(
            text=raw.strip(),
            source=source,
            chunk_id=f"{source}::{idx}",
            language=language,
            start_line=start_line,
            end_line=end_line,
            metadata={"language": language, "token_count": _token_len(raw)},
        ))

    logger.debug(f"Chunked '{source}' → {len(chunks)} chunks (lang={language}, total_tokens={total_tokens})")
    return chunks


def chunk_file(path: str | Path, max_tokens: int = 512, overlap: int = 64) -> List[Chunk]:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return chunk_text(text, source=str(path), max_tokens=max_tokens, overlap=overlap)