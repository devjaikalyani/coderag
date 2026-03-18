"""
loaders.py
----------
Load source documents from:
  - Local file / directory
  - GitHub repository URL
  - Plain text string (for testing)
"""

import os
import tempfile
from pathlib import Path
from typing import List, Iterator, Optional
from urllib.parse import urlparse

import requests
from loguru import logger

from src.ingestion.chunker import Chunk, chunk_file

# File extensions we care about
CODE_EXTENSIONS = {
    # Python
    ".py", ".pyw", ".pyx",
    # JavaScript / TypeScript / Frontend
    ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".vue", ".svelte",
    # Template engines
    ".ejs", ".hbs", ".handlebars", ".jinja", ".jinja2",
    ".pug", ".jade",
    # Systems / Backend
    ".java", ".kt", ".scala",
    ".go", ".rs",
    ".cpp", ".c", ".h", ".cc", ".cxx",
    ".rb", ".php", ".swift", ".m",
    ".cs", ".fs", ".vb",
    # Shell
    ".sh", ".bash", ".zsh", ".fish", ".ps1",
    # Data / Config
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".env",
    # Docs
    ".md", ".rst", ".txt", ".mdx",
    # SQL / DB
    ".sql", ".prisma", ".graphql", ".gql",
    # Other
    ".r", ".lua", ".dart", ".ex", ".exs",
    ".dockerfile", ".makefile",
}

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".mypy_cache", ".pytest_cache",
}


# ---------------------------------------------------------------------------
# Source path cleaner
# ---------------------------------------------------------------------------

def _clean_source(full_path: str, root_dir: str, repo_name: Optional[str] = None) -> str:
    """
    Convert an absolute temp path into a clean relative path.

    /var/folders/.../tmpXYZ/blog-web-app/routes/index.js
        → blog-web-app/routes/index.js

    If repo_name is provided, ensure it appears as the root prefix.
    """
    try:
        rel = Path(full_path).relative_to(root_dir)
        parts = rel.parts
        # Strip any auto-generated extraction prefix like "owner-repo-abc123"
        # (GitHub zip extracts to a folder like "devjaikalyani-blog-web-app-a1b2c3")
        if parts and repo_name:
            # If first part looks like an extraction dir, skip it
            first = parts[0]
            if first != repo_name and (
                repo_name in first or first.count("-") >= 2
            ):
                parts = (repo_name,) + parts[1:]
        return str(Path(*parts)) if parts else str(rel)
    except ValueError:
        # Fallback: just return the filename
        return Path(full_path).name


# ---------------------------------------------------------------------------
# Local loader
# ---------------------------------------------------------------------------

def load_local(
    path: str | Path,
    max_tokens: int = 512,
    overlap: int = 64,
    repo_name: Optional[str] = None,
    _root_dir: Optional[str] = None,
) -> List[Chunk]:
    """Load all supported files from a local file or directory."""
    path = Path(path)
    root_dir = _root_dir or str(path)
    all_chunks: List[Chunk] = []

    if path.is_file():
        if path.suffix.lower() in CODE_EXTENSIONS:
            chunks = chunk_file(path, max_tokens, overlap)
            for c in chunks:
                c.source = _clean_source(c.source, root_dir, repo_name)
                c.chunk_id = f"{c.source}::{c.chunk_id.split('::')[-1]}"
            all_chunks.extend(chunks)
    elif path.is_dir():
        for file in _walk_directory(path):
            try:
                chunks = chunk_file(file, max_tokens, overlap)
                for c in chunks:
                    c.source = _clean_source(str(file), root_dir, repo_name)
                    c.chunk_id = f"{c.source}::{c.chunk_id.split('::')[-1]}"
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Skipping {file}: {e}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    logger.info(f"Loaded {len(all_chunks)} chunks from '{path}'")
    return all_chunks


# Extensionless files to include by exact name
EXTENSIONLESS_FILES = {
    "dockerfile", "makefile", "procfile", "gemfile", "rakefile",
    "vagrantfile", "brewfile", "justfile", "caddyfile",
    ".env.example", ".env.sample", ".gitignore", ".eslintrc",
    ".prettierrc", ".babelrc", ".npmrc",
}

def _walk_directory(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fname in filenames:
            p = Path(dirpath) / fname
            if (p.suffix.lower() in CODE_EXTENSIONS
                    or fname.lower() in EXTENSIONLESS_FILES):
                yield p


# ---------------------------------------------------------------------------
# GitHub loader
# ---------------------------------------------------------------------------

def _parse_repo_name(repo_url: str) -> str:
    """Extract clean repo name from GitHub URL."""
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        return parts[1].replace(".git", "")
    return "repo"


def load_github(
    repo_url: str,
    branch: str = "main",
    max_tokens: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """
    Clone a GitHub repo and load all supported files with clean source paths.
    """
    repo_name = _parse_repo_name(repo_url)
    try:
        import git
        return _load_github_git(repo_url, branch, max_tokens, overlap, repo_name)
    except ImportError:
        logger.warning("gitpython not available, using GitHub API zip download")
        return _load_github_zip(repo_url, branch, max_tokens, overlap, repo_name)


def _load_github_git(repo_url, branch, max_tokens, overlap, repo_name):
    import git
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = str(Path(tmpdir) / repo_name)
        logger.info(f"Cloning {repo_url} → {repo_name} (branch={branch})…")
        git.Repo.clone_from(repo_url, clone_dir, branch=branch, depth=1)
        return load_local(
            clone_dir,
            max_tokens=max_tokens,
            overlap=overlap,
            repo_name=repo_name,
            _root_dir=tmpdir,
        )


def _load_github_zip(repo_url, branch, max_tokens, overlap, repo_name):
    """Download repo as zip via GitHub API."""
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse GitHub URL: {repo_url}")
    owner, repo = parts[0], parts[1].replace(".git", "")
    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{branch}"

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "repo.zip"
        logger.info(f"Downloading {zip_url}…")
        r = requests.get(zip_url, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)

        import zipfile
        extract_dir = Path(tmpdir) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        return load_local(
            extract_dir,
            max_tokens=max_tokens,
            overlap=overlap,
            repo_name=repo_name,
            _root_dir=str(extract_dir),
        )


# ---------------------------------------------------------------------------
# Plain text loader
# ---------------------------------------------------------------------------

def load_text(
    text: str,
    source_name: str = "inline",
    max_tokens: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    from src.ingestion.chunker import chunk_text
    return chunk_text(text, source=source_name, max_tokens=max_tokens, overlap=overlap)