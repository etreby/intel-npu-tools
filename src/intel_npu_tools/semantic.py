"""Private local semantic indexing backed by OpenVINO's NPU embedding pipeline."""

import argparse
import contextlib
import errno
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from .paths import EMBEDDING_MODEL, RERANK_MODEL, SEMANTIC_DB
from .runtime import npu_properties


TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".css", ".csv", ".go", ".h", ".hpp", ".html",
    ".ini", ".java", ".js", ".json", ".jsx", ".log", ".md", ".php",
    ".py", ".rb", ".rs", ".rst", ".sh", ".sql", ".toml", ".ts", ".tsx",
    ".txt", ".xml", ".yaml", ".yml",
}
OFFICE_SUFFIXES = {".pdf", ".docx", ".pptx"}
# Matched only against directories below the indexed root, never against its
# ancestors.
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", ".cache", ".ipynb_checkpoints", ".mypy_cache",
    ".pytest_cache", ".ruff_cache",
}
# A directory holding any of these is a model directory, so it is pruned whole.
# Keying on the weights rather than on the directory's name keeps a source
# directory named "models" indexable while skipping the text sidecars that ship
# beside real weights -- tokenizer vocabularies and OpenVINO IR graphs, which
# are .json/.txt/.xml and so pass TEXT_SUFFIXES on their own.
WEIGHT_SUFFIXES = {".bin", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_OFFICE_FILE_BYTES = 25 * 1024 * 1024
# The pipeline truncates at 512 tokens, so chunks are kept short enough that
# their tail still reaches the model. See docs/SEMANTIC_SEARCH.md.
TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 1500
QUERY_INSTRUCTION = "Given a search query, retrieve relevant passages from the user's local files"
# Recorded per file. Bump when chunking changes: each file is then re-chunked
# the next time its root is indexed, so a file's passages never mix chunkers.
# Files belonging to roots that are not re-indexed keep working untouched.
CHUNKER_VERSION = "2"
LEGACY_CHUNKER_VERSION = "1"
# Passages handed to the cross-encoder. Twenty is the ceiling search() already
# imposes on `limit`, so reranking never scores a passage the API could not have
# returned anyway, and the cost stays bounded at about one second regardless of
# `limit`. The trade-off is that at limit=20 reranking only reorders what the
# embedding scan already chose; the recall benefit is largest at small limits.
RERANK_CANDIDATES = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, root TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL, digest TEXT NOT NULL,
    chunker TEXT NOT NULL DEFAULT '1', device INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL, content TEXT NOT NULL, vector BLOB NOT NULL,
    dimensions INTEGER NOT NULL, FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
"""

_pipeline = None
_pipeline_lock = threading.RLock()


def embedding_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            import openvino_genai as ov_genai

            if not EMBEDDING_MODEL.is_dir():
                raise FileNotFoundError(
                    f"Embedding model not found at {EMBEDDING_MODEL}; run scripts/download-models.py"
                )
            config = ov_genai.TextEmbeddingPipeline.Config()
            config.max_length = 512
            config.pad_to_max_length = True
            config.batch_size = 1
            config.pooling_type = ov_genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN
            config.normalize = True
            config.padding_side = "left"
            config.query_instruction = f"Instruct: {QUERY_INSTRUCTION}\nQuery:"
            # The Ubuntu NPU packages provide the compiler through the driver.
            _pipeline = ov_genai.TextEmbeddingPipeline(
                str(EMBEDDING_MODEL), "NPU", config, **npu_properties()
            )
    return _pipeline


def _default_rerank_factory():
    """Imported lazily so that OpenVINO stays out of this module's import path."""
    from .rerank import rerank_pipeline

    return rerank_pipeline()


def _split_lines(lines: list[str], first_line: int, max_chars: int) -> list[tuple[int, int, str]]:
    """Emit pieces of at most max_chars, each labelled with the lines it covers.

    A line longer than max_chars has no boundary to break on, so it is split
    across several pieces that all cite that one line rather than the whole
    surrounding range.
    """
    pieces: list[tuple[int, int, str]] = []
    buffer: list[str] = []
    buffer_start = first_line
    length = 0

    def flush(last_line: int) -> None:
        nonlocal buffer, length
        if buffer:
            pieces.append((buffer_start, last_line, "\n".join(buffer)))
            buffer, length = [], 0

    for offset, line in enumerate(lines):
        number = first_line + offset
        if len(line) > max_chars:
            flush(number - 1)
            for cut in range(0, len(line), max_chars):
                pieces.append((number, number, line[cut:cut + max_chars]))
            buffer_start = number + 1
            continue
        if buffer and length + len(line) + 1 > max_chars:
            flush(number - 1)
            buffer_start = number
        buffer.append(line)
        length += len(line) + 1
    flush(first_line + len(lines) - 1)
    return [(start, end, content) for start, end, content in pieces if content.strip()]


def chunk_text(
    text: str,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap_lines: int = 2,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[tuple[int, int, str]]:
    """Split text on line boundaries and return (start_line, end_line, content)."""
    lines = text.splitlines()
    chunks = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines) and (size < target_chars or end == start):
            size += len(lines[end]) + 1
            end += 1
        chunks.extend(_split_lines(lines[start:end], start + 1, max_chars))
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def _is_indexable(path: Path, include_office: bool = True) -> bool:
    try:
        if not path.is_file() or path.is_symlink():
            return False
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            return path.stat().st_size <= MAX_FILE_BYTES
        if include_office and suffix in OFFICE_SUFFIXES:
            return path.stat().st_size <= MAX_OFFICE_FILE_BYTES
        return False
    except OSError:
        return False


def _is_definitely_gone(path: str, device: int | None) -> bool:
    """True only when the file is missing from a filesystem that is still attached.

    os.path.exists() answers False in three different situations: the file was
    deleted, a parent directory lost its execute bit, and the filesystem holding
    it was unmounted. Only the first justifies dropping index rows.

    A permission error is ruled out by the errno: ENOENT and ENOTDIR are the only
    codes meaning the name is really absent. An offline mount also reports ENOENT,
    though, so the device recorded at index time is compared against the nearest
    existing ancestor: an unmounted tree leaves the mountpoint's own filesystem
    behind, and the mismatch keeps the rows. Rows recorded before 0.2.1 carry no
    device and fall back to the errno alone until the next run records one.
    """
    try:
        os.lstat(path)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTDIR):
            return False
    else:
        return False
    if device is None:
        return True
    return _living_device(Path(path).parent) == device


def _living_device(directory: Path) -> int | None:
    """Device of the nearest ancestor that still exists, or None if unknowable."""
    for candidate in [directory, *directory.parents]:
        try:
            return os.stat(candidate).st_dev
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ENOTDIR):
                continue
            return None
    return None


def iter_text_files(root: Path, include_office: bool = True) -> Iterable[Path]:
    """Yield indexable files below root. SKIP_DIRS applies to root's descendants only."""
    if root.is_file():
        if _is_indexable(root, include_office=include_office):
            yield root.resolve()
        return
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        if any(Path(name).suffix.lower() in WEIGHT_SUFFIXES for name in filenames):
            subdirectories[:] = []
            continue
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in SKIP_DIRS and not name.endswith(".egg-info")
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if _is_indexable(path, include_office=include_office):
                yield path.resolve()


def _like_prefix(resolved: str) -> str:
    """Escape LIKE wildcards so a literal '_' or '%' in a path cannot match siblings."""
    escaped = resolved.rstrip("/").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "/%"


def _root_predicate(resolved: str) -> tuple[str, list[str]]:
    return "(path = ? OR path LIKE ? ESCAPE '\\')", [resolved, _like_prefix(resolved)]


def _migrate(connection: sqlite3.Connection) -> None:
    """Add the per-file chunker and device columns to databases written before 0.2.1."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(files)")}
    if not columns:
        return
    if "chunker" not in columns:
        connection.execute(
            f"ALTER TABLE files ADD COLUMN chunker TEXT NOT NULL DEFAULT '{LEGACY_CHUNKER_VERSION}'"
        )
    if "device" not in columns:
        # Legacy rows carry no device, so they fall back to the plain
        # existence check until the next index run records one.
        connection.execute("ALTER TABLE files ADD COLUMN device INTEGER")


@contextlib.contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        _migrate(connection)
        with connection:
            yield connection
    finally:
        connection.close()


class SemanticIndex:
    def __init__(
        self,
        db_path: Path = SEMANTIC_DB,
        pipeline_factory: Callable = embedding_pipeline,
        rerank_factory: Callable | None = None,
    ):
        self.db_path = Path(db_path)
        self.pipeline_factory = pipeline_factory
        self.rerank_factory = rerank_factory

    def _rerank_available(self) -> bool:
        """An injected factory means available; otherwise ask the filesystem.

        Consulting the filesystem only when nothing was injected keeps tests
        from silently depending on whether the developer's machine happens to
        have the optional model downloaded.
        """
        if self.rerank_factory is not None:
            return True
        from .rerank import available

        return available()

    def _prune_removed(self, db: sqlite3.Connection, root: Path, present: set[str]) -> int:
        """Drop rows for files under this root that no longer exist on disk.

        A row is removed only when its file is genuinely gone, never merely
        because this traversal skipped it. That keeps files inside excluded
        directories and separately indexed nested roots intact, and it does not
        depend on which root last recorded a file: overlapping roots can be
        indexed in any order and deletions are still noticed. Nor does it depend
        on the filesystem being reachable: see _is_definitely_gone.
        """
        predicate, params = _root_predicate(str(root))
        stale = [
            row[0]
            for row in db.execute(f"SELECT path, device FROM files WHERE {predicate}", params)
            if row[0] not in present and _is_definitely_gone(row[0], row[1])
        ]
        db.executemany("DELETE FROM chunks WHERE path=?", [(path,) for path in stale])
        db.executemany("DELETE FROM files WHERE path=?", [(path,) for path in stale])
        return len(stale)

    def _forget(self, root: Path) -> dict:
        """Drop everything indexed under a root that has since been deleted.

        Re-indexing the path is the only handle a user has on it, so this must
        clean up rather than refuse; otherwise a deleted file or directory stays
        searchable forever.
        """
        with _connect(self.db_path) as db:
            removed = self._prune_removed(db, root, set())
        if not removed:
            raise ValueError(f"Path does not exist: {root}")
        return {
            "root": str(root), "files_seen": 0, "files_updated": 0, "files_unchanged": 0,
            "files_removed": removed, "files_rechunked": 0, "chunks_added": 0, "device": "NPU",
            "removed_root": (
                f"{root} no longer exists; its {removed} indexed file(s) were removed "
                "from the index."
            ),
        }

    def index(
        self,
        value: str,
        include_office: bool = True,
        include_media: bool = False,
    ) -> dict:
        root = Path(value).expanduser().resolve()
        if not root.exists():
            return self._forget(root)
        files = list(iter_text_files(root, include_office=include_office))
        updated = skipped = rechunked = chunks_added = 0
        with _connect(self.db_path) as db:
            for path in files:
                stat = path.stat()
                row = db.execute(
                    "SELECT mtime_ns, size, chunker FROM files WHERE path=?", (str(path),)
                ).fetchone()
                if row and (row[0], row[1]) == (stat.st_mtime_ns, stat.st_size):
                    if row[2] == CHUNKER_VERSION:
                        skipped += 1
                        continue
                    # Unchanged on disk, but chunked by an older release.
                    rechunked += 1
                if path.suffix.lower() in OFFICE_SUFFIXES:
                    from .extractors import DocumentExtractor

                    try:
                        sections = DocumentExtractor.extract_text(path)
                    except Exception:
                        continue
                    if not sections:
                        continue
                    text = "\n\n".join(section_text for _, section_text in sections)
                else:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                pieces = chunk_text(text)
                if not pieces:
                    continue
                vectors = []
                pipe = self.pipeline_factory()
                for _, _, content in pieces:
                    vector = np.asarray(pipe.embed_documents([content])[0], dtype=np.float32)
                    vectors.append(vector)
                digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                db.execute("DELETE FROM chunks WHERE path=?", (str(path),))
                db.execute(
                    "INSERT OR REPLACE INTO files(path,root,mtime_ns,size,digest,chunker,device)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (
                        str(path), str(root), stat.st_mtime_ns, stat.st_size, digest,
                        CHUNKER_VERSION, stat.st_dev,
                    ),
                )
                db.executemany(
                    "INSERT INTO chunks(path,start_line,end_line,content,vector,dimensions) VALUES(?,?,?,?,?,?)",
                    [
                        (str(path), start, end, content, vector.tobytes(), vector.size)
                        for (start, end, content), vector in zip(pieces, vectors)
                    ],
                )
                updated += 1
                chunks_added += len(pieces)
            removed = self._prune_removed(db, root, {str(path) for path in files})
        result = {
            "root": str(root), "files_seen": len(files), "files_updated": updated,
            "files_unchanged": skipped, "files_removed": removed,
            "files_rechunked": rechunked, "chunks_added": chunks_added, "device": "NPU",
        }
        if rechunked:
            result["rebuilt"] = (
                f"Chunking changed in this release, so {rechunked} unmodified file(s) under this "
                "root were re-embedded. Other indexed roots are untouched until you index them."
            )
        if not files:
            all_suffixes = sorted(TEXT_SUFFIXES | (OFFICE_SUFFIXES if include_office else set()))
            result["warning"] = (
                f"No indexable files found under {root}. Supported suffixes are "
                f"{', '.join(all_suffixes)}; directories named "
                f"{', '.join(sorted(SKIP_DIRS))}, directories holding model weights "
                f"({', '.join(sorted(WEIGHT_SUFFIXES))}), and files over "
                f"{MAX_FILE_BYTES // (1024 * 1024)} MiB are skipped."
            )
        return result

    def search(
        self,
        query: str,
        limit: int = 5,
        root: str | None = None,
        rerank: bool | None = None,
    ) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        limit = max(1, min(int(limit), 20))
        # Off unless asked for. Measured on this repository, the cross-encoder
        # helps when it is confident and hurts when it is not: on a query whose
        # answer it recognised it promoted the right passage with a +4.95 logit,
        # but on one where nothing scored above -6.2 it discarded the correct
        # hit that cosine had ranked first. Presence of the model is not
        # evidence the user wants that trade on every search.
        if rerank is None:
            rerank = False
        elif rerank and not self._rerank_available():
            raise FileNotFoundError(
                f"Reranking was requested but the model is not at {RERANK_MODEL}; run "
                "scripts/download-models.py --with-reranker"
            )
        vector = np.asarray(self.pipeline_factory().embed_query(query), dtype=np.float32)
        sql = "SELECT path,start_line,end_line,content,vector,dimensions FROM chunks"
        params: list[str] = []
        if root:
            predicate, params = _root_predicate(str(Path(root).expanduser().resolve()))
            sql += f" WHERE {predicate}"
        results = []
        with _connect(self.db_path) as db:
            for path, start, end, content, blob, dimensions in db.execute(sql, params):
                candidate = np.frombuffer(blob, dtype=np.float32, count=dimensions)
                if candidate.size != vector.size:
                    continue
                score = float(np.dot(vector, candidate))
                results.append({
                    "score": round(score, 6), "path": path, "start_line": start,
                    "end_line": end, "text": content,
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        if not rerank or not results:
            return results[:limit]
        return self._rerank(query, results[:RERANK_CANDIDATES], limit)

    def _rerank(self, query: str, candidates: list[dict], limit: int) -> list[dict]:
        """Re-score the embedding shortlist with the cross-encoder and reorder it.

        The cosine value stays in `score` so callers that already read it keep
        working, and `rerank_score` is added only to passages the cross-encoder
        actually saw. That is also how a caller tells a reranked result set from
        a plain one without the return type changing. The two are not
        comparable: `rerank_score` is an unbounded logit, so a reranked list is
        no longer sorted by `score`.
        """
        factory = self.rerank_factory or _default_rerank_factory
        ranked = factory().rerank(query, [item["text"] for item in candidates])
        reordered = []
        for index, score in ranked[:limit]:
            item = dict(candidates[index])
            item["rerank_score"] = round(float(score), 6)
            reordered.append(item)
        return reordered

    def status(self) -> dict:
        with _connect(self.db_path) as db:
            files = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            roots = [row[0] for row in db.execute("SELECT DISTINCT root FROM files ORDER BY root")]
        return {"database": str(self.db_path), "files": files, "chunks": chunks, "roots": roots}


def main():
    parser = argparse.ArgumentParser(description="Private semantic search using Intel AI Boost")
    sub = parser.add_subparsers(dest="command", required=True)
    index_parser = sub.add_parser("index", help="Index a text or office file or directory")
    index_parser.add_argument("path")
    index_parser.add_argument(
        "--no-office", dest="include_office", action="store_false", default=True,
        help="Exclude office documents (.pdf, .docx, .pptx)",
    )
    search_parser = sub.add_parser("search", help="Search indexed text")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.add_argument("--root")
    # Tri-state: unset means use the reranker when its model is installed.
    rerank_group = search_parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank", dest="rerank", action="store_true", default=None,
        help="Re-score results with the cross-encoder (error if not installed)",
    )
    rerank_group.add_argument(
        "--no-rerank", dest="rerank", action="store_false",
        help="Skip the cross-encoder even when its model is installed",
    )
    sub.add_parser("status", help="Show index statistics")
    args = parser.parse_args()
    index = SemanticIndex()
    if args.command == "index":
        result = index.index(args.path, include_office=args.include_office)
    elif args.command == "search":
        result = index.search(args.query, args.limit, args.root, args.rerank)
    else:
        result = index.status()
    print(json.dumps(result, indent=2, ensure_ascii=False))
