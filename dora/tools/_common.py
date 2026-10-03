"""Runtime shared by the MCP tool servers.

Every server is launched as ``python -m dora.tools.<name> --temp_dir <dir>``. Tool
outputs are written below ``<dir>``; while a question is running, the benchmark
runner writes its id to ``<dir>/.session`` so artifacts land in ``<dir>/<question_id>/``.
"""
import argparse
import hashlib
import os
from pathlib import Path


class SessionTempDir:
    """Path-like temp directory that resolves to the current question's subdirectory."""

    def __init__(self, base):
        self._base = Path(base)
        self._base.mkdir(parents=True, exist_ok=True)

    @property
    def _dir(self) -> Path:
        session_file = self._base / ".session"
        if session_file.is_file():
            try:
                session_id = session_file.read_text().strip()
                if session_id:
                    d = self._base / session_id
                    d.mkdir(parents=True, exist_ok=True)
                    return d
            except Exception:
                pass
        return self._base

    def __truediv__(self, other):
        return self._dir / other

    def __rtruediv__(self, other):
        return Path(other) / self._dir

    def __str__(self):
        return str(self._dir)

    def __repr__(self):
        return f"SessionTempDir({self._base})"

    def __fspath__(self):
        return str(self._dir)

    def mkdir(self, **kwargs):
        return self._dir.mkdir(**kwargs)

    def exists(self):
        return self._dir.exists()

    def iterdir(self):
        return self._dir.iterdir()

    def resolve(self):
        return self._dir.resolve()

    @property
    def parent(self):
        return self._dir.parent

    @property
    def name(self):
        return self._dir.name


def temp_dir_arg() -> str:
    """``--temp_dir`` of the current server process (defaults to ``out/tmp``)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--temp_dir", type=str, default="out/tmp")
    args, _ = parser.parse_known_args()
    return args.temp_dir


def session_temp_dir() -> SessionTempDir:
    return SessionTempDir(temp_dir_arg())


def short_name(*parts: str, suffix: str = ".geojson", max_len: int = 50) -> str:
    """Readable, collision-free output filename built from ``parts`` (truncated + md5 suffix)."""
    base = "_".join([p for p in parts if p])
    base = base.replace(os.sep, "_").replace("\\", "_").replace("/", "_")
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:6]
    head = base[: max_len - (1 + len(h) + len(suffix))].rstrip("_")
    return f"{head}_{h}{suffix}"
