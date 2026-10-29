from __future__ import annotations

import io
from pathlib import Path

import pytest

from deapi._files import normalize_file, normalize_files


class TestNormalizeFile:
    """A plain str is never read from disk; local files must be explicit."""

    def test_rejects_plain_str(self) -> None:
        with pytest.raises(TypeError, match="plain str is not accepted"):
            normalize_file("some/local/path.png", "image")  # type: ignore[arg-type]

    def test_rejects_str_that_looks_like_a_secret_path(self) -> None:
        # the exact thing the guard exists for: untrusted input on a server
        with pytest.raises(TypeError):
            normalize_file("/etc/passwd", "image")  # type: ignore[arg-type]

    def test_accepts_path(self, tmp_path: Path) -> None:
        p = tmp_path / "pic.png"
        p.write_bytes(b"png-bytes")
        name, data, content_type = normalize_file(p, "image")
        assert name == "pic.png"
        assert data == b"png-bytes"
        assert content_type == "image/png"

    def test_accepts_bytes(self) -> None:
        name, data, content_type = normalize_file(b"raw", "image")
        assert data == b"raw"
        assert name == "image.bin"
        assert content_type == "application/octet-stream"

    def test_accepts_binary_file_object(self) -> None:
        buf = io.BytesIO(b"abc")  # no .name attribute -> falls back to default name
        name, data, content_type = normalize_file(buf, "image")
        assert data == b"abc"
        assert name == "image.bin"

    def test_normalize_files_rejects_str_in_list(self) -> None:
        with pytest.raises(TypeError):
            normalize_files(["a.png"], "image")  # type: ignore[list-item]

    def test_normalize_files_accepts_paths(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.png"
        p1.write_bytes(b"a")
        p2 = tmp_path / "b.png"
        p2.write_bytes(b"b")
        out = normalize_files([p1, p2], "image")
        assert [t[1] for t in out] == [b"a", b"b"]
