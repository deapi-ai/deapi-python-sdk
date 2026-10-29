from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import BinaryIO, Union

# Types accepted for file parameters. A plain ``str`` is intentionally NOT
# accepted: it must never be read as a filesystem path by default, so untrusted
# input handed to a server cannot make it read and exfiltrate an arbitrary local
# file. Pass a local file explicitly as ``Path(...)``, or pass ``bytes`` / an open
# binary file object.
FileInput = Union[Path, bytes, BinaryIO]


def normalize_file(file: FileInput, param_name: str = "file") -> tuple[str, bytes, str]:
    """Convert accepted file input types to httpx's expected tuple format.

    Accepts ``Path`` (read from disk), ``bytes``, or a binary file object. A plain
    ``str`` is rejected: wrap a local path in ``Path(...)`` so reading from disk is
    always an explicit choice, never an implicit side effect of a bare string.

    Returns:
        (filename, file_bytes, content_type)
    """
    if isinstance(file, str):
        raise TypeError(
            f"{param_name}: a plain str is not accepted as a file. Wrap a local path "
            "in Path(...), or pass bytes / an open binary file object. (A bare str is "
            "refused so untrusted input can't make a server read an arbitrary local file.)"
        )

    if isinstance(file, Path):
        content_type = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
        return (file.name, file.read_bytes(), content_type)

    if isinstance(file, bytes):
        return (f"{param_name}.bin", file, "application/octet-stream")

    # BinaryIO / file-like object
    name = getattr(file, "name", None)
    if isinstance(name, str):
        filename = Path(name).name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    else:
        filename = f"{param_name}.bin"
        content_type = "application/octet-stream"

    data = file.read()
    return (filename, data, content_type)


def normalize_files(
    files: list[FileInput] | FileInput,
    param_name: str = "file",
) -> list[tuple[str, bytes, str]]:
    """Normalize a single file or list of files."""
    if not isinstance(files, list):
        return [normalize_file(files, param_name)]
    return [normalize_file(f, param_name) for f in files]
