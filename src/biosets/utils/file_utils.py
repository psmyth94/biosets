"""
This file is adapted from the datasets library, which in turn is adapted from the AllenNLP library.

datasets
~~~~~~~~
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import io
import os
import posixpath
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, TypeVar, Union
from urllib.parse import unquote, urlparse

from huggingface_hub.utils import insecure_hashlib

from . import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

INCOMPLETE_SUFFIX = ".incomplete"

PathLike = Union[str, Path]
T = TypeVar("T", str, Path)


def is_remote_url(url_or_filename: Union[str, Path]) -> bool:
    return urlparse(str(url_or_filename)).scheme != "" and not os.path.ismount(
        urlparse(str(url_or_filename)).scheme + ":/"
    )


def is_local_path(url_or_filename: Union[str, Path]) -> bool:
    # On unix the scheme of a local path is empty (for both absolute and relative),
    # while on windows the scheme is the drive name (ex: "c") for absolute paths.
    # for details on the windows behavior, see https://bugs.python.org/issue42215
    url_or_filename = Path(url_or_filename).resolve().as_posix()

    return urlparse(url_or_filename).scheme == "" or os.path.ismount(
        urlparse(url_or_filename).scheme + ":/"
    )


def is_relative_path(url_or_filename: Union[str, Path]) -> bool:
    return urlparse(str(url_or_filename)).scheme == "" and not os.path.isabs(
        str(url_or_filename)
    )


def expand_path(path):
    """
    Check if a path is relative and expand it if necessary.
    Handles file paths and URLs, including user home directory expansion.

    :param path: str representing the path or URL to check and expand
    :return: str with the expanded absolute path or full URL
    """
    # Parse the path as a URL
    parsed = urlparse(path)

    # If it's a URL (not a local file path)
    if parsed.scheme and parsed.netloc:
        return path  # Return the original URL

    # If it's a file URL, convert it to a local path
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        # On Windows, remove leading slash
        if sys.platform == "win32" and path.startswith("/"):
            path = path[1:]
    else:
        # It's a regular path, use the full original string
        path = unquote(path)

    # Convert to Path object
    path = Path(path)

    # Expand user's home directory if present
    path = path.expanduser()

    # Check if the path is absolute
    if path.is_absolute():
        return str(path.resolve())

    # If path is relative, make it absolute
    return os.path.normpath(str((Path.cwd() / path).resolve()))


def relative_to_absolute_path(path: T) -> T:
    """Convert relative path to absolute path."""
    abs_path_str = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    return Path(abs_path_str) if isinstance(path, Path) else abs_path_str


def is_file_name(url_or_path_or_file: T) -> bool:
    if is_local_path(url_or_path_or_file):
        if is_relative_path(url_or_path_or_file):
            if "/" not in Path(url_or_path_or_file).as_posix():
                return True
    return False


def has_ext(url_or_path: Union[str, Path], ext: Optional[str] = None) -> bool:
    if ext is None:
        return Path(url_or_path).suffix != ""
    return Path(url_or_path).suffix == ext


def has_separator(url_or_path: Union[str, Path]) -> bool:
    return "/" in str(url_or_path) or "\\" in str(url_or_path)


def url_or_path_join(base_name: str, *pathnames: str) -> str:
    if is_remote_url(base_name):
        return posixpath.join(
            base_name,
            *(str(pathname).replace(os.sep, "/").lstrip("/") for pathname in pathnames),
        )
    else:
        return Path(base_name, *pathnames).as_posix()


def url_or_path_parent(url_or_path: str) -> str:
    if is_remote_url(url_or_path):
        return url_or_path[: url_or_path.rindex("/")]
    else:
        return os.path.dirname(url_or_path)


def hash_url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    url_hash = insecure_hashlib.sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = insecure_hashlib.sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    if url.endswith(".py"):
        filename += ".py"

    return filename


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (
            "".join(docstr) + "\n\n" + (fn.__doc__ if fn.__doc__ is not None else "")
        )
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (
            (fn.__doc__ if fn.__doc__ is not None else "") + "\n\n" + "".join(docstr)
        )
        return fn

    return docstring_decorator


def estimate_dataset_size(paths):
    return sum(path.stat().st_size for path in paths)


def readline(f: io.RawIOBase):
    # From: https://github.com/python/cpython/blob/d27e2f4d118e7a9909b6a3e5da06c5ff95806a85/Lib/_pyio.py#L525
    res = bytearray()
    while True:
        b = f.read(1)
        if not b:
            break
        res += b
        if res.endswith(b"\n"):
            break
    return bytes(res)


def move_temp_file(
    temp_file: Union[tempfile._TemporaryFileWrapper, str, Path], final_file: str
):
    if isinstance(temp_file, tempfile._TemporaryFileWrapper):
        temp_file.close()
        temp_file_name = Path(temp_file.name)
    elif not isinstance(temp_file, Path):
        temp_file_name = Path(temp_file)

    if not isinstance(final_file, Path):
        final_file = Path(final_file)

    if temp_file_name.exists():
        if final_file.exists():
            final_file.unlink()
        # is source a windows path?
        if temp_file_name.resolve().drive:
            if len(str(temp_file_name)) > 255:
                src_file = "\\\\?\\" + str(temp_file_name.resolve())
            else:
                src_file = temp_file_name.as_posix()
        else:
            src_file = temp_file_name.as_posix()

        if final_file.resolve().drive:
            if len(str(final_file)) > 255:
                dst_file = "\\\\?\\" + str(final_file.resolve())
            else:
                dst_file = final_file.as_posix()
        else:
            dst_file = final_file.as_posix()
        shutil.move(src_file, dst_file)
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(dst_file, 0o666 & ~umask)
        return dst_file
    else:
        raise FileNotFoundError(f"Temporary file {temp_file_name.name} not found.")
