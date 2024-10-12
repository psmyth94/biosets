from functools import partial
from typing import Callable, List, Optional

import datasets
import datasets.data_files
from packaging import version

from biosets.utils.import_util import _fsspec_version
from biosets.utils import logging

logger = logging.get_logger(__name__)
_SAMPLE_METADATA_NAMES = [
    "metadata",
    "sample_metadata",
    "sample",
    "samples",
]
_FEATURE_METADATA_NAMES = [
    "feature_metadata",
    "feature",
    "features",
    "annotation",
    "annotations",
]

_EXTENSIONS = [
    "csv",
    "jsonl",
    "tsv",
    "json",
    "parquet",
    "arrow",
    "feather",
    "txt",
]

SAMPLE_METADATA_FILENAMES = [
    f"{name}.{e}" for name in _SAMPLE_METADATA_NAMES for e in _EXTENSIONS
]
FEATURE_METADATA_FILENAMES = [
    f"{name}.{e}" for name in _FEATURE_METADATA_NAMES for e in _EXTENSIONS
]
if version.parse(_fsspec_version) < version.parse("2023.9.0"):
    SAMPLE_METADATA_PATTERNS = [f"**/{file}" for file in SAMPLE_METADATA_FILENAMES]
    SAMPLE_METADATA_PATTERNS.extend([f"{file}" for file in SAMPLE_METADATA_FILENAMES])
    FEATURE_METADATA_PATTERNS = [f"**/{file}" for file in FEATURE_METADATA_FILENAMES]
    FEATURE_METADATA_PATTERNS.extend([f"{file}" for file in FEATURE_METADATA_FILENAMES])
else:
    SAMPLE_METADATA_PATTERNS = [f"**/{file}" for file in SAMPLE_METADATA_FILENAMES]
    FEATURE_METADATA_PATTERNS = [f"**/{file}" for file in FEATURE_METADATA_FILENAMES]


METADATA_PATTERNS = SAMPLE_METADATA_PATTERNS + FEATURE_METADATA_PATTERNS


def get_metadata_patterns(
    base_path: str,
    download_config: Optional[datasets.DownloadConfig] = None,
) -> List[str]:
    """
    Get the supported metadata patterns from a local directory.
    """
    resolver = partial(
        datasets.data_files.resolve_pattern,
        base_path=base_path,
        download_config=download_config,
    )
    try:
        return _get_sample_metadata_files_patterns(resolver)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The directory at {base_path} doesn't contain any metadata file"
        ) from None


def _get_sample_metadata_files_patterns(
    pattern_resolver: Callable[[str], List[str]],
) -> List[str]:
    """
    Get the supported metadata patterns from a directory or repository.
    """
    non_empty_patterns = []
    for pattern in SAMPLE_METADATA_PATTERNS:
        try:
            sample_metadata_files = pattern_resolver(pattern)
            if len(sample_metadata_files) > 0:
                non_empty_patterns.append(pattern)
        except FileNotFoundError:
            pass
    if non_empty_patterns:
        return non_empty_patterns
    raise FileNotFoundError(
        f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}"
    )


def get_feature_metadata_patterns(
    base_path: str,
    download_config: Optional[datasets.DownloadConfig] = None,
) -> List[str]:
    """
    Get the supported metadata patterns from a local directory.
    """
    resolver = partial(
        datasets.data_files.resolve_pattern,
        base_path=base_path,
        download_config=download_config,
    )
    try:
        return _get_feature_metadata_files_patterns(resolver)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The directory at {base_path} doesn't contain any metadata file"
        ) from None


def _get_feature_metadata_files_patterns(
    pattern_resolver: Callable[[str], List[str]],
) -> List[str]:
    """
    Get the supported metadata patterns from a directory or repository.
    """
    non_empty_patterns = []
    for pattern in FEATURE_METADATA_PATTERNS:
        try:
            metadata_files = pattern_resolver(pattern)
            if len(metadata_files) > 0:
                non_empty_patterns.append(pattern)
        except FileNotFoundError:
            pass
    if non_empty_patterns:
        return non_empty_patterns
    raise FileNotFoundError(
        f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}"
    )
