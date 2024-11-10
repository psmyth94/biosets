from functools import partial
import re
from typing import Callable, Dict, List, Optional, Set

import datasets
import datasets.config
import datasets.data_files
from datasets.utils.file_utils import xbasename
from datasets.naming import _split_re
from packaging import version

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
if datasets.config.FSSPEC_VERSION < version.parse("2023.9.0"):
    SAMPLE_METADATA_PATTERNS = [f"**/{file}" for file in SAMPLE_METADATA_FILENAMES]
    SAMPLE_METADATA_PATTERNS.extend([f"{file}" for file in SAMPLE_METADATA_FILENAMES])
    FEATURE_METADATA_PATTERNS = [f"**/{file}" for file in FEATURE_METADATA_FILENAMES]
    FEATURE_METADATA_PATTERNS.extend([f"{file}" for file in FEATURE_METADATA_FILENAMES])
else:
    SAMPLE_METADATA_PATTERNS = [f"**/{file}" for file in SAMPLE_METADATA_FILENAMES]
    FEATURE_METADATA_PATTERNS = [f"**/{file}" for file in FEATURE_METADATA_FILENAMES]


METADATA_PATTERNS = SAMPLE_METADATA_PATTERNS + FEATURE_METADATA_PATTERNS


ALL_SPLIT_SAMPLE_METADATA_PATTERNS = [
    f"data/{{split}}-{name}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"
    for name in _SAMPLE_METADATA_NAMES
]

ALL_SPLIT_FEATURE_METADATA_PATTERNS = [
    f"data/{{split}}-{name}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*"
    for name in _FEATURE_METADATA_NAMES
]


KEYWORDS_IN_DIR_NAME_BASE_SAMPLE_METADATA_PATTERNS = []
KEYWORDS_IN_DIR_NAME_BASE_FEATURE_METADATA_PATTERNS = []
for pattern in datasets.data_files.KEYWORDS_IN_DIR_NAME_BASE_PATTERNS:
    for name in _SAMPLE_METADATA_NAMES:
        name_pattern = name.replace("_", "[{sep}]")
        if "/{keyword}/" in pattern:
            KEYWORDS_IN_DIR_NAME_BASE_SAMPLE_METADATA_PATTERNS.extend(
                [
                    pattern.replace("/{keyword}/", "/{keyword}/" + name_pattern + "/"),
                    pattern.replace("/{keyword}/", "/" + name_pattern + "/{keyword}/"),
                ]
            )
        else:
            KEYWORDS_IN_DIR_NAME_BASE_SAMPLE_METADATA_PATTERNS.extend(
                [
                    pattern.replace("{keyword}", "{keyword}[{sep}]" + name_pattern),
                    pattern.replace("{keyword}", name_pattern + "[{sep}]{keyword}"),
                ]
            )
    for name in _FEATURE_METADATA_NAMES:
        name_pattern = name.replace("_", "[{sep}]")
        if "/{keyword}/" in pattern:
            KEYWORDS_IN_DIR_NAME_BASE_FEATURE_METADATA_PATTERNS.extend(
                [
                    pattern.replace("/{keyword}/", "/{keyword}/" + name_pattern + "/"),
                    pattern.replace("/{keyword}/", "/" + name_pattern + "/{keyword}/"),
                ]
            )
        else:
            KEYWORDS_IN_DIR_NAME_BASE_FEATURE_METADATA_PATTERNS.extend(
                [
                    pattern.replace("{keyword}", "{keyword}[{sep}]" + name_pattern),
                    pattern.replace("{keyword}", name_pattern + "[{sep}]{keyword}"),
                ]
            )

KEYWORDS_IN_FILENAME_BASE_SAMPLE_METADATA_PATTERNS = []
KEYWORDS_IN_FILENAME_BASE_FEATURE_METADATA_PATTERNS = []
DEFAULT_SAMPLE_METADATA_PATTERNS_ALL = {datasets.data_files.Split.TRAIN: []}
DEFAULT_FEATURE_METADATA_PATTERNS_ALL = {datasets.data_files.Split.TRAIN: []}
for pattern in datasets.data_files.KEYWORDS_IN_FILENAME_BASE_PATTERNS:
    for name in _SAMPLE_METADATA_NAMES:
        name_pattern = name.replace("_", "[{sep}]")
        KEYWORDS_IN_FILENAME_BASE_SAMPLE_METADATA_PATTERNS.extend(
            [
                pattern.replace("{keyword}", "{keyword}[{sep}]" + name_pattern),
                pattern.replace("{keyword}", name_pattern + "[{sep}]{keyword}"),
            ]
        )
        sep = datasets.data_files.NON_WORDS_CHARS
        if "metadata" in name:
            continue
        if "{keyword}*" in pattern and name.endswith("s"):
            continue
        DEFAULT_SAMPLE_METADATA_PATTERNS_ALL[datasets.data_files.Split.TRAIN].append(
            pattern.replace("{keyword}", name_pattern).format(sep=sep)
        )
    for name in _FEATURE_METADATA_NAMES:
        name_pattern = name.replace("_", "[{sep}]")
        KEYWORDS_IN_FILENAME_BASE_FEATURE_METADATA_PATTERNS.extend(
            [
                pattern.replace("{keyword}", "{keyword}[{sep}]" + name_pattern),
                pattern.replace("{keyword}", name_pattern + "[{sep}]{keyword}"),
            ]
        )
        sep = datasets.data_files.NON_WORDS_CHARS
        if "metadata" in name:
            continue
        if "{keyword}*" in pattern and name.endswith("s"):
            continue
        DEFAULT_FEATURE_METADATA_PATTERNS_ALL[datasets.data_files.Split.TRAIN].append(
            pattern.replace("{keyword}", name_pattern).format(sep=sep)
        )

DEFAULT_SAMPLE_METADATA_PATTERNS_SPLIT_IN_FILENAME = {
    split: [
        pattern.format(keyword=keyword, sep=datasets.data_files.NON_WORDS_CHARS)
        for keyword in datasets.data_files.SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_FILENAME_BASE_SAMPLE_METADATA_PATTERNS
    ]
    for split in datasets.data_files.DEFAULT_SPLITS
}

DEFAULT_SAMPLE_METADATA_PATTERNS_SPLIT_IN_DIR_NAME = {
    split: [
        pattern.format(keyword=keyword, sep=datasets.data_files.NON_WORDS_CHARS)
        for keyword in datasets.data_files.SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_DIR_NAME_BASE_SAMPLE_METADATA_PATTERNS
    ]
    for split in datasets.data_files.DEFAULT_SPLITS
}

ALL_DEFAULT_SAMPLE_METADATA_PATTERNS = [
    DEFAULT_SAMPLE_METADATA_PATTERNS_SPLIT_IN_FILENAME,
    DEFAULT_SAMPLE_METADATA_PATTERNS_SPLIT_IN_DIR_NAME,
    DEFAULT_SAMPLE_METADATA_PATTERNS_ALL,
]

DEFAULT_FEATURE_METADATA_PATTERNS_SPLIT_IN_FILENAME = {
    split: [
        pattern.format(keyword=keyword, sep=datasets.data_files.NON_WORDS_CHARS)
        for keyword in datasets.data_files.SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_FILENAME_BASE_FEATURE_METADATA_PATTERNS
    ]
    for split in datasets.data_files.DEFAULT_SPLITS
}

DEFAULT_FEATURE_METADATA_PATTERNS_SPLIT_IN_DIR_NAME = {
    split: [
        pattern.format(keyword=keyword, sep=datasets.data_files.NON_WORDS_CHARS)
        for keyword in datasets.data_files.SPLIT_KEYWORDS[split]
        for pattern in KEYWORDS_IN_DIR_NAME_BASE_FEATURE_METADATA_PATTERNS
    ]
    for split in datasets.data_files.DEFAULT_SPLITS
}

ALL_DEFAULT_FEATURE_METADATA_PATTERNS = [
    DEFAULT_FEATURE_METADATA_PATTERNS_SPLIT_IN_FILENAME,
    DEFAULT_FEATURE_METADATA_PATTERNS_SPLIT_IN_DIR_NAME,
    DEFAULT_FEATURE_METADATA_PATTERNS_ALL,
]


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


def _get_metadata_files_patterns(
    pattern_resolver: Callable[[str], List[str]],
    all_split_patterns: List[str],
    all_default_patterns: List[Dict[str, List[str]]],
):
    # first check the split patterns like data/{split}-00000-of-00001.parquet
    for split_pattern in all_split_patterns:
        pattern = split_pattern.replace("{split}", "*")
        try:
            data_files = pattern_resolver(pattern)
        except FileNotFoundError:
            continue
        if len(data_files) > 0:
            splits: Set[str] = {
                datasets.data_files.string_to_dict(
                    xbasename(p),
                    datasets.data_files.glob_pattern_to_regex(xbasename(split_pattern)),
                )["split"]
                for p in data_files
            }
            if any(not re.match(_split_re, split) for split in splits):
                raise ValueError(
                    f"Split name should match '{_split_re}'' but got '{splits}'."
                )
            sorted_splits = [
                str(split)
                for split in datasets.data_files.DEFAULT_SPLITS
                if split in splits
            ] + sorted(splits - set(datasets.data_files.DEFAULT_SPLITS))
            return {
                split: [split_pattern.format(split=split)] for split in sorted_splits
            }
    # then check the default patterns based on train/valid/test splits
    for patterns_dict in all_default_patterns:
        non_empty_splits = []
        for split, patterns in patterns_dict.items():
            for pattern in patterns:
                try:
                    data_files = pattern_resolver(pattern)
                except FileNotFoundError:
                    continue
                if len(data_files) > 0:
                    non_empty_splits.append(split)
                    break
        if non_empty_splits:
            return {split: patterns_dict[split] for split in non_empty_splits}
    raise FileNotFoundError(
        f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}"
    )


def _get_sample_metadata_files_patterns(
    pattern_resolver: Callable[[str], List[str]],
) -> Dict[str, List[str]]:
    """
    Get the default pattern from a directory or repository by testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    return _get_metadata_files_patterns(
        pattern_resolver,
        ALL_SPLIT_SAMPLE_METADATA_PATTERNS,
        ALL_DEFAULT_SAMPLE_METADATA_PATTERNS,
    )


def _get_feature_metadata_files_patterns(
    pattern_resolver: Callable[[str], List[str]],
) -> List[str]:
    """
    Get the supported metadata patterns from a directory or repository.
    """
    return _get_metadata_files_patterns(
        pattern_resolver,
        ALL_SPLIT_FEATURE_METADATA_PATTERNS,
        ALL_DEFAULT_FEATURE_METADATA_PATTERNS,
    )
