# NOTE: The contents of this file have been inlined from the fingerprint and _dill
# module in the datasets package's source code
# https://github.com/huggingface/datasets/blob/c47cc141c9e6e0edafffdcfde55b171612f1de76/src/datasets/fingerprint.py
#
# This module has fixes / adaptations for this software's use cases that make it
# different from the original datasets library
#
# The following modifications have been made:
#     - Added python source code parsing logic to the `Hasher` class. This is used to
#       hash the contents of a python file without comments since we only care about
#       the code that is actually executed.
#
# datasets
# ~~~~~~~~
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import os
import posixpath
import re
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Dict, List, Union

import datasets.utils.file_utils
import xxhash
from biocore.data_handling import DataHandler, get_data_format
from datasets.fingerprint import generate_random_fingerprint
from datasets.packaged_modules import _hash_python_lines
from datasets.utils._dill import dumps

import biosets.config as config
import biosets.utils.logging as logging
from biosets.utils.version import __version__

from .file_utils import is_file_name

_CACHING_ENABLED = True

logger = logging.get_logger(__name__)


class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: Dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    @staticmethod
    def _hash_python_lines(self, module: Union[ModuleType, FunctionType, type]) -> str:
        filtered_lines = []
        lines = inspect.getsource(module).splitlines()
        for line in lines:
            line = re.sub(r"#.*", "", line)  # remove comments
            if line:
                filtered_lines.append(line)
        return "\n".join(filtered_lines)

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


def enable_caching():
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = True


def disable_caching():
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    _CACHING_ENABLED = False


def is_caching_enabled() -> bool:
    """
    When applying transforms on a dataset, the data are stored in cache files.
    The caching mechanism allows to reload an existing cache file if it's already been computed.

    Reloading a dataset is possible since the cache files are named using the dataset fingerprint, which is updated
    after each transform.

    If disabled, the library will no longer reload cached datasets files when applying transforms to the datasets.
    More precisely, if the caching is disabled:
    - cache files are always recreated
    - cache files are written to a temporary directory that is deleted when session closes
    - cache files are named using a random hash instead of the dataset fingerprint
    - use [`~datasets.Dataset.save_to_disk`]] to save a transformed dataset or it will be deleted when session closes
    - caching doesn't affect [`~datasets.load_dataset`]. If you want to regenerate a dataset from scratch you should use
    the `download_mode` parameter in [`~datasets.load_dataset`].
    """
    global _CACHING_ENABLED
    return bool(_CACHING_ENABLED)


def fingerprint_from_kwargs(fingerprint, kwargs):
    hash = Hasher()
    if fingerprint:
        hash.update(fingerprint)

    for key, value in kwargs.items():
        if isinstance(key, str) and "fingerprint" in key:
            continue
        hash.update(key)
        if isinstance(value, dict):
            hash.update(fingerprint_from_kwargs(fingerprint, value))
        else:
            hash.update(str(value))

    return hash.hexdigest()


def fingerprint_from_data(data):
    if hasattr(data, "_fingerprint"):
        return data._fingerprint
    if hasattr(data, "fingerprint"):
        return data.fingerprint
    hasher = Hasher()
    original_format = get_data_format(data)

    if original_format is not None:
        features = DataHandler.get_column_names(data, generate_cols=True)
        n_samples = DataHandler.get_shape(data)[0]
        first_row = DataHandler.select_row(data, 0)
        hasher.update(original_format)
        hasher.update(features)
        hasher.update(n_samples)
        hasher.update(first_row)
    elif hasattr(data, "__dict__"):
        state = data.__dict__
        for key in sorted(state):
            hasher.update(key)
            try:
                hasher.update(state[key])
            except Exception:
                hasher.update(str(state[key]))
    else:
        raise ValueError("Data object is not hashable")
    return hasher.hexdigest()


def generate_cache_dir(
    X, fingerprint, cache_dir=None, root_dir=config.BIOSETS_DATASETS_CACHE
):
    if cache_dir is None:
        if isinstance(root_dir, Path):
            root_dir = root_dir.as_posix()
        if (
            root_dir == config.BIOSETS_DATASETS_CACHE.as_posix()
            and hasattr(X, "cache_files")
            and X.cache_files
        ):
            cache_dir = os.path.dirname(X.cache_files[0]["filename"])
        else:
            cache_dir = _build_cache_dir(
                X,
                fingerprint,
                cache_dir_root=root_dir,
            )

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return Path(cache_dir).resolve().as_posix()
    return None


def get_cache_file_name(cache_dir, fingerprint, cache_file_name=None):
    if cache_file_name:
        if is_file_name(cache_file_name):
            cache_file_name = Path(cache_dir) / Path(cache_file_name).with_suffix(
                ".json"
            )
        else:
            cache_file_name = Path(cache_file_name).with_suffix(".json")
    else:
        cache_file_name = Path(cache_dir) / f"cache-{fingerprint}.json"
    return cache_file_name.resolve().as_posix()


def _relative_data_dir(
    fingerprint, builder_name, dataset_name, version=None, with_hash=True
) -> str:
    """
    Constructs a relative directory path for a dataset based on its properties.

    Args:
        dataset (Dataset): The dataset for which to construct the path.
        with_version (bool, optional): Include version information in the path.
        with_hash (bool, optional): Include hash information in the path.

    Returns:
        str: Relative path for the dataset.
    """
    builder_data_dir = posixpath.join(builder_name, f"{dataset_name}-{fingerprint}")
    if version:
        version = str(version) if isinstance(version, str) else __version__
        builder_data_dir = posixpath.join(builder_data_dir, version)
    if with_hash:
        hash = _hash_python_lines(
            inspect.getsource(
                getattr(importlib.import_module("datasets.table"), "InMemoryTable")
            )
        )
        builder_data_dir = posixpath.join(builder_data_dir, hash)
    return builder_data_dir


def _build_cache_dir(
    obj,
    fingerprint: str,
    cache_dir_root: str = config.BIOSETS_DATASETS_CACHE,
) -> str:
    """
    Builds the cache directory path for storing processed dataset versions.

    Args:
        dataset (Union[Dataset, IterableDataset]): The dataset to cache.
        cache_dir_root (str, optional): Root directory for caching datasets.

    Returns:
        str: The path to the dataset's cache directory.
    """
    if (
        hasattr(obj, "version")
        and hasattr(obj, "config_name")
        and hasattr(obj, "builder_name")
    ):
        version = str(obj.version) if isinstance(obj.version, str) else __version__
        dataset_name = obj.config_name or "default"
        builder_name = obj.builder_name or "in_memory"
    elif (
        hasattr(obj, "config")
        and hasattr(obj.config, "version")
        and hasattr(obj.config, "processor_name")
        and hasattr(obj.config, "processor_type")
    ):
        version = (
            str(obj.config.version)
            if obj.config.version and not str(obj.config.version) == "0.0.0"
            else __version__
        )
        dataset_name = obj.config.processor_name or "default"
        builder_name = obj.config.processor_type or "in"

    else:
        version = __version__
        dataset_name = "default"
        builder_name = "in_memory"

    builder_data_dir = posixpath.join(
        cache_dir_root,
        _relative_data_dir(
            fingerprint=fingerprint,
            builder_name=builder_name,
            dataset_name=dataset_name,
        ),
    )
    version_data_dir = posixpath.join(
        cache_dir_root,
        _relative_data_dir(
            fingerprint=fingerprint,
            builder_name=builder_name,
            dataset_name=dataset_name,
            version=version,
        ),
    )

    def _other_versions_on_disk():
        """Returns previous versions on disk."""
        if not os.path.exists(builder_data_dir):
            return []

        version_dirnames = []
        for dir_name in os.listdir(builder_data_dir):
            try:
                version_dirnames.append((datasets.utils.Version(dir_name), dir_name))
            except ValueError:  # Invalid version (ex: incomplete data dir)
                pass
        version_dirnames.sort(reverse=True)
        return version_dirnames
        # Check and warn if other versions exist

    if not datasets.utils.file_utils.is_remote_url(builder_data_dir):
        version_dirs = _other_versions_on_disk()
        if version_dirs:
            other_version = version_dirs[0][0]
            if other_version != version:
                warn_msg = (
                    f"Found a different version {str(other_version)} of dataset {dataset_name} in "
                    f"cache_dir {cache_dir_root}. Using currently defined version "
                    f"{str(version)}."
                )
                logger.warning(warn_msg)
    if not os.path.exists(version_data_dir):
        os.makedirs(version_data_dir, exist_ok=True)

    return version_data_dir


def update_fingerprint(fingerprint, value, key=None):
    if key is None and value is None:
        return fingerprint
    hasher = Hasher()
    if fingerprint:
        hasher.update(fingerprint)
    if key:
        hasher.update(key)
    try:
        hasher.update(value)
    except Exception:
        return generate_random_fingerprint()
    else:
        return hasher.hexdigest()
