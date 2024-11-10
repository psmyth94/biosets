"""
This module provides functions for loading and manipulating datasets in the biosets package.

Functions:
- load_dataset: Loads a dataset from a specified path.
- concatenate_datasets: Concatenates multiple datasets into a single dataset.
"""

import inspect
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import datasets.config
from biocore.utils.inspect import get_kwargs
from datasets import (
    DownloadMode,
    VerificationMode,
)
from datasets import (
    concatenate_datasets as _concatenate_datasets,
)
from datasets import (
    load_dataset as _load_dataset,
)
from datasets import (
    load_from_disk as _load_from_disk,
)
from datasets.load import load_dataset_builder

from biosets.packaged_modules import (
    _MODULE_SUPPORTS_METADATA,
    _PACKAGED_DATASETS_MODULES,
)
from biosets.streaming import extend_dataset_builder_for_streaming

from .packaged_modules import (
    EXPERIMENT_TYPE_ALIAS,
    EXPERIMENT_TYPE_TO_BUILDER_CLASS,
)


@contextmanager
def patch_dataset_load():
    with patch(
        "datasets.load._PACKAGED_DATASETS_MODULES", _PACKAGED_DATASETS_MODULES
    ), patch(
        "datasets.load._MODULE_SUPPORTS_METADATA", _MODULE_SUPPORTS_METADATA
    ), patch(
        "datasets.builder.extend_dataset_builder_for_streaming",
        extend_dataset_builder_for_streaming,
    ):
        yield


def prepare_load_dataset(
    path,
    data_files=None,
    streaming=False,
    num_proc=None,
    download_mode=None,
    verification_mode=None,
    save_infos=False,
    **kwargs,
):
    """Prepares the arguments for the load_dataset function.

    Args:
    - args: The arguments to prepare.
    - kwargs: The keyword arguments to prepare.

    Returns:
    - The prepared arguments.
    """
    if data_files is not None and not data_files:
        raise ValueError(
            f"Empty 'data_files': '{data_files}'. It should be either non-empty or None (default)."
        )
    if Path(path, datasets.config.DATASET_STATE_JSON_FILENAME).exists():
        raise ValueError(
            "You are trying to load a dataset that was saved using `save_to_disk`. "
            "Please use `load_from_disk` instead."
        )

    if streaming and num_proc is not None:
        raise NotImplementedError(
            "Loading a streaming dataset in parallel with `num_proc` is not implemented. "
            "To parallelize streaming, you can wrap the dataset with a PyTorch DataLoader using `num_workers` > 1 instead."
        )

    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    verification_mode = VerificationMode(
        (verification_mode or VerificationMode.BASIC_CHECKS)
        if not save_infos
        else VerificationMode.ALL_CHECKS
    )
    return (
        path,
        {
            "data_files": data_files,
            "download_mode": download_mode,
            "verification_mode": verification_mode,
            **kwargs,
        },
    )


@wraps(_load_dataset)
def load_dataset(*args, **kwargs):
    """Loads a dataset from a specified path.

    Args:
    - path: The path to the dataset.
    - experiment_type: The type of experiment that was used to generate the dataset.
    - name: The name of the dataset to load.
    - data_files: The data files to load.
    - split: The split to load.
    - cache_dir: The cache directory to use.
    - features: The features to load.
    - download_config: The download configuration to use.
    - ignore_verifications: Whether to ignore verifications.
    - builder_kwargs: The builder keyword arguments to use.
    - download_and_prepare_kwargs: The download and prepare keyword arguments to use.
    - as_supervised: Whether to load the dataset as supervised.
    - with_info: Whether to load the dataset with info.
    - as_dataset: Whether to load the dataset as a dataset.
    - shuffle_files: Whether to shuffle the files.
    - read_config: The read configuration to use.
    - write_config: The write configuration to use.
    - **kwargs: Additional keyword arguments to use.
    """

    if "streaming" in kwargs and kwargs["streaming"]:
        raise NotImplementedError(
            "Loading a streaming dataset is not implemented. "
            "To load a streaming dataset, you can use the `datasets.load_dataset` instead"
        )

    load_dataset_args = inspect.signature(_load_dataset).parameters.keys()
    args_to_kwargs = {
        k: v for k, v in zip(load_dataset_args, args) if k in load_dataset_args
    }
    kwargs.update(args_to_kwargs)
    path = kwargs.pop("path", None)

    if path is not None and not (
        path in EXPERIMENT_TYPE_ALIAS.keys()
        or path in EXPERIMENT_TYPE_TO_BUILDER_CLASS.keys()
    ):
        path, new_kwargs = prepare_load_dataset(path, **kwargs)
        load_dataset_builder_kwargs = get_kwargs(kwargs, load_dataset_builder)

        with patch_dataset_load():
            dataset_builder = load_dataset_builder(path, **load_dataset_builder_kwargs)
        dataset_builder_args = inspect.signature(
            load_dataset_builder.__init__
        ).parameters.keys()
        matching_args = set(dataset_builder_args).intersection(load_dataset_args)
        new_kwargs.update({k: getattr(dataset_builder, k) for k in matching_args})

        builder_config = dataset_builder.config
        builder_config_args = inspect.signature(
            builder_config.__init__
        ).parameters.keys()
        matching_args = set(builder_config_args).intersection(load_dataset_args)
        new_kwargs.update({k: getattr(builder_config, k) for k in matching_args})

        builder_info = dataset_builder.info
        builder_info_args = inspect.signature(builder_info.__init__).parameters.keys()
        matching_args = set(builder_info_args).intersection(load_dataset_args)
        new_kwargs.update({k: getattr(builder_info, k) for k in matching_args})

        experiment_type = new_kwargs.pop("experiment_type", "biodata")
        new_kwargs["module_path"] = inspect.getfile(dataset_builder.__class__)
        return load_dataset(
            **new_kwargs,
            experiment_type=experiment_type,
        )

    path = kwargs.pop("experiment_type", "biodata")
    path = EXPERIMENT_TYPE_ALIAS.get(path, path)
    # only patch if we are loading a custom packaged module
    if path in EXPERIMENT_TYPE_TO_BUILDER_CLASS.keys():
        if "builder_kwargs" in kwargs:
            kwargs["builder_kwargs"].update(kwargs)
        else:
            kwargs["builder_kwargs"] = kwargs
            # uncomment for debugging

        with patch_dataset_load():
            return _load_dataset(path, **kwargs)
    else:
        return _load_dataset(path, *args, **kwargs)


def concatenate_datasets(dsets, info=None, split=None, axis=0):
    if axis == 1:
        cols = set(dsets[0].column_names)
        for i in range(1, len(dsets)):
            if any(col in cols for col in dsets[i].column_names):
                cols_i = [col for col in dsets[i].column_names if col not in cols]
                dsets[i] = dsets[i].select_columns(cols_i)
            cols.update(dsets[i].column_names)
    out = _concatenate_datasets(dsets, info=info, split=split, axis=axis)
    # out.replays[-1][-1]["axis"] = axis
    return out


@wraps(_load_from_disk)
def load_from_disk(
    file_path,
    fs="deprecated",
    keep_in_memory: Optional[dict] = None,
    storage_options: Optional[dict] = None,
):
    return _load_from_disk(
        file_path,
        fs=fs,
        keep_in_memory=keep_in_memory,
        storage_options=storage_options,
    )


load_dataset.__doc__ = _load_dataset.__doc__
