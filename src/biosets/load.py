"""
This module provides functions for loading and manipulating datasets in the biosets package.

Functions:
- load_dataset: Loads a dataset from a specified path.
- concatenate_datasets: Concatenates multiple datasets into a single dataset.
"""

import inspect
from functools import wraps
from typing import Optional

from datasets import (
    concatenate_datasets as _concatenate_datasets,
)

# functions to use patch context on
from datasets import (
    load_dataset as _load_dataset,
)
from datasets import (
    load_from_disk as _load_from_disk,
)

from .packaged_modules import DATASET_NAME_ALIAS, DATASET_NAME_TO_OMIC_TYPE


@wraps(_load_dataset)
def load_dataset(*args, **kwargs):
    """Loads a dataset from a specified path.

    Args:
    - path: The path to the dataset.
    - dataset_type: The type of dataset to load.
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
    if len(args) > 0:
        path = args[0]
    else:
        path = kwargs.pop("path", None)
    path = kwargs.pop("experiment_type", path)
    path = DATASET_NAME_ALIAS.get(path, path)
    args = (path, *args[1:])
    # only patch if we are loading a custom packaged module
    if path in DATASET_NAME_TO_OMIC_TYPE.keys():
        from biosets.integration import DatasetsPatcher

        load_dataset_args = inspect.signature(_load_dataset).parameters.keys()
        with DatasetsPatcher():
            builder_kwargs = {
                k: v for k, v in zip(load_dataset_args, args) if k in load_dataset_args
            }
            builder_kwargs.update(kwargs)

            kwargs["builder_kwargs"] = builder_kwargs
            return _load_dataset(*args, **kwargs)
    else:
        return _load_dataset(*args, **kwargs)


def concatenate_datasets(dsets, info=None, split=None, axis=0):
    from biosets.integration import DatasetsPatcher

    with DatasetsPatcher():
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
    keep_in_memory: Optional[dict | None] = None,
    storage_options: Optional[dict | None] = None,
):
    from biosets.integration import DatasetsPatcher

    with DatasetsPatcher():
        return _load_from_disk(
            file_path,
            fs=fs,
            keep_in_memory=keep_in_memory,
            storage_options=storage_options,
        )


load_dataset.__doc__ = _load_dataset.__doc__
