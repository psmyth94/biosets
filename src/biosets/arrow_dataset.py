import copy
import os
from functools import wraps
from typing import TYPE_CHECKING, List, Union

import pyarrow as pa
from datasets import Dataset, Features
from datasets.arrow_dataset import update_metadata_with_features

from biosets.features import (
    METADATA_FEATURE_TYPES,
    TARGET_FEATURE_TYPES,
    Batch,
    Sample,
)
from biosets.utils import logging

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

PathLike = Union[str, bytes, os.PathLike]


class Bioset(Dataset):
    def cleanup_cache_files(self) -> int:
        EXT_TO_DELETE = [".arrow", ".json", ".joblib", ".png", ".jpeg", ".jpg"]

        current_cache_files = [
            os.path.abspath(cache_file["filename"]) for cache_file in self.cache_files
        ]
        if not current_cache_files:
            return 0
        cache_directory = os.path.dirname(current_cache_files[0])
        logger.info(f"Listing files in {cache_directory}")
        files: List[str] = os.listdir(cache_directory)
        files_to_remove = []
        for f_name in files:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            if f_name.startswith("cache-") and any(
                f_name.endswith(ext) for ext in EXT_TO_DELETE
            ):
                if full_name in current_cache_files:
                    logger.info(f"Keeping currently used cache file at {full_name}")
                    continue
                files_to_remove.append(full_name)
        for file_path in files_to_remove:
            logger.info(f"Removing {file_path}")
            os.remove(file_path)
        return len(files_to_remove)

    @wraps(Dataset.train_test_split)
    def train_test_split(self, *args, **kwargs):
        return super().train_test_split(*args, **kwargs)

    @wraps(Dataset._get_cache_file_path)
    def _get_cache_file_path(self, *args, **kwargs):
        return super()._get_cache_file_path(*args, **kwargs)

    @wraps(Dataset._new_dataset_with_indices)
    def _new_dataset_with_indices(self, *args, **kwargs):
        return super()._new_dataset_with_indices(*args, **kwargs)

    # @wraps(Dataset.add_column)
    # def add_column(self, *args, **kwargs):
    #     return super().add_column(*args, **kwargs)

    # @wraps(Dataset.add_item)
    # def add_item(self, *args, **kwargs):
    #     return super().add_item(*args, **kwargs)

    @wraps(Dataset.cast_column)
    def cast_column(self, *args, **kwargs):
        return super().cast_column(*args, **kwargs)

    @wraps(Dataset.filter)
    def filter(self, *args, **kwargs):
        return super().filter(*args, **kwargs)

    @wraps(Dataset.flatten)
    def flatten(self, *args, **kwargs):
        return super().flatten(*args, **kwargs)

    @wraps(Dataset.flatten_indices)
    def flatten_indices(self, *args, **kwargs):
        return super().flatten_indices(*args, **kwargs)

    @wraps(Dataset.map)
    def map(self, *args, **kwargs):
        return super().map(*args, **kwargs)

    @wraps(Dataset.remove_columns)
    def remove_columns(self, *args, **kwargs):
        return super().remove_columns(*args, **kwargs)

    @classmethod
    @wraps(Dataset.from_csv)
    def from_csv(cls, *args, **kwargs):
        return super().from_csv(*args, **kwargs)

    @classmethod
    @wraps(Dataset.from_file)
    def from_file(self, *args, **kwargs):
        return super().from_file(*args, **kwargs)

    @wraps(Dataset.rename_column)
    def rename_column(self, *args, **kwargs):
        return super().rename_column(*args, **kwargs)

    @wraps(Dataset.rename_columns)
    def rename_columns(self, *args, **kwargs):
        return super().rename_columns(*args, **kwargs)

    @wraps(Dataset.select)
    def select(self, *args, **kwargs):
        return super().select(*args, **kwargs)

    @wraps(Dataset.select_columns)
    def select_columns(self, *args, **kwargs):
        return super().select_columns(*args, **kwargs)

    @wraps(Dataset.shuffle)
    def shuffle(self, *args, **kwargs):
        return super().shuffle(*args, **kwargs)

    @wraps(Dataset.sort)
    def sort(self, *args, **kwargs):
        return super().sort(*args, **kwargs)

    @wraps(Dataset.save_to_disk)
    def save_to_disk(self, *args, **kwargs):
        return super().save_to_disk(*args, **kwargs)

    @classmethod
    @wraps(Dataset.load_from_disk)
    def load_from_disk(self, *args, **kwargs):
        return super().load_from_disk(*args, **kwargs)


def get_sample_col_name(X):
    sample_column = [k for k, v in X._info.features.items() if isinstance(v, Sample)]
    if sample_column:
        sample_column = sample_column[0]
    else:
        sample_column = None
    return sample_column


def get_batch_col_name(X):
    batch_column = [k for k, v in X._info.features.items() if isinstance(v, Batch)]
    if batch_column:
        batch_column = batch_column[0]
    else:
        batch_column = None
    return batch_column


def get_metadata_col_names(X):
    metadata_columns = [
        k for k, v in X._info.features.items() if isinstance(v, METADATA_FEATURE_TYPES)
    ]
    if not metadata_columns:
        metadata_columns = None
    return metadata_columns


def get_target_col_names(X, flatten=True):
    target_columns = [
        k for k, v in X._info.features.items() if isinstance(v, TARGET_FEATURE_TYPES)
    ]
    if target_columns:
        if len(target_columns) == 1 and flatten:
            target_columns = target_columns[0]
    else:
        target_columns = None
    return target_columns


def get_data_col_names(X):
    data_columns = [
        k
        for k, v in X._info.features.items()
        if not isinstance(v, (METADATA_FEATURE_TYPES, TARGET_FEATURE_TYPES))
    ]
    return data_columns


def get_target(X, decode=False) -> Bioset:
    target_columns = get_target_col_names(X, flatten=False)
    if target_columns:
        out = X.select_columns(target_columns)
        return decode(out, target_columns[0]) if decode else out
    return None


def decode(X, target_name=None) -> Bioset:
    if target_name is None:
        target_name = X.column_names[0]
    label_feature = copy.deepcopy(X._info.features[target_name])
    if not isinstance(label_feature, TARGET_FEATURE_TYPES):
        return X
    target = copy.deepcopy(X)
    int2str = dict(zip(range(len(label_feature._int2str)), label_feature._int2str))
    target._data.table = target._data.table.set_column(
        target._data.table.column_names.index(target_name),
        pa.field(target_name, pa.string()),
        pa.array(
            [
                int2str.get(i, None) if isinstance(i, int) else i
                for i in target._data.table.column(target_name).to_numpy().tolist()
            ],
        ),
    )
    label_feature.dtype = "string"
    label_feature.pa_type = pa.string()
    target._info.features = Features(
        {
            col: target._info.features[col] if col != target_name else label_feature
            for col in target._info.features
        }
    )
    target._data = update_metadata_with_features(target._data, target._info.features)
    return target


def get_sample_metadata(X) -> Bioset:
    metadata_columns = get_metadata_col_names(X)
    target_columns = get_target_col_names(X, flatten=False)
    if metadata_columns:
        out = X.select_columns(metadata_columns)
    else:
        return None
    if target_columns:
        for label in target_columns:
            out = decode(out, label)
    return out


def get_data(X) -> Bioset:
    data_columns = get_data_col_names(X)
    if data_columns:
        return X.select_columns(data_columns)
    return None


def get_feature_metadata(X) -> dict:
    out = get_data(X)
    feat_metadata = {
        k: v.metadata for k, v in out._info.features.items() if hasattr(v, "metadata")
    }
    return feat_metadata


def cleanup_cache_files(cache_directory) -> int:
    """Clean up all cache files in the dataset cache directory, excepted the currently used cache file if there is
    one.

    Be careful when running this command that no other process is currently using other cache files.

    Returns:
        `int`: Number of removed files.

    Example:

    ```py
    >>> from datasets import load_dataset
    >>> ds = load_dataset("rotten_tomatoes", split="validation")
    >>> ds.cleanup_cache_files()
    10
    ```
    """
    count = 0
    for root, dirs, files in os.walk(cache_directory):
        files_to_delete = []
        delete_files = False
        for f in files:
            files_to_delete.append(os.path.join(root, f))
            if (
                (f.startswith("cache-") and f.endswith(".arrow"))
                or f.endswith("train.arrow")
                or f.endswith("test.arrow")
            ):
                delete_files = True
        if delete_files:
            count += len(files_to_delete)
            for f in files_to_delete:
                os.remove(f)
            # delete the directory if it is empty
            if not os.listdir(root):
                os.rmdir(root)
    return count
