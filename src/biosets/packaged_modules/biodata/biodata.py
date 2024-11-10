import inspect
import itertools
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import datasets
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
import pyarrow.parquet as pq
from biocore.data_handling import DataHandler
from biocore.utils.import_util import is_polars_available
from datasets.data_files import (
    DataFilesDict,
    DataFilesList,
    DataFilesPatternsList,
    sanitize_patterns,
)
from datasets.features.features import ClassLabel, Features, Value
from datasets.naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from datasets.packaged_modules.arrow import arrow as hf_arrow
from datasets.packaged_modules.csv import csv as hf_csv
from datasets.packaged_modules.json import json as hf_json
from datasets.packaged_modules.parquet import parquet as hf_parquet
from datasets.utils.py_utils import asdict, tqdm

from biosets.data_files import (
    FEATURE_METADATA_PATTERNS,
    SAMPLE_METADATA_PATTERNS,
    get_feature_metadata_patterns,
    get_metadata_patterns,
)
from biosets.features import (
    Batch,
    BinClassLabel,
    Metadata,
    RegressionTarget,
    Sample,
    ValueWithMetadata,
)
from biosets.packaged_modules.csv import csv as biosets_csv
from biosets.packaged_modules.npz.npz import SparseReaderConfig
from biosets.utils import (
    as_py,
    is_file_name,
    logging,
)

if TYPE_CHECKING:
    import polars as pl
    from datasets.builder import ArrowBasedBuilder

logger = logging.get_logger(__name__)

SAMPLE_COLUMN = "samples"
BATCH_COLUMN = "batches"
METADATA_COLUMN = "metadata"
TARGET_COLUMN = "labels"
FEATURE_COLUMN = "features"
DTYPE_MAP = {
    "double": "float32",
    "object": "string",
    "int": "int64",
    "float": "float32",
}


def is_regression_type(table_or_id) -> bool:
    if isinstance(table_or_id, pd.Series):
        return pdt.is_float_dtype(table_or_id)
    elif isinstance(table_or_id, pa.DataType):
        return (
            pa.types.is_floating(table_or_id)
            or pa.types.is_temporal(table_or_id)
            or pa.types.is_decimal(table_or_id)
        )
    elif isinstance(table_or_id, pa.Table):
        return table_or_id.dtype.is_float()
    elif isinstance(table_or_id, (pa.ChunkedArray, pa.Array)):
        return pa.types.is_floating(table_or_id.type)


def is_classification_type(
    id: pa.DataType, range: Optional[Tuple[Any, Any]] = None
) -> bool:
    return (
        pa.types.is_boolean(id)
        or pa.types.is_string(id)
        or pa.types.is_large_string(id)
        or pa.types.is_integer(id)
    )


class MetadataColumnNotFound(Exception):
    """Raised when the metadata column is not found in the table"""


class SampleColumnNotFound(Exception):
    """Raised when the sample column is not found in the table"""


class BatchColumnNotFound(Exception):
    """Raised when the batch column is not found in the table"""


class LabelColumnNotFound(Exception):
    """Raised when the label column is not found in the table"""


class FeatureColumnNotFound(Exception):
    """Raised when the feature column is not found in the table"""


class InvalidPath(Exception):
    """Raised when an invalid path is provided"""


SAMPLE_COLUMN_WARN_MSG = (
    "Could not find the sample column in the sample metadata table.\n"
    "Please provide the sample column by setting the `sample_column` argument. For example:\n"
    "   >>> `load_dataset(..., sample_column='samples')`.\n"
    "Sample metadata will be ignored."
)

FEATURE_COLUMN_WARN_MSG = (
    "Could not find the feature column in the feature metadata table.\n"
    "Please provide the feature column by setting the `feature_column` argument. For example:\n"
    "   >>> `load_dataset(..., feature_column='my_feature_column')`.\n"
    "Feature metadata will be ignored."
)

_COLUMN_TO_ERROR = {
    SAMPLE_COLUMN: SampleColumnNotFound,
    BATCH_COLUMN: BatchColumnNotFound,
    METADATA_COLUMN: MetadataColumnNotFound,
    TARGET_COLUMN: LabelColumnNotFound,
    FEATURE_COLUMN: FeatureColumnNotFound,
}


@dataclass
class BioDataConfig(datasets.BuilderConfig):
    """BuilderConfig for AutoFolder."""

    features: Optional[datasets.Features] = None

    sample_metadata_dir: Optional[str] = None
    sample_metadata_files: Optional[DataFilesDict] = None

    feature_metadata_dir: Optional[str] = None
    feature_metadata_files: Optional[DataFilesDict] = None

    labels: Optional[List[Any]] = None
    positive_labels: Optional[List[Any]] = None
    negative_labels: Optional[List[Any]] = None
    positive_label_name: Optional[str] = None
    negative_label_name: Optional[str] = None

    batch_column: Optional[str] = None
    sample_column: Optional[str] = None
    metadata_columns: Optional[List[str]] = None
    data_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    feature_column: Optional[str] = None
    columns: Optional[List[str]] = None

    drop_labels: bool = None
    drop_samples: bool = None
    drop_batches: bool = None
    drop_metadata: bool = None
    drop_feature_metadata: bool = None

    builder_kwargs: dict = None
    data_kwargs: dict = None
    metadata_kwargs: dict = None
    feature_metadata_kwargs: dict = None

    use_polars: bool = True

    rows_are_features: bool = False
    use_first_schema: Optional[bool] = None
    add_missing_columns: bool = False
    zero_as_missing: bool = False

    module_path: Optional[str] = None

    EXTENSION_MAP = {
        ".csv": ("csv", biosets_csv.CsvConfig, hf_csv.CsvConfig),
        ".tsv": ("csv", biosets_csv.CsvConfig, hf_csv.CsvConfig),
        ".txt": ("csv", biosets_csv.CsvConfig, hf_csv.CsvConfig),
        ".json": ("json", hf_json.JsonConfig, None),
        ".jsonl": ("json", hf_json.JsonConfig, None),
        ".parquet": ("parquet", hf_parquet.ParquetConfig, None),
        ".arrow": ("arrow", hf_arrow.ArrowConfig, None),
        ".npz": ("npz", SparseReaderConfig, None),
    }

    ALLOWED_METADATA_EXTENSIONS = [
        ".csv",
        ".tsv",
        ".txt",
        ".json",
        ".jsonl",
        ".parquet",
        ".arrow",
    ]
    DEFAULT_SAMPLE_METADATA_PATTERNS = SAMPLE_METADATA_PATTERNS
    DEFAULT_FEATURE_METADATA_PATTERNS = FEATURE_METADATA_PATTERNS

    def __post_init__(self):
        # Check for invalid characters in the config name
        if any(char in self.name for char in INVALID_WINDOWS_CHARACTERS_IN_PATH):
            raise datasets.builder.InvalidConfigName(
                f"Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{self.name}'. "
                f"They could create issues when creating a directory for this config on Windows filesystem."
            )

        # Validate and process data_files
        if not self.data_files:
            raise ValueError(
                "At least one data file must be specified, but got data_files=None"
            )
        if not isinstance(self.data_files, DataFilesDict):
            raise ValueError(
                f"Expected a DataFilesDict in data_files but got {self.data_files}"
            )

        if not self.sample_metadata_files:
            sample_metadata_dir = self.sample_metadata_dir or self.data_dir

            if sample_metadata_dir is not None:
                try:
                    sample_metadata_patterns = get_metadata_patterns(
                        sample_metadata_dir
                    )
                    if sample_metadata_patterns:
                        self.sample_metadata_files = DataFilesDict.from_patterns(
                            sample_metadata_patterns,
                            base_path=sample_metadata_dir,
                            allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
                        )
                except FileNotFoundError:
                    pass

        if not self.feature_metadata_files:
            feature_metadata_dir = self.feature_metadata_dir or self.data_dir
            if feature_metadata_dir is not None:
                try:
                    feature_metadata_patterns = get_feature_metadata_patterns(
                        feature_metadata_dir
                    )
                    if feature_metadata_patterns:
                        self.feature_metadata_files = DataFilesDict.from_patterns(
                            feature_metadata_patterns,
                            base_path=feature_metadata_dir,
                            allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
                        )
                except FileNotFoundError:
                    pass

        self.sample_metadata_files = self._ensure_metadata_files_dict(
            self.sample_metadata_files, self.data_files, is_sample=True
        )
        self.feature_metadata_files = self._ensure_metadata_files_dict(
            self.feature_metadata_files, self.data_files, is_sample=False
        )

        # Separate metadata files from data_files
        data_files_dict = {}
        for split, files in self.data_files.items():
            valid_files, origin_metadata = [], []
            for file, metadata in zip(files, self.data_files[split].origin_metadata):
                if not (
                    self.sample_metadata_files
                    and split in self.sample_metadata_files
                    and file in self.sample_metadata_files[split]
                ) and not (
                    self.feature_metadata_files
                    and split in self.feature_metadata_files
                    and file in self.feature_metadata_files[split]
                ):
                    valid_files.append(file)
                    origin_metadata.append(metadata)
            data_files_dict[split] = DataFilesList(valid_files, origin_metadata)
        self.data_files = DataFilesDict(data_files_dict)

        # Process metadata files
        self.sample_metadata_files = self._process_metadata_files(
            self.sample_metadata_files,
            self.sample_metadata_dir,
            get_metadata_patterns,
            is_sample=True,
        )
        self.feature_metadata_files = self._process_metadata_files(
            self.feature_metadata_files,
            self.feature_metadata_dir,
            get_feature_metadata_patterns,
            is_sample=False,
        )

        # Validate the number of sample metadata files
        for split in self.sample_metadata_files:
            num_data_files = len(self.data_files.get(split, []))
            num_metadata_files = len(self.sample_metadata_files[split])
            if (
                num_metadata_files > 1
                and num_data_files > 1
                and num_metadata_files != num_data_files
            ):
                raise ValueError(
                    "The number of sharded sample metadata files must match the number "
                    f"of sharded data files in split '{split}'."
                )

        self.data_kwargs = self._get_builder_kwargs(self.data_files)

    def _ensure_list(self, value):
        if value is None:
            return []
        return [value] if isinstance(value, (str, Path)) else value

    def _ensure_metadata_files_dict(self, metadata_files, data_files, is_sample):
        if metadata_files is None:
            return defaultdict(list)
        if isinstance(metadata_files, dict):
            missing_keys = set(metadata_files.keys()) - set(data_files.keys())
            if missing_keys:
                file_type = "Sample" if is_sample else "Feature"
                raise ValueError(
                    f"{file_type} metadata files contain keys {missing_keys} which are not present "
                    "in data_files."
                )
            for split in metadata_files:
                metadata_files[split] = self._ensure_list(metadata_files[split])
            if not is_sample:
                missing_keys = set(data_files.keys()) - set(metadata_files.keys())
                if missing_keys:
                    for split in missing_keys:
                        metadata_files[split] = metadata_files[
                            next(iter(metadata_files))
                        ]
            return metadata_files
        else:
            metadata_files = self._ensure_list(metadata_files)
            metadata_files_dict = {}
            if is_sample and len(data_files) > 1:
                raise ValueError(
                    "When data_files has multiple splits, sample_metadata_files must "
                    "be a dict with matching keys."
                )
            if not is_sample and isinstance(metadata_files, list):
                # For feature_metadata_files, copy to all splits if data_files has multiple keys
                if len(data_files) > 1:
                    for split in data_files.keys():
                        metadata_files_dict[split] = metadata_files
                else:
                    split = next(iter(data_files))
                    metadata_files_dict[split] = metadata_files
            else:
                split = next(iter(data_files))
                metadata_files_dict[split] = metadata_files
            return metadata_files_dict

    def _process_metadata_files(
        self, metadata_files_dict, metadata_dir, get_patterns_func, is_sample
    ):
        processed_metadata_files = {}

        if metadata_dir and not metadata_files_dict:
            base_path = Path(metadata_dir).expanduser().resolve()
            if not base_path.is_dir():
                raise FileNotFoundError(
                    f"Directory {base_path} does not exist or is not a directory."
                )
            base_path = base_path.as_posix()

            metadata_patterns = get_patterns_func(base_path)
            metadata_files = (
                DataFilesPatternsList.from_patterns(
                    metadata_patterns,
                    allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
                ).resolve(base_path)
                if metadata_patterns
                else []
            )
        for split, metadata_files in metadata_files_dict.items():
            if metadata_files and not isinstance(metadata_files, DataFilesPatternsList):
                data_dir = ""

                metadata_files = self._ensure_list(metadata_files)
                # Check for invalid characters in the config name
                metadata_files = [Path(f).resolve().as_posix() for f in metadata_files]
                if len(metadata_files) > 1:
                    data_dir = os.path.commonpath(metadata_files)
                else:
                    first_file = metadata_files[0]
                    if is_file_name(first_file):
                        paths = [f for files in self.data_files.values() for f in files]
                        data_dir = (
                            os.path.commonpath(paths)
                            if len(paths) > 1
                            else Path(paths[0]).parent.resolve().as_posix()
                        )
                    else:
                        data_dir = Path(first_file).parent.resolve().as_posix()
                        metadata_files = [Path(f).name for f in metadata_files]
                metadata_patterns = next(
                    iter(sanitize_patterns(metadata_files).values())
                )
                metadata_files = DataFilesPatternsList.from_patterns(
                    metadata_patterns,
                    allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
                ).resolve(data_dir)
            processed_metadata_files[split] = metadata_files
        return DataFilesDict(processed_metadata_files)

    def _get_builder_kwargs(self, files):
        if files is None:
            return {}
        builder_kwargs = {}
        iter_files = files

        if isinstance(files, (str, list, tuple)):
            iter_files = [files]
        if isinstance(iter_files, (dict, DataFilesDict)):
            iter_files = iter_files.values()
        for file in itertools.chain.from_iterable(iter_files):
            file_ext = os.path.splitext(file)[-1].lower()

            config_path, _config_class, hf_config_class = self.EXTENSION_MAP.get(
                file_ext, (None, None, None)
            )

            if config_path is None and self.module_path is not None:
                # in case file path is a zip file and the module path is provided
                # from datasets.load_dataset_builder within `biosets.load.load_dataset`
                file_ext = "." + os.path.split(self.module_path)[-1].split(".")[0]
                config_path, _config_class, hf_config_class = self.EXTENSION_MAP.get(
                    file_ext, (None, None, None)
                )
            if config_path is not None:
                builder_kwargs = {}
                builder_kwargs["data_files"] = files
                if hf_config_class:
                    builder_kwargs["hf_kwargs"] = {}
                if self.builder_kwargs:
                    for k, v in self.builder_kwargs.items():
                        if k in ["data_dir", "name"]:
                            continue
                        if k in _config_class.__dataclass_fields__:
                            builder_kwargs[k] = v
                        if (
                            hf_config_class
                            and k in hf_config_class.__dataclass_fields__
                        ):
                            builder_kwargs["hf_kwargs"][k] = v
                for k, v in self.__dict__.items():
                    if k in ["data_dir", "name"]:
                        continue
                    if k in _config_class.__dataclass_fields__:
                        builder_kwargs[k] = v
                    if hf_config_class and k in hf_config_class.__dataclass_fields__:
                        builder_kwargs["hf_kwargs"][k] = v
                if file_ext in [".tsv", ".txt"]:
                    if is_polars_available() and "separator" not in builder_kwargs:
                        builder_kwargs["separator"] = "\t"
                    if (
                        "hf_kwargs" in builder_kwargs
                        and "sep" not in builder_kwargs["hf_kwargs"]
                    ):
                        builder_kwargs["hf_kwargs"]["sep"] = "\t"
                module_path = inspect.getfile(_config_class)
                self.config_path = config_path
                self.module_path = module_path
                if "datasets" == _config_class.__module__.split(".")[0]:
                    builder_kwargs["path"] = config_path
                else:
                    builder_kwargs["path"] = config_path
                    # builder_kwargs["path"] = module_path
                    # builder_kwargs["trust_remote_code"] = True
                break
        return builder_kwargs


class BioData(datasets.ArrowBasedBuilder):
    """
    Base class for generic data loaders for bioinformatics data stored in folders.


    Abstract class attributes to be overridden by a child class:
        SAMPLE_COLUMN_NAME: the name of the column containing the sample IDs
        BATCH_COLUMN_NAME: the name of the column containing the batch IDs
        feature_column: the name of the column containing the feature_metadata IDs
        SAMPLE_COLUMN_PATTERNS: the list of default column names containing the sample IDs
        BATCH_COLUMN_PATTERNS: the list of default column names containing the batch IDs
        DEFAULT_FEATURE_METADATA_COLUMNS: the list of default column names containing the feature_metadata IDs
        BUILDER_CONFIG_CLASS: the builder config class to use
        EXTENSIONS: the list of supported file extensions
        CLASSIFICATION_TASK: the classification (or regression) task to use
        DEFAULT_METADATA_FILENAMES: the list of default metadata filenames
        DEFAULT_FEATURE_METADATA_FILENAMES: the list of default feature_metadata filenames
    """

    config: BioDataConfig

    SAMPLE_COLUMN: str = SAMPLE_COLUMN
    BATCH_COLUMN: str = BATCH_COLUMN
    METADATA_COLUMN: str = METADATA_COLUMN
    TARGET_COLUMN: str = TARGET_COLUMN
    FEATURE_COLUMN: str = "features"
    SAMPLE_METADATA_COLUMNS: List[str] = None

    INPUT_FEATURE = ValueWithMetadata  # must accept metadata argument

    BUILDER_CONFIG_CLASS = BioDataConfig
    EXTENSIONS: List[str]
    # CLASSIFICATION_TASK = BioinformaticsClassification(
    #     sample_column=SAMPLE_COLUMN,
    #     batch_column=BATCH_COLUMN,
    #     metadata_column=METADATA_COLUMN,
    #     target_column=TARGET_COLUMN,
    # )

    SAMPLE_COLUMN_PATTERNS: List[str] = ["sample", "name", "id"]
    BATCH_COLUMN_PATTERNS: List[str] = ["batch"]
    LABEL_COLUMN_PATTERNS: List[str] = ["label", "target", "class", "y*"]
    FEATURE_COLUMN_PATTERNS: List[str] = ["feat", "gene", "prot", "anno", "peptide"]

    _input_schema: dict = None
    rename_map = {}
    _metadata_tables: Dict[str, pa.Table] = {}
    _drop_columns = set()
    _all_columns = set()
    _label_map: dict = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _set_columns(
        self,
        data_features: List[str],
        sample_metadata: Union["pl.DataFrame", pd.DataFrame] = None,
        feature_metadata=None,
    ):
        sample_metadata_columns = (
            list(sample_metadata.columns) if sample_metadata is not None else None
        )

        self.config.sample_column = _infer_column_name(
            column_name=self.config.sample_column,
            default_column_name=self.SAMPLE_COLUMN,
            features=data_features,
            metadata_columns=sample_metadata_columns,
            patterns=self.SAMPLE_COLUMN_PATTERNS,
            required=None,
            is_index=True,
        )

        self.config.batch_column = _infer_column_name(
            column_name=self.config.batch_column,
            default_column_name=self.BATCH_COLUMN,
            features=data_features,
            metadata_columns=sample_metadata_columns,
            patterns=self.BATCH_COLUMN_PATTERNS,
            required=None,
        )

        self.config.target_column = _infer_column_name(
            column_name=self.config.target_column,
            default_column_name=self.TARGET_COLUMN,
            features=data_features,
            metadata_columns=sample_metadata_columns,
            patterns=self.LABEL_COLUMN_PATTERNS,
            required=None,
        )

        if (
            self.config.target_column
            and self.config.target_column == self.TARGET_COLUMN
        ):
            self.TARGET_COLUMN = f"{self.TARGET_COLUMN}_"
        excluded_cols = [
            self.config.sample_column,
            self.config.batch_column,
            self.TARGET_COLUMN,
        ]
        if self.config.metadata_columns:
            not_found_cols = set()
            _metadata_cols = []
            # first check if the metadata table was given
            cols = set(data_features)
            if sample_metadata_columns:
                cols.update(sample_metadata_columns)

            for c in self.config.metadata_columns:
                if c not in excluded_cols:
                    if c not in cols:
                        not_found_cols.add(c)
                    else:
                        _metadata_cols.append(c)

            # since the metadata columns were specified, we notify the user of any columns that were not found
            if not_found_cols:
                logger.warning_once(
                    f"Could not find the following metadata columns in the table: {not_found_cols}"
                )

        # gather the sample metadata columns if not already specified
        if not self.config.metadata_columns and sample_metadata_columns:
            self.config.metadata_columns = [
                c for c in sample_metadata_columns if c not in excluded_cols
            ]

        if feature_metadata:
            feature_metadata_columns = list(feature_metadata.column_names)
            self.config.feature_column = _infer_column_name(
                column_name=self.config.feature_column,
                default_column_name=self.FEATURE_COLUMN,
                features=None,
                metadata_columns=feature_metadata_columns,
                patterns=self.FEATURE_COLUMN_PATTERNS,
                required=None,
            )
            if self.config.feature_column is None:
                dfeat = set(data_features)
                for col in feature_metadata.columns:
                    if len(dfeat - set(col.to_numpy())) == 0:
                        self.config.feature_column = col.name
                        break
            if self.config.feature_column:
                target_schema = feature_metadata.schema

                field_idx = target_schema.get_field_index(self.config.feature_column)
                target_schema = target_schema.set(
                    field_idx, pa.field(self.config.feature_column, pa.string())
                )
                feature_metadata = feature_metadata.cast(target_schema)
                missing_columns = set(
                    feature_metadata.column(self.config.feature_column).to_numpy()
                ) - set(data_features)
                if missing_columns:
                    logger.warning_once(
                        f"Could not find the following columns in the data table: {missing_columns}"
                    )
            else:
                logger.warning_once(FEATURE_COLUMN_WARN_MSG)

        return self

    def _convert_feature_metadata_to_dict(self, feature_metadata):
        if feature_metadata is None:
            raise ValueError("Feature metadata is not available.")
        if len(feature_metadata) == 0:
            raise ValueError("Feature metadata provided is empty.")

        _feature_metadata = {}
        features = feature_metadata.column(self.config.feature_column).to_pylist()
        metadata = feature_metadata.drop([self.config.feature_column]).to_pylist()
        _feature_metadata = {str(k): as_py(v) for k, v in zip(features, metadata)}
        return _feature_metadata

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )

        generator = datasets.load_dataset_builder(**self.config.data_kwargs)
        data_splits = generator._split_generators(dl_manager)

        splits = []
        for data_split in data_splits:
            # retrieve the labels from the metadata table if not already specified

            if self.config.features is None:
                # don't rely on the features from the data file
                if (
                    self.config.add_missing_columns or self.config.zero_as_missing
                ) and any(
                    self.config.module_path.endswith(f)
                    for f in ["parquet.py", "arrow.py"]
                ):
                    self.info.features = Features()
                    if (
                        len(self.config.data_files[data_split.name]) > 1
                        and self.config.use_first_schema is None
                    ):
                        file_type = self.config.module_path.split("/")[-1].split(".")[0]
                        logger.warning(
                            f"Multiple {file_type} files were provided. The schema "
                            "will use the combined schema of all data files. To use "
                            "the schema of only the first file, set "
                            "`use_first_schema=True`. To turn off this warning, set "
                            "`use_first_schema=False`."
                        )
                    num_files = len(self.config.data_files[data_split.name])
                    for idx, file in tqdm(
                        enumerate(
                            itertools.chain.from_iterable(
                                [self.config.data_files[data_split.name]]
                            )
                        ),
                        total=num_files,
                        desc=f"Loading schema for files in {data_split.name}",
                    ):
                        with open(file, "rb") as f:
                            if self.config.module_path.endswith("parquet.py"):
                                self.info.features.update(
                                    datasets.Features.from_arrow_schema(
                                        pq.read_schema(f)
                                    )
                                )
                            elif self.config.module_path.endswith("arrow.py"):
                                self.info.features.update(
                                    datasets.Features.from_arrow_schema(
                                        pa.ipc.read_schema(f)
                                    )
                                )
                        if self.config.use_first_schema:
                            break
                    generator.info.features = self.info.features
            else:
                self.info.features = self.config.features
            splits.append(
                datasets.SplitGenerator(
                    name=data_split.name,
                    gen_kwargs={
                        "generator": generator,
                        **data_split.gen_kwargs,
                        "split_name": data_split.name,
                    },
                )
            )

        return splits

    def _set_labels(self, tbl, labels=None):
        def fn(tbl, labels, all_labels):
            lab2int = {label: i for i, label in enumerate(labels)}
            all_labels = [lab2int.get(label, -1) for label in all_labels]
            tbl_format = DataHandler.get_format(tbl)
            all_labels = DataHandler.to_format(all_labels, tbl_format)
            tbl = DataHandler.append_column(
                tbl,
                self.TARGET_COLUMN,
                all_labels,
            )

            return tbl

        if self.config.target_column in DataHandler.get_column_names(tbl):
            # provide either positive_labels and/or negative_labels, or labels.
            if self.config.positive_labels or self.config.negative_labels:
                self.config.labels = self.config.labels or []

                if len(self.config.labels) == 0:
                    if self.config.negative_label_name:
                        self.config.labels.append(self.config.negative_label_name)
                    else:
                        self.config.labels.append("negative")

                    if self.config.positive_label_name:
                        self.config.labels.append(self.config.positive_label_name)
                    else:
                        self.config.labels.append("positive")

                if not self._label_map:
                    self._label_map = {}
                    if self.config.positive_labels:
                        self._label_map.update(
                            {label: 1 for label in self.config.positive_labels}
                        )
                    if self.config.negative_labels:
                        self._label_map.update(
                            {label: 0 for label in self.config.negative_labels}
                        )
                bin_labels = [
                    self._label_map.get(label, -1)
                    for label in DataHandler.to_list(
                        DataHandler.select_column(tbl, self.config.target_column)
                    )
                ]
                tbl = DataHandler.append_column(
                    tbl,
                    self.TARGET_COLUMN,
                    bin_labels,
                )

            # create labels only if a single sample metadata and/or data_file
            # is provide, or else throw
            elif not is_regression_type(tbl[self.config.target_column]):
                current_labels = DataHandler.to_list(
                    DataHandler.select_column(tbl, self.config.target_column)
                )
                if labels is None:
                    labels = list(set(current_labels))
                if None in labels:
                    labels.remove(None)
                tbl = fn(tbl, labels, current_labels)
                if self.config.labels is None:
                    self.config.labels = [str(label) for label in labels]

        return tbl

    def _add_sample_metadata(self, table, sample_metadata=None):
        if self.config.sample_metadata_files:
            if self.config.sample_column in table.column_names:
                colliding_names = list(
                    (set(table.column_names) & set(sample_metadata.columns))
                    - set([self.config.sample_column])
                )
                if isinstance(sample_metadata, pd.DataFrame):
                    pd_table: pd.DataFrame = table.drop(colliding_names).to_pandas()
                    tbl_cols = [
                        c for c in pd_table.columns if c != self.config.sample_column
                    ]
                    col_order = list(sample_metadata.columns) + tbl_cols
                    table = pa.Table.from_pandas(
                        pd_table.merge(
                            sample_metadata,
                            how="left",
                            on=self.config.sample_column,
                        ).reindex(columns=col_order),
                        preserve_index=False,
                    )
                else:
                    import polars as pl

                    pl_table = pl.from_arrow(table.drop(colliding_names))
                    tbl_cols = [
                        c for c in pl_table.columns if c != self.config.sample_column
                    ]
                    col_order = sample_metadata.columns + tbl_cols
                    table = (
                        pl_table.join(
                            sample_metadata,
                            on=self.config.sample_column,
                            how="left",
                        )
                        .select(col_order)
                        .to_arrow()
                    )
            else:
                # we are gonna assume that the row order in metadata is the same as the data
                colliding_names = list(
                    (set(table.column_names) & set(sample_metadata.columns))
                )
                if isinstance(sample_metadata, pd.DataFrame):
                    # concat the table without join, making sure that colliding columns are dropped
                    pd_table: pd.DataFrame = table.drop(colliding_names).to_pandas()
                    new_cols = (
                        pd_table.columns.tolist() + sample_metadata.columns.tolist()
                    )
                    table = pa.Table.from_pandas(
                        pd.concat(
                            [pd_table, sample_metadata],
                            axis=1,
                            ignore_index=True,
                        ),
                        preserve_index=False,
                    )
                    table = DataHandler.set_column_names(table, new_cols)
                else:
                    import polars as pl

                    pl_table = pl.from_arrow(table.drop(colliding_names))
                    table = pl.concat(
                        [sample_metadata, pl_table], how="horizontal"
                    ).to_arrow()
        return table

    def _prepare_labels(self, table, sample_metadata=None, split_name=None):
        if self.config.target_column:
            labels = self.config.labels
            if (
                not self.config.positive_labels
                and not self.config.negative_labels
                and not self.config.labels
            ):
                if (
                    self.config.sample_metadata_files
                    and len(self.config.sample_metadata_files.get(split_name, [])) == 1
                    and self.config.target_column
                    in DataHandler.get_column_names(sample_metadata)
                ):
                    all_labels = DataHandler.to_list(
                        DataHandler.select_column(
                            sample_metadata, self.config.target_column
                        )
                    )
                    labels = list(set(all_labels))
                elif len(
                    self.config.data_files[split_name]
                ) == 1 and self.config.target_column in DataHandler.get_column_names(
                    table
                ):
                    all_labels = DataHandler.to_list(
                        DataHandler.select_column(table, self.config.target_column)
                    )
                    labels = list(set(all_labels))
                else:
                    if (
                        sample_metadata is not None
                        and self.config.target_column
                        in DataHandler.get_column_names(sample_metadata)
                    ):
                        raise ValueError(
                            "Labels must be provided if multiple sample metadata files "
                            "are provided. Either set `labels`, `positive_labels` "
                            "and/or `negative_labels` in `load_dataset`."
                        )
                    else:
                        raise ValueError(
                            "Labels must be provided if multiple data files "
                            "are provided and the target column is found in the "
                            "data table. Either set `labels`, `positive_labels` "
                            "and/or `negative_labels` in `load_dataset`."
                        )

            table = self._set_labels(table, labels=labels)
        return table

    def _generate_tables(self, generator: "ArrowBasedBuilder", *args, **gen_kwargs):
        """Generate tables from a list of generators."""

        split_name = gen_kwargs.pop("split_name")
        feature_metadata = None
        if self.config.feature_metadata_files:
            feature_metadata = self._read_metadata(
                self.config.feature_metadata_files[split_name]
            )

        check_columns = True
        feature_metadata_dict = None
        # key might not correspond to the current index of the file received (e.g. npz)
        file_index = 0
        sample_metadata = None
        for key, table in generator._generate_tables(*args, **gen_kwargs):
            stored_metadata_schema = table.schema.metadata or {}

            sample_metadata = self._load_metadata(
                file_index, sample_metadata, split_name=split_name
            )
            if check_columns:
                features = table.column_names
                self = self._set_columns(
                    features,
                    sample_metadata=sample_metadata,
                    feature_metadata=feature_metadata,
                )

                if feature_metadata and self.config.feature_column:
                    feature_metadata_dict = self._convert_feature_metadata_to_dict(
                        feature_metadata
                    )
                check_columns = False

            if self.config.data_columns is not None:
                table = DataHandler.set_column_names(table, self.config.data_columns)
            elif (
                feature_metadata is not None
                and self.config.config_path == "npz"
                and self.config.feature_column is not None
            ):
                feature_metadata_dims = DataHandler.get_shape(feature_metadata)
                table_dims = DataHandler.get_shape(table)
                if feature_metadata_dims[0] == table_dims[1]:
                    logger.warning_once(
                        "No data columns were provided, but the number of columns "
                        "in the data table matches the number of features in the "
                        "feature metadata. Using feature metadata as column names."
                    )

                    features = DataHandler.to_list(
                        DataHandler.select_column(
                            feature_metadata, self.config.feature_column
                        )
                    )
                    table = DataHandler.set_column_names(table, features)
                else:
                    logger.warning_once(
                        "Feature metadata was provided along with an npz file, but the "
                        "number of features in the metadata does not match the number "
                        "of columns in the data table. Ignoring feature metadata."
                    )

            table = self._add_sample_metadata(table, sample_metadata)
            table = self._prepare_labels(table, sample_metadata, split_name=split_name)
            # table = self._prepare_data(table)
            metadata_schema = None
            if b"huggingface" in stored_metadata_schema:
                metadata_schema = json.loads(
                    stored_metadata_schema[b"huggingface"].decode()
                )
                if (
                    "info" not in metadata_schema
                    or "features" not in metadata_schema["info"]
                ):
                    metadata_schema = Features.from_arrow_schema(table.schema)
                else:
                    metadata_schema = Features.from_dict(
                        metadata_schema["info"]["features"]
                    )
            else:
                metadata_schema = Features.from_arrow_schema(table.schema)

            if not self.info.features:
                if self.config.features is not None:
                    self.info.features = self.config.features
                else:
                    self.info.features = self._create_features(
                        metadata_schema,
                        column_names=DataHandler.get_column_names(table),
                        feature_metadata=feature_metadata_dict,
                    )
            if self.info.features:
                missing_columns = set(self.info.features) - set(table.column_names)
                if missing_columns:
                    if self.config.add_missing_columns:
                        num_rows = table.num_rows
                        fill_value = 0 if self.config.zero_as_missing else None
                        new_columns = {
                            name: pa.array([fill_value] * num_rows)
                            for name in missing_columns
                        }
                        combined_columns = {**table.to_pydict(), **new_columns}
                        table = pa.table(combined_columns)

            metadata_dump = {}
            for k, v in stored_metadata_schema.items():
                try:
                    value = json.loads(v.decode())
                except json.JSONDecodeError:
                    value = v.decode()
                metadata_dump[k.decode()] = value
            metadata_dump["huggingface"] = metadata_dump.get("huggingface", {})
            metadata_dump["huggingface"]["info"] = metadata_dump["huggingface"].get(
                "info", {}
            )
            metadata_dump["huggingface"]["info"]["features"] = asdict(
                self.info.features
            )

            table = table.replace_schema_metadata(
                metadata={k: json.dumps(v) for k, v in metadata_dump.items()}
            )

            file_index += 1
            yield key, table

    def _load_metadata(self, file_index, sample_metadata=None, split_name=None):
        if self.config.sample_metadata_files:
            if not isinstance(self.config.sample_metadata_files, DataFilesDict):
                raise ValueError(
                    "Sample metadata files must be a dictionary with split names as keys."
                )
            if split_name not in self.config.sample_metadata_files:
                return None

            files = self.config.sample_metadata_files[split_name]
            if len(files) == 1:
                if file_index != 0:
                    return sample_metadata
            # only use file_index if data files are sharded
            elif len(self.config.data_files[split_name]) > 1:
                files = [files[file_index]]

            sample_metadata = self._read_metadata(files, to_arrow=False)
            # TODO: temporary fix for not getting a pandas DataFrame
            if isinstance(sample_metadata, pa.Table):
                sample_metadata = DataHandler.to_pandas(sample_metadata)
            return sample_metadata

    def _read_metadata(self, metadata_files, use_polars: bool = True, to_arrow=True):
        if not metadata_files:
            raise ValueError("Empty list of metadata files provided.")

        metadata_ext = os.path.splitext(metadata_files[0])[-1][1:]

        if "json" in metadata_ext:
            metadata_ext = "json"

        dataset = next(
            iter(
                datasets.load_dataset(
                    metadata_ext,
                    data_files=metadata_files,
                    cache_dir=self._cache_dir_root,
                ).values()
            )
        )
        if to_arrow:
            return dataset._data.table
        return dataset.to_pandas()

    def _create_features(
        self,
        schema: Union[Features, Dict[str, Any], pa.Schema, pa.Table],
        column_names,
        feature_metadata=None,
    ):
        if schema is None or len(schema) == 0:
            raise ValueError("The schema is not available.")
        if not isinstance(column_names, set):
            column_names = set(column_names)
        _schema: Features = None
        if isinstance(schema, dict):
            entry = next(iter(schema.values()))
            if isinstance(entry, dict):
                if len(entry) > 0:
                    _schema = Features.from_dict(schema)
                else:
                    return Features({})
            elif not hasattr(entry, "pa_type"):
                raise ValueError(
                    "Could not infer the schema of the dataset. Please provide the "
                    "schema in the `features` argument."
                )
            else:
                _schema = schema
        elif isinstance(schema, pa.Schema):
            _schema = Features.from_arrow_schema(schema)
        elif isinstance(schema, pa.Table):
            _schema = Features.from_arrow_schema(schema.schema)

        def _get_schema(
            _schema: Features,
            new_schema={},
        ):
            for k, v in _schema.items():
                if self.info.features:
                    v_ = self.info.features.get(k, None)
                    if v_:
                        new_schema[k] = v_
                        continue
                elif v.dtype == "null":
                    v = Value("string")

                if feature_metadata and k in feature_metadata:
                    new_schema[k] = self.INPUT_FEATURE(
                        dtype=DTYPE_MAP.get(v.dtype, v.dtype),
                        metadata=feature_metadata[k],
                        id=v.id,
                    )
                elif k == self.config.sample_column or k == self.SAMPLE_COLUMN:
                    self.config.sample_column = (
                        self.config.sample_column or self.SAMPLE_COLUMN
                    )
                    new_schema[self.config.sample_column] = Sample(
                        dtype=DTYPE_MAP.get(v.dtype, v.dtype)
                    )
                elif k == self.config.batch_column or k == self.BATCH_COLUMN:
                    self.config.batch_column = (
                        self.config.batch_column or self.BATCH_COLUMN
                    )
                    new_schema[self.config.batch_column] = Batch(
                        dtype=DTYPE_MAP.get(v.dtype, v.dtype)
                    )
                elif k == self.TARGET_COLUMN:
                    self.config.target_column = (
                        self.config.target_column or self.TARGET_COLUMN
                    )
                    if isinstance(v, (RegressionTarget, ClassLabel)):
                        if isinstance(v, ClassLabel):
                            if (
                                self.config.positive_labels
                                or self.config.negative_labels
                            ):
                                new_schema[self.TARGET_COLUMN] = BinClassLabel(
                                    positive_labels=self.config.positive_labels,
                                    negative_labels=self.config.negative_labels,
                                    names=self.config.labels,
                                    id=self.config.target_column,
                                )

                            else:
                                self.config.labels = v.names
                                v.id = self.config.target_column
                                new_schema[self.TARGET_COLUMN] = v

                    elif is_regression_type(v.pa_type):
                        new_schema[self.TARGET_COLUMN] = RegressionTarget(
                            v.dtype, id=self.config.target_column
                        )
                    elif self.config.labels and is_classification_type(v.pa_type):
                        if self.config.positive_labels or self.config.negative_labels:
                            new_schema[self.TARGET_COLUMN] = BinClassLabel(
                                positive_labels=self.config.positive_labels,
                                negative_labels=self.config.negative_labels,
                                names=self.config.labels,
                                id=self.config.target_column,
                            )
                        else:
                            new_schema[self.TARGET_COLUMN] = ClassLabel(
                                num_classes=len(self.config.labels),
                                names=self.config.labels,
                                id=self.config.target_column,
                            )
                    else:
                        raise ValueError(
                            "The dataset seems to be a classification task, but the labels are not provided.\n"
                            "Please provide the labels in the `labels` argument. For example:\n"
                            "   >>> dataset = dataset.load_dataset('csv', data_files='data.csv', labels=[0, 1, 2])\n"
                            "Or provide a sample metadata table with the labels column. For example:\n"
                            "   >>> # metadata.csv contains a column named 'disease state' with labels 0, 1, 2\n"
                            "   >>> dataset.load_dataset('csv', data_files='data.csv', sample_metadata_files='metadata.csv', target_column='disease state')\n"
                        )
                elif k == self.METADATA_COLUMN and isinstance(v, dict):
                    new_schema[self.METADATA_COLUMN] = Metadata(feature=v)
                elif self.config.metadata_columns and k in self.config.metadata_columns:
                    new_schema[k] = Metadata(
                        dtype=DTYPE_MAP.get(v.dtype, v.dtype), id=k
                    )

                else:
                    if k in column_names:
                        new_schema[k] = self.INPUT_FEATURE(
                            dtype=DTYPE_MAP.get(v.dtype, v.dtype), metadata={}, id=v.id
                        )
                    else:
                        raise ValueError(
                            f"Could not find the column '{k}' in the data table."
                        )

            return new_schema

        new_schema = _get_schema(
            _schema,
        )

        return Features(new_schema)


def _infer_column_name(
    column_name: str,
    default_column_name: str,
    features: List[str],
    metadata_columns: List[str],
    patterns: List[str],
    required: Optional[bool] = False,
    is_index=False,
):
    """_summary_

    Args:
        column_name (_type_): column name within the config.
        features (List[str]): data columns
        metadata_columns (List[str]): metadata columns
        patterns (List[str]): patterns to search for in the columns
        required (Optional[bool], optional): _description_. Defaults to False.

    Raises:
        _COLUMN_TO_ERROR: _description_
        _COLUMN_TO_ERROR: _description_

    Returns:
        _type_: _description_
    """
    logger.debug(
        f"Starting _infer_column_name with column_name: {column_name}, default_column_name: {default_column_name}, is_index: {is_index}"
    )

    # no table was given, so we can't infer the column name
    if not features and not metadata_columns:
        logger.debug(
            "No features or metadata_columns provided. Returning original column_name."
        )
        return column_name

    # check if the column name is in the features or metadata columns. Default is True if the list is empty
    name_in_features = column_name in features if features else is_index
    name_in_metadata = column_name in metadata_columns if metadata_columns else is_index

    logger.debug(
        f"name_in_features: {name_in_features}, name_in_metadata: {name_in_metadata}"
    )

    # check if sample column is in one of the tables
    if is_index:
        if name_in_features and name_in_metadata:
            logger.debug(
                f"Column {column_name} found in both features and metadata as index."
            )
            return column_name
    elif name_in_features or name_in_metadata:
        logger.debug(f"Column {column_name} found in features or metadata.")
        return column_name

    default_in_features = default_column_name in features if features else is_index
    default_in_metadata = (
        default_column_name in metadata_columns if metadata_columns else is_index
    )

    logger.debug(
        f"default_in_features: {default_in_features}, default_in_metadata: {default_in_metadata}"
    )

    if is_index:
        if default_in_features and default_in_metadata:
            logger.debug(
                f"Default column {default_column_name} found in both features and metadata as index."
            )
            return default_column_name
    elif default_in_features or default_in_metadata:
        logger.debug(
            f"Default column {default_column_name} found in features or metadata."
        )
        return default_column_name

    def generate_err_msg():
        max_char_len = 200
        err_msg = ""
        features_str = None
        metadata_columns_str = None

        if features:
            features_str = str(features)
            if len(features_str) > max_char_len:
                i = max_char_len // 2
                features_str = features_str[:i] + "..." + features_str[-i:]
        if metadata_columns:
            metadata_columns_str = str(metadata_columns)
            if len(metadata_columns_str) > max_char_len:
                i = max_char_len // 2
                metadata_columns_str = (
                    metadata_columns_str[:i] + "..." + metadata_columns_str[-i:]
                )

        if not features_str:
            err_msg += (
                "metadata table. Available columns in metadata table: "
                "{metadata_columns_str}"
            )
        elif not metadata_columns_str:
            err_msg += f"data table. Available columns in data table: {features_str}"
        else:
            err_msg += (
                "data or metadata table\n"
                f"Available columns in data: {features_str}\n"
                f"Available columns in metadata table: {metadata_columns_str}"
            )
        return err_msg

    if column_name:
        if name_in_features and not name_in_metadata:
            logger.warning_once(
                f"'{column_name}' was found in the data table but not in the metadata table.\n"
                f"Please add or rename the column in the metadata table.\n"
            )
        elif not name_in_features and name_in_metadata:
            logger.warning_once(
                f"'{column_name}' was found in the metadata table but not in the data table.\n"
                f"Please add or rename the column in the data table.\n"
            )
        else:
            err_msg = (
                f"Could not find the {column_name} column in " + generate_err_msg()
            )
            logger.debug(f"Raising error for missing column: {column_name}")
            raise _COLUMN_TO_ERROR[column_name](err_msg)
        return None

    # try searching for the column name from patterns
    other_cols = None
    cols = None
    if metadata_columns:
        cols = metadata_columns
        if is_index:  # must be in both metadata and features
            other_cols = set([feat.lower() for feat in features])

    if features and (cols is None or not is_index):
        if cols:
            cols = cols + features
        else:
            cols = features
        other_cols = None

    # if the column is an index, both the features and metadata columns must contain the column
    if metadata_columns and features and is_index:
        sample_column = list(set(metadata_columns) & set(features))
        # if there is only one matching column, use it
        if len(sample_column) == 1:
            logger.debug(f"Single matching column found for index: {sample_column[0]}")
            return sample_column[0]

    possible_col = None
    which_table = None
    for dcol in patterns:
        # we prioritize the left-most match
        for col in cols:
            if "*" in dcol:
                if dcol[:-1].lower() == col.lower():
                    possible_col = col
                    logger.debug(f"Pattern match found with wildcard: {dcol} -> {col}")
                    if not other_cols or col.lower() in other_cols:
                        return col
            elif dcol.lower() in col.lower():
                possible_col = col
                which_table = (
                    "metadata"
                    if metadata_columns is not None and col in metadata_columns
                    else "data"
                )
                logger.debug(f"Pattern match found: {dcol} -> {col}")
                if not other_cols or col.lower() in other_cols:
                    return col
    if possible_col and not required:
        other_possible_col = None
        other_table = "data" if which_table == "metadata" else "metadata"
        for dcol in patterns:
            # we prioritize the left-most match
            for col in other_cols:
                if "*" in dcol:
                    if dcol[:-1].lower() == col.lower():
                        possible_col = col
                        logger.debug(
                            f"Pattern match found with wildcard: {dcol} -> {col}"
                        )
                        if not other_cols or col.lower() in other_cols:
                            return col
                elif dcol.lower() in col.lower():
                    other_possible_col = col
                    logger.debug(f"Pattern match found: {dcol} -> {col}")
        if other_possible_col:
            logger.warning_once(
                f"Two possible matches found for the {default_column_name} column:\n"
                f"1. '{possible_col}' in {which_table} table\n"
                f"2. '{other_possible_col}' in {other_table} table\n"
                "Please rename the columns or provide the `sample_column` argument to "
                "avoid ambiguity.\n"
                f"Using the {default_column_name} column detected from the "
                f"{which_table} table."
            )
        else:
            logger.warning_once(
                f"A match for the {default_column_name} column was found in "
                f"{which_table} table: '{possible_col}'\n"
                f"But it was not found in {other_table} table.\n"
                "Please add or rename the column in the appropriate table.\n"
                f"Using the {default_column_name} column detected from the "
                f"{which_table} table."
            )
        return possible_col
    elif required:
        err_msg = (
            f"Could not find the {default_column_name} column in " + generate_err_msg()
        )
        logger.debug(
            f"Raising error for missing required column: {default_column_name}"
        )
        # throw an error as the metadata_table was given and we couldn't find a sample column
        raise _COLUMN_TO_ERROR[default_column_name](err_msg)

    logger.warning_once(
        f"Could not find the {default_column_name} column in " + generate_err_msg()
    )
    return None


SUPPORTED_EXTENSIONS = [
    ".csv",
    ".parquet",
    ".arrow",
    ".txt",
    ".tar",
    ".tsv",
    ".zip",
    ".npz",
]

BioData.EXTENSIONS = SUPPORTED_EXTENSIONS
