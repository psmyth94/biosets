import copy
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import datasets
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
import pyarrow.json as paj
from datasets.data_files import DataFilesDict, DataFilesPatternsList, sanitize_patterns
from datasets.features.features import ClassLabel, Features, Value
from datasets.naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from datasets.packaged_modules.arrow import arrow as hf_arrow
from datasets.packaged_modules.csv import csv as hf_csv
from datasets.packaged_modules.json import json as hf_json
from datasets.packaged_modules.parquet import parquet as hf_parquet
from datasets.utils.py_utils import asdict

from biosets.integration.datasets.datasets import DatasetsPatcher
from biosets.packaged_modules.npz.npz import SparseReaderConfig
from biosets.utils import (
    as_py,
    concat_blocks,
    is_file_name,
    is_polars_available,
    logging,
    upcast_tables,
)

from ...data_files import (
    FEATURE_METADATA_FILENAMES,
    SAMPLE_METADATA_FILENAMES,
    get_feature_metadata_patterns,
    get_metadata_patterns,
)
from ...features import (
    Batch,
    BinClassLabel,
    Metadata,
    RegressionTarget,
    Sample,
    ValueWithMetadata,
)
from ..csv import csv as biosets_csv

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

    metadata_dir: Optional[str] = None
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
    target_column: Optional[str] = None
    feature_column: Optional[str] = None

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
    DEFAULT_SAMPLE_METADATA_FILENAMES = SAMPLE_METADATA_FILENAMES
    DEFAULT_FEATURE_METADATA_FILENAMES = FEATURE_METADATA_FILENAMES

    def __post_init__(self):
        # The config name is used to name the cache directory.
        for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
            if invalid_char in self.name:
                raise datasets.builder.InvalidConfigName(
                    f"Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{self.name}'. "
                    f"They could create issues when creating a directory for this config on Windows filesystem."
                )

        if self.data_files and not isinstance(self.data_files, DataFilesDict):
            raise ValueError(
                f"Expected a DataFilesDict in data_files but got {self.data_files}"
            )
        elif self.data_files:
            data_files_dict = copy.deepcopy(self.data_files)
            for split, files in self.data_files.items():
                for i in range(len(files) - 1, -1, -1):
                    file = files[i]
                    if os.path.basename(file) in self.DEFAULT_SAMPLE_METADATA_FILENAMES:
                        self.sample_metadata_files = self.sample_metadata_files or []
                        if isinstance(self.sample_metadata_files, str):
                            self.sample_metadata_files = [self.sample_metadata_files]
                        self.sample_metadata_files.append(
                            Path(file).resolve().as_posix()
                        )
                        data_files_dict[split].pop(i)
                        data_files_dict[split].origin_metadata.pop(i)
                    elif (
                        os.path.basename(file)
                        in self.DEFAULT_FEATURE_METADATA_FILENAMES
                    ):
                        self.feature_metadata_files = self.feature_metadata_files or []
                        self.feature_metadata_files.append(
                            Path(file).resolve().as_posix()
                        )
                        data_files_dict[split].pop(i)
                        data_files_dict[split].origin_metadata.pop(i)
                    elif os.path.splitext(file)[-1].lower() not in self.EXTENSION_MAP:
                        data_files_dict[split].pop(i)
                        data_files_dict[split].origin_metadata.pop(i)
            self.data_files = data_files_dict

        if self.metadata_dir and not self.sample_metadata_files:
            base_path = Path(self.metadata_dir or "").expanduser().resolve().as_posix()
            metadata_patterns = get_metadata_patterns(base_path)
            self.sample_metadata_files = (
                DataFilesPatternsList.from_patterns(
                    metadata_patterns,
                    allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
                )
                if metadata_patterns
                else None
            )
            self.sample_metadata_files = (
                [Path(f) for f in self.sample_metadata_files]
                if self.sample_metadata_files
                else None
            )

            self.sample_metadata_files = [
                f.resolve().as_posix() for f in self.sample_metadata_files if f.exists()
            ]

        if self.sample_metadata_files and not isinstance(
            self.sample_metadata_files, DataFilesPatternsList
        ):
            if isinstance(self.sample_metadata_files, (str, Path)):
                self.sample_metadata_files = [self.sample_metadata_files]
            data_dir = ""
            if len(self.sample_metadata_files) > 1:
                self.sample_metadta_files = [
                    Path(f).resolve().as_posix() for f in self.sample_metadata_files
                ]
                data_dir = os.path.commonpath(self.sample_metadata_files)

            if not data_dir:
                if is_file_name(self.sample_metadata_files[0]):
                    paths = [f for split in self.data_files.values() for f in split]
                    if len(paths) > 1:
                        data_dir = os.path.commonpath(paths)
                    else:
                        data_dir = Path(paths[0]).parent.resolve().as_posix()
                else:
                    data_dir = (
                        Path(self.sample_metadata_files[0]).parent.resolve().as_posix()
                    )
                    self.sample_metadata_files = [
                        Path(f).name for f in self.sample_metadata_files
                    ]

            metadata_patterns = next(
                iter(sanitize_patterns(self.sample_metadata_files).values())
            )
            self.sample_metadata_files = DataFilesPatternsList.from_patterns(
                metadata_patterns,
                allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
            ).resolve(data_dir)

        if self.feature_metadata_dir and not self.feature_metadata_files:
            base_path = (
                Path(self.feature_metadata_dir or "").expanduser().resolve().as_posix()
            )
            feature_metadata_patterns = get_feature_metadata_patterns(base_path)
            self.feature_metadata_files = (
                DataFilesPatternsList.from_patterns(
                    feature_metadata_patterns,
                    allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
                )
                if feature_metadata_patterns
                else None
            )
            self.feature_metadata_files = (
                [Path(f).resolve().as_posix() for f in self.feature_metadata_files]
                if self.feature_metadata_files
                else None
            )

        if self.feature_metadata_files and not isinstance(
            self.feature_metadata_files, DataFilesPatternsList
        ):
            if isinstance(self.feature_metadata_files, (str, Path)):
                self.feature_metadata_files = [self.feature_metadata_files]
            data_dir = ""
            if len(self.feature_metadata_files) > 1:
                data_dir = os.path.commonpath(self.feature_metadata_files)

            if not data_dir:
                if is_file_name(self.feature_metadata_files[0]):
                    paths = [f for split in self.data_files.values() for f in split]
                    if len(paths) > 1:
                        data_dir = os.path.commonpath(paths)
                    else:
                        data_dir = Path(paths[0]).parent.resolve().as_posix()
                else:
                    data_dir = (
                        Path(self.feature_metadata_files[0]).parent.resolve().as_posix()
                    )
                    self.feature_metadata_files = [
                        Path(f).name for f in self.feature_metadata_files
                    ]

            feature_metadata_patterns = next(
                iter(sanitize_patterns(self.feature_metadata_files).values())
            )
            self.feature_metadata_files = DataFilesPatternsList.from_patterns(
                feature_metadata_patterns,
                allowed_extensions=self.ALLOWED_METADATA_EXTENSIONS,
            ).resolve(data_dir)

        for split in list(self.data_files.keys()):
            for i in reversed(range(len(self.data_files[split]))):
                if (
                    self.sample_metadata_files
                    and self.data_files[split][i] in self.sample_metadata_files
                ):
                    self.data_files[split].pop(i)
                    self.data_files[split].origin_metadata.pop(i)
                elif (
                    self.feature_metadata_files
                    and self.data_files[split][i] in self.feature_metadata_files
                ):
                    self.data_files[split].pop(i)
                    self.data_files[split].origin_metadata.pop(i)

        self.data_kwargs = self._get_builder_kwargs(self.data_files)

    def _get_builder_kwargs(self, files):
        if files is None:
            return {}
        builder_kwargs = None
        iter_files = files

        if isinstance(files, (str, list, tuple)):
            iter_files = [files]
        if isinstance(iter_files, (dict, DataFilesDict)):
            iter_files = iter_files.values()
        for file in itertools.chain.from_iterable(iter_files):
            file_ext = os.path.splitext(file)[-1].lower()
            if file_ext in self.EXTENSION_MAP:
                config_path, _config_class, hf_config_class = self.EXTENSION_MAP[
                    file_ext
                ]
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
                builder_kwargs["path"] = config_path
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

    def __init__(self, *args, **kwargs):
        with DatasetsPatcher():
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
        _feature_metadata = {}
        features = feature_metadata.column(self.config.feature_column).to_pylist()
        metadata = feature_metadata.drop([self.config.feature_column]).to_pylist()
        _feature_metadata = {str(k): as_py(v) for k, v in zip(features, metadata)}
        return _feature_metadata

    def _check_columns(self, data_features: List[str]):
        _all_columns = set(data_features)
        if self.config.sample_column:
            _all_columns.add(self.config.sample_column)
        if self.config.batch_column:
            _all_columns.add(self.config.batch_column)
        if self.config.target_column:
            _all_columns.add(self.config.target_column)
        if self.config.metadata_columns:
            _all_columns.update(set(self.config.metadata_columns))

        if not self.info.features:
            self.info.features = self._create_features(data_features)

        column_names = set(self.info.features) - set(_all_columns)
        if column_names:
            logger.warning_once(
                "\nFeatures were provided but the following columns were not found in the data table or metadata table:\n"
                f"{list(column_names)}\n"
                "These columns will be ignored.\n"
            )

        return self

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

            splits.append(
                datasets.SplitGenerator(
                    name=data_split.name,
                    gen_kwargs={"generator": generator, **data_split.gen_kwargs},
                )
            )

        return splits

    def _generate_tables(self, generator: "ArrowBasedBuilder", *args, **gen_kwargs):
        """Generate tables from a list of generators."""
        from biosets.data_handling import DataHandler

        def set_labels(tbl):
            def fn(tbl, labels, all_labels):
                if any(isinstance(label, str) for label in labels):
                    lab1int = {label: i for i, label in enumerate(labels)}
                    all_labels = [lab1int.get(label, -1) for label in all_labels]
                    tbl_format = DataHandler.get_format(tbl)
                    all_labels = DataHandler.to_format(all_labels, tbl_format)
                    tbl = DataHandler.append_column(
                        tbl,
                        self.TARGET_COLUMN,
                        all_labels,
                    )

                return tbl

            if (
                not self.config.labels
                and self.config.target_column in DataHandler.get_column_names(tbl)
            ):
                if self.config.positive_labels or self.config.negative_labels:
                    self.config.labels = []

                    if self.config.negative_label_name:
                        self.config.labels.append(self.config.negative_label_name)
                    else:
                        self.config.labels.append("negative")

                    if self.config.positive_label_name:
                        self.config.labels.append(self.config.positive_label_name)
                    else:
                        self.config.labels.append("positive")

                    label_map = {}
                    if self.config.positive_labels:
                        label_map.update(
                            {label: 1 for label in self.config.positive_labels}
                        )
                    if self.config.negative_labels:
                        label_map.update(
                            {label: 0 for label in self.config.negative_labels}
                        )
                    bin_labels = [
                        label_map.get(label, -1)
                        for label in DataHandler.to_list(
                            DataHandler.select_column(tbl, self.config.target_column)
                        )
                    ]
                    tbl = DataHandler.append_column(
                        tbl,
                        self.TARGET_COLUMN,
                        bin_labels,
                    )

                elif not is_regression_type(tbl[self.config.target_column]):
                    all_labels = DataHandler.to_list(
                        DataHandler.select_column(tbl, self.config.target_column)
                    )
                    labels = list(set(all_labels))
                    if None in labels:
                        labels.remove(None)
                    tbl = fn(tbl, labels, all_labels)
                    self.config.labels = [str(label) for label in labels]

            return tbl

        sample_metadata = None
        if self.config.sample_metadata_files:
            sample_metadata: Union[pd.DataFrame, "pl.DataFrame"] = self._read_metadata(
                self.config.sample_metadata_files, to_arrow=False
            )

            # TODO: temporary fix for not getting a pandas DataFrame
            if isinstance(sample_metadata, pa.Table):
                from biosets.data_handling import DataHandler

                data_handler = DataHandler()
                sample_metadata = data_handler.to_pandas(sample_metadata)

        feature_metadata = None
        if self.config.feature_metadata_files:
            feature_metadata: pa.Table = self._read_metadata(
                self.config.feature_metadata_files
            )

        check_columns = True
        feature_metadata_dict = None
        for key, table in generator._generate_tables(*args, **gen_kwargs):
            stored_metadata_schema = table.schema.metadata or {}

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

            if sample_metadata is not None:
                if self.config.sample_column in table.column_names:
                    colliding_names = list(
                        (set(table.column_names) & set(sample_metadata.columns))
                        - set([self.config.sample_column])
                    )
                    if isinstance(sample_metadata, pd.DataFrame):
                        pd_table: pd.DataFrame = table.drop(colliding_names).to_pandas()
                        tbl_cols = [
                            c
                            for c in pd_table.columns
                            if c != self.config.sample_column
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
                            c
                            for c in pl_table.columns
                            if c != self.config.sample_column
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
                                [pd_table, sample_metadata], axis=1, ignore_index=True
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

            if self.config.target_column and not self.config.labels:
                table = set_labels(table)
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
                self.info.features = self._create_features(
                    metadata_schema,
                    feature_metadata=feature_metadata_dict,
                )

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

            yield key, table

    def _read_metadata(self, metadata_files, use_polars: bool = True, to_arrow=True):
        def setup_readers(format="csv"):
            if use_polars and is_polars_available():
                import polars as pl

                csv = pl.read_csv
                parquet = pl.read_parquet

                def arrow_converter(x: pl.DataFrame):
                    # remove null columns
                    x = x.filter(~pl.all_horizontal(pl.all().is_null()))
                    return x.to_arrow() if isinstance(x, pl.DataFrame) else x

                def concat_tables(tables):
                    return pl.concat(tables, axis=0)

                tab_sep = {"separator": "\t"}
            else:
                csv = pd.read_csv
                parquet = pd.read_parquet

                def arrow_converter(x: pd.DataFrame):
                    # remove null columns
                    x = x.dropna(axis=1, how="all")
                    return (
                        pa.Table.from_pandas(x, preserve_index=False)
                        if isinstance(x, pd.DataFrame)
                        else x
                    )

                def concat_tables(tables):
                    return pd.concat(tables, axis=0, copy=False)

                tab_sep = {"sep": "\t"}

            def arrow(x):
                return pa.ipc.open_stream(x).read_all()

            def json_file(x):
                with open(x, "rb") as f:
                    return paj.read_json(f)

            return {
                ".csv": (csv, {}, arrow_converter),
                ".tsv": (csv, tab_sep, arrow_converter),
                ".txt": (csv, tab_sep, arrow_converter),
                ".json": (json_file, {}, lambda x: x),
                ".parquet": (parquet, {}, lambda x: x),
                ".arrow": (arrow, {}, lambda x: x),
            }

        try:
            readers = setup_readers()
            metadata = []
            for metadata_file in metadata_files:
                metadata_ext = os.path.splitext(metadata_file)[-1]
                reader, kwargs, arrow_converter = readers.get(metadata_ext)
                metadata.append(arrow_converter(reader(metadata_file, **kwargs)))

            if len(metadata) == 1:
                out = metadata[0]
            else:
                out = concat_blocks(upcast_tables(metadata), axis=0)

            if to_arrow:
                return out
            else:
                if use_polars and is_polars_available():
                    import polars as pl

                    return pl.from_arrow(out)
                else:
                    return out.to_pandas()

        except Exception as e:
            if use_polars and is_polars_available():
                import polars.exceptions as pl_ex

                if isinstance(e, (pl_ex.ComputeError, pl_ex.SchemaError)):
                    return self._read_metadata(metadata_files, use_polars=False)
            raise e

    def _create_features(
        self,
        schema: Union[Features, Dict[str, Any], pa.Schema, pa.Table],
        feature_metadata=None,
    ):
        _schema: Features = None
        if isinstance(schema, dict):
            entry = next(iter(schema.values()))
            if isinstance(entry, dict):
                _schema = Features.from_dict(schema)
            elif not hasattr(entry, "pa_type"):
                raise ValueError(
                    "Could not infer the schema of the dataset. Please provide the schema in the `features` argument."
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
                    new_schema[k] = self.INPUT_FEATURE(
                        dtype=DTYPE_MAP.get(v.dtype, v.dtype), metadata={}, id=v.id
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
                "data table\n"
                f"Available columns in data table: {metadata_columns_str}"
            )
        elif not metadata_columns_str:
            err_msg += (
                "metadata table\n" f"Available columns in data table: {features_str}"
            )
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
                logger.debug(f"Pattern match found: {dcol} -> {col}")
                if not other_cols or col.lower() in other_cols:
                    return col
    if possible_col and not required:
        logger.warning_once(
            f"A possible match for the {default_column_name} column was found: {possible_col}\n"
            "But it was not found in both the data and metadata tables.\n"
            "Please add or rename the column in the appropriate table.\n"
            f"Ignoring the {default_column_name} data.\n"
        )
        return None
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
