import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset, IterableDataset

from biosets.utils.import_util import (
    is_dask_available,
    is_polars_available,
)

from ..utils.inspect import InvalidColumnSelectionError
from .arrow import ArrowConverter
from .base import BaseDataConverter, get_data_format
from .dask import DaskConverter
from .dataset import DatasetConverter
from .dict import DictConverter
from .dicts import DictsConverter
from .io import IOConverter
from .iterabledataset import IterableDatasetConverter
from .list import ListConverter
from .numpy import NumPyConverter
from .pandas import PandasConverter
from .polars import PolarsConverter
from .row_dict import RowDictConverter

if TYPE_CHECKING:
    import polars as pl
    import ray.data.dataset

_FORMAT_TO_CONVERTER: Dict[str, BaseDataConverter] = {
    "np": NumPyConverter(),
    "numpy": NumPyConverter(),
    "list": ListConverter(),
    "dict": DictConverter(),
    "row_dict": RowDictConverter(),
    "dicts": DictsConverter(),
    "pd": PandasConverter(),
    "pandas": PandasConverter(),
    "pl": PolarsConverter(),
    "polars": PolarsConverter(),
    "arrow": ArrowConverter(),
    "ds": DatasetConverter(),
    "dataset": DatasetConverter(),
    "ids": IterableDatasetConverter(),
    "iterabledataset": IterableDatasetConverter(),
    "iterable": IterableDatasetConverter(),
    "io": IOConverter(),
    # "ray": RayConverter(),
    # "sparse": CSRConverter(),
    # "csr": CSRConverter(),
}

if is_dask_available():
    _FORMAT_TO_CONVERTER["dask"] = DaskConverter()

SHORT_TO_LONG_FORMAT_NAMES = {
    "np": "numpy",
    "pd": "pandas",
    "pl": "polars",
    "ds": "dataset",
    "ids": "iterabledataset",
}


class DataHandler:
    @staticmethod
    def get_format(X, options: Optional[Union[List[str], str]] = None):
        """Function to retrieve the source format and target format for the data.
        If source format is within the options, the target format is the same as the source format.
        Otherwise, the target format is the first option in the list.
        """
        if not options:
            from_format = get_data_format(X)
            return from_format

        from_format = get_data_format(X)

        to_format = options
        if isinstance(to_format, list):
            to_format = [SHORT_TO_LONG_FORMAT_NAMES.get(f, f) for f in to_format]
            if from_format in to_format:
                return from_format, from_format
            to_format = to_format[0]

        if to_format not in _FORMAT_TO_CONVERTER:
            raise ValueError(f"Unsupported format: {to_format}")

        return to_format

    @staticmethod
    def to_format(
        X,
        options: Optional[Union[List[str], str]] = None,
        input_columns=None,
        target_format=None,
        streaming=None,
        **kwargs,
    ):
        if X is None:
            raise ValueError("Cannot convert NoneType to a table format")
        from_format = get_data_format(X)
        if from_format is None:
            raise ValueError(f"Cannot determine format of {type(X)}")
        to_format = (
            SHORT_TO_LONG_FORMAT_NAMES.get(target_format, target_format)
            if target_format is not None
            else None
        )
        if not to_format:
            to_format = DataHandler.get_format(X, options=options)

        converter: BaseDataConverter = _FORMAT_TO_CONVERTER[from_format]

        if from_format == "iterabledataset" and streaming is not None:
            converter.streaming = streaming

        if input_columns:
            column_selector = DataHandler.select_columns
            if isinstance(input_columns, (int, str)):
                column_selector = DataHandler.select_column

            if DataHandler.supports_named_columns(from_format):
                return converter.converters()[to_format](
                    column_selector(X, input_columns), **kwargs
                )
            elif DataHandler.supports_named_columns(to_format):
                return column_selector(
                    converter.converters()[to_format](X, **kwargs), input_columns
                )
            else:
                return converter.converters()[to_format](
                    column_selector(X, input_columns), **kwargs
                )
        else:
            return converter.converters()[to_format](X, **kwargs)

    @staticmethod
    def to_pandas(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> pd.DataFrame:
        return DataHandler.to_format(
            X,
            target_format="pandas",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_numpy(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> np.ndarray:
        return DataHandler.to_format(
            X,
            target_format="numpy",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_polars(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> "pl.DataFrame":
        return DataHandler.to_format(
            X,
            target_format="polars",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_arrow(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> pa.Table:
        return DataHandler.to_format(
            X,
            target_format="arrow",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_dict(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> Dict:
        return DataHandler.to_format(
            X,
            target_format="dict",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_list(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> List:
        return DataHandler.to_format(
            X,
            target_format="list",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_dataset(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> Dataset:
        return DataHandler.to_format(
            X,
            target_format="dataset",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_iterabledataset(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> IterableDataset:
        return DataHandler.to_format(
            X,
            target_format="iterabledataset",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_ray(
        X,
        input_columns=None,
        streaming=None,
        **kwargs,
    ) -> "ray.data.dataset.MaterializedDataset":
        return DataHandler.to_format(
            X,
            target_format="ray",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def to_csr(X, input_columns=None, streaming=None, **kwargs):
        return DataHandler.to_format(
            X,
            target_format="csr",
            input_columns=input_columns,
            streaming=streaming,
            **kwargs,
        )

    @staticmethod
    def supports_named_columns(data_or_format):
        if not isinstance(data_or_format, str):
            format = get_data_format(data_or_format)
        else:
            format = data_or_format
        converter = _FORMAT_TO_CONVERTER.get(format, None)
        if converter:
            return converter.supports_named_columns
        return False

    @staticmethod
    def append_column(X, name, column):
        format = get_data_format(X)
        col = DataHandler.to_format(column, target_format=format)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].append_column(X, name=name, column=col)

        raise ValueError(f"{type(X)} does not support column appending")

    @staticmethod
    def add_column(X, index, name, column):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].add_column(
                X, index=index, name=name, column=column
            )
        raise ValueError(f"{type(X)} does not support column adding")

    @staticmethod
    def select_rows(X, indices, **kwargs):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].select_rows(
                X, indices=indices, **kwargs
            )

        raise ValueError(f"{type(X)} does not support integer row selection")

    @staticmethod
    def select_row(X, index):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].select_row(X, index=index)

        raise ValueError(f"{type(X)} does not support integer row selection")

    @staticmethod
    def select_columns(X, columns=None, feature_type=None, **kwargs):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].select_columns(
                X, columns=columns, feature_type=feature_type, **kwargs
            )

        raise InvalidColumnSelectionError(
            f"{type(X)} does not support string column selection"
        )

    @staticmethod
    def drop_columns(X, columns):
        if columns is None:
            return X
        if isinstance(columns, str):
            columns = [columns]
        return DataHandler.select_columns(
            X, columns=DataHandler.exclude_column_names(X, columns)
        )

    @staticmethod
    def drop_column(X, column):
        if column is None:
            return X
        if isinstance(column, str):
            column = [column]
        return DataHandler.select_columns(
            X, columns=DataHandler.exclude_column_names(X, column)
        )

    @staticmethod
    def rename_column(X, name, new_name):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].rename_column(X, name, new_name)

        raise ValueError(f"{type(X)} does not support column renaming")

    @staticmethod
    def rename_columns(X, mapping):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].rename_columns(X, mapping=mapping)

        raise ValueError(f"{type(X)} does not support column renaming")

    @staticmethod
    def select_column(X, column, **kwargs):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].select_column(
                X, column=column, **kwargs
            )

    @staticmethod
    def unique(X, column=None):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].unique(X, column=column)

        raise ValueError(f"{type(X)} does not support unique")

    @staticmethod
    def replace(X, column=None, mapping={}):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].replace(
                X, column=column, mapping=mapping
            )

        raise ValueError(f"{type(X)} does not support replace")

    @staticmethod
    def nunique(X, column=None):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].nunique(X, column=column)

        raise ValueError(f"{type(X)} does not support nunique")

    @staticmethod
    def argmax(X, axis=0):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].argmax(X, axis=axis)

        raise ValueError(f"{type(X)} does not support argmax")

    @staticmethod
    def ge(X, other):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].ge(X, other)

        raise ValueError(f"{type(X)} does not support greater than or equal to")

    @staticmethod
    def gt(X, other):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].gt(X, other)

        raise ValueError(f"{type(X)} does not support greater than")

    @staticmethod
    def le(X, other):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].le(X, other)

        raise ValueError(f"{type(X)} does not support less than or equal to")

    @staticmethod
    def lt(X, other):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].lt(X, other)

        raise ValueError(f"{type(X)} does not support less than")

    @staticmethod
    def eq(X, other):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].eq(X, other)

        raise ValueError(f"{type(X)} does not support equal to")

    @staticmethod
    def ne(X, other):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].ne(X, other)

        raise ValueError(f"{type(X)} does not support not equal to")

    @staticmethod
    def set_column(X, column, value):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].set_column(
                X, column=column, value=value
            )

        raise ValueError(f"{type(X)} does not support setting columns")

    @staticmethod
    def set_column_names(X, names, new_fingerprint=None):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].set_column_names(
                X, names=names, new_fingerprint=new_fingerprint
            )

    @staticmethod
    def exclude_column_names(X, columns):
        if not columns:
            return X
        input_columns = DataHandler.get_column_names(X, generate_cols=True)

        input_map = dict.fromkeys(input_columns)
        if (
            isinstance(columns, list)
            and len(columns) > 0
            and isinstance(columns[0], int)
        ):
            columns = [input_columns[col] for col in columns]
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, int):
            columns = [input_columns[columns]]

        for col in columns:
            input_map.pop(col)
        return list(input_map.keys())

    @staticmethod
    def get_column_names(X, generate_cols=False):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].get_column_names(
                X, generate_cols=generate_cols
            )

    @staticmethod
    def get_column_names_by_feature_type(X, feature_type):
        if isinstance(X, (Dataset, IterableDataset)):
            if feature_type:
                return [
                    k
                    for k, v in X._info.features.items()
                    if isinstance(v, feature_type)
                ]
            else:
                return X.column_names
        raise ValueError(f"{type(X)} does not support feature type selection")

    @staticmethod
    def get_column_indices(X, col_names=None, raise_if_missing=True):
        if col_names is None:
            col_names = DataHandler.get_column_names(X, generate_cols=True)

        if len(col_names) > 0 and isinstance(col_names[0], int):
            return col_names
        col_map = {
            name: i
            for i, name in enumerate(
                DataHandler.get_column_names(X, generate_cols=True)
            )
        }

        return [
            col_map[name] for name in col_names if raise_if_missing or name in col_map
        ]

    @staticmethod
    def get_shape(X):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].get_shape(X)

        raise ValueError(f"{type(X)} does not support shape retrieval")

    @staticmethod
    def to_frame(X, name=None):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].to_frame(X, name=name)

        return X

    @staticmethod
    def iter(X, batch_size, drop_last_batch=False):
        if isinstance(X, IterableDataset):
            return X.iter(batch_size, drop_last_batch=drop_last_batch)
        if isinstance(X, Dataset):
            return X.iter(batch_size, drop_last_batch=drop_last_batch)

        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].iter(
                X, batch_size=batch_size, drop_last_batch=drop_last_batch
            )

        raise ValueError(f"{type(X)} does not support iteration")

    @staticmethod
    def concat(tables, axis=0, **kwargs):
        if len(tables) == 1:
            return tables[0]
        format = get_data_format(tables[0])
        tbls = [tables[0]]
        for table in tables[1:]:
            tbls.append(DataHandler.to_format(table, format, **kwargs))

        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].concat(tbls, axis=axis, **kwargs)

        raise ValueError(f"{type(tbls[0])} is not a recognized data type")

    @staticmethod
    def is_array_like(X, min_row=10, min_col=10):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].is_array_like(
                X, min_row=min_row, min_col=min_col
            )

        return False

    @staticmethod
    def get_dtypes(X, columns=None):
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].get_dtypes(X, columns=columns)

        raise ValueError(f"Cannot determine dtypes for {type(X)}")

    @staticmethod
    def get_numeric_features(
        X,
    ):
        if is_polars_available() and "polars" in sys.modules:
            import polars

        if isinstance(X, pd.DataFrame):
            return X.select_dtypes(include=["number"]).columns

        if isinstance(X, pa.Table):
            dtypes = {s.name: s.type for s in X.schema}
            return [
                k
                for k, dtype in dtypes.items()
                if pa.types.is_floating(dtype) or pa.types.is_integer(dtype)
            ]

        if is_polars_available() and isinstance(X, polars.DataFrame):
            import polars as pl
            import polars.selectors as plsel

            return X.select(plsel.by_dtype(pl.NUMERIC_DTYPES)).columns

        if isinstance(X, (Dataset, IterableDataset)):
            from biosets.features import NUMERIC_FEATURES

            dtypes = {s.name: s.type for s in X.data.table.schema}
            return [
                k
                for k, dtype in dtypes.items()
                if pa.types.is_floating(dtype)
                or pa.types.is_integer(dtype)
                or isinstance(X._info.features[k], NUMERIC_FEATURES)
            ]

        if isinstance(X, (list, dict, np.ndarray)):
            dtypes = DataHandler._check_first_n_rows(
                X,
                input_columns=DataHandler.get_column_names(X, generate_cols=True),
                n=1000,
            )
            return [
                k for k, dtype in dtypes.items() if pd.api.types.is_numeric_dtype(dtype)
            ]

        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].get_numeric_features(X)

        raise ValueError(f"Cannot determine if {type(X)} has numeric type features")

    @staticmethod
    def is_categorical(X, column, threshold=None):
        format = get_data_format(X)
        if threshold is not None and threshold > 1:
            threshold = min(threshold / DataHandler.get_shape(X)[0], 1)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].is_categorical(
                X, column=column, threshold=threshold
            )
        raise ValueError(f"Cannot determine if {type(X)} is categorical")

    @staticmethod
    def get_categorical_features(X):
        if isinstance(X, pd.DataFrame):
            return list(X.select_dtypes(include=["category", "object"]).columns)
        if isinstance(X, pd.Series):
            return list(
                X.to_frame().select_dtypes(include=["category", "object"]).columns
            )
        if isinstance(X, pa.Table):
            dtypes = {s.name: s.type for s in X.schema}
            return [
                k
                for k, dtype in dtypes.items()
                if pa.types.is_dictionary(dtype) or pa.types.is_string(dtype)
            ]

        if (
            is_polars_available()
            and "polars" in sys.modules
            and isinstance(X, sys.modules["polars"].DataFrame)
        ):
            import polars.selectors as plsel

            return X.select(plsel.by_dtype((pl.Categorical, pl.Object))).columns

        if isinstance(X, (Dataset, IterableDataset)):
            from biosets.features import CATEGORICAL_FEATURES

            return [
                k
                for k, v in X._info.features.items()
                if pa.types.is_dictionary(v.pa_type)
                or pa.types.is_string(v.pa_type)
                or isinstance(v, CATEGORICAL_FEATURES)
            ]

        if isinstance(X, (list, dict)):
            dtypes = DataHandler._check_first_n_rows(
                X,
                input_columns=DataHandler.get_column_names(X, generate_cols=True),
                n=1000,
            )
            return [
                k
                for k, dtype in dtypes.items()
                if pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ]
        format = get_data_format(X)
        if format in _FORMAT_TO_CONVERTER:
            return _FORMAT_TO_CONVERTER[format].get_categorical_features(X)

        raise ValueError(f"Cannot determine if {type(X)} is categorical")
