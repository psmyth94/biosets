import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Optional, Union

import numpy as np
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
from datasets import Dataset, IterableDataset

import biosets.utils.logging
from biosets.utils.import_util import (
    is_polars_available,
    is_ray_available,
)

if TYPE_CHECKING:
    pass

logger = biosets.utils.logging.get_logger(__name__)


def get_data_format(data):
    if is_ray_available() and "ray" in sys.modules:
        import ray.data.dataset

        if isinstance(data, ray.data.dataset.MaterializedDataset):
            return "ray"

    if isinstance(data, (Path, str)):
        return "io"
    if isinstance(data, np.ndarray):
        return "numpy"
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return "pandas"
    if is_polars_available() and "polars" in sys.modules:
        import polars as pl

        if isinstance(data, (pl.DataFrame, pl.Series)):
            return "polars"
    if isinstance(data, (pa.Table, pa.Array, pa.ChunkedArray)):
        return "arrow"
    if isinstance(data, (Dataset, IterableDataset)):
        return "dataset" if isinstance(data, Dataset) else "iterabledataset"
    if isinstance(data, list):
        if len(data) == 0:
            return None
        if isinstance(data[0], dict):
            return "dicts"
        return "list"
    if isinstance(data, (dict, MutableMapping)):
        if len(data) == 0:
            return None
        first_entry = next(iter(data.values()))
        if isinstance(first_entry, list):
            return "dict"
        return "row_dict"

    return None


class BaseDataConverter:
    """
    A class that provides methods to convert data between different formats.
    """

    dtype = None
    supports_named_columns = False

    def converters(self):
        return {
            "np": self.to_numpy,
            "numpy": self.to_numpy,
            "list": self.to_list,
            "row_dict": self.to_dict,
            "dict": self.to_dict,
            "dicts": self.to_dicts,
            "pd": self.to_pandas,
            "pandas": self.to_pandas,
            "dataframe": self.to_pandas,
            "series": self.to_series,
            "pl": self.to_polars,
            "polars": self.to_polars,
            "arrow": self.to_arrow,
            "ds": self.to_dataset,
            "dataset": self.to_dataset,
            "ids": self.to_iterabledataset,
            "iterabledataset": self.to_iterabledataset,
            "iterable": self.to_iterabledataset,
            "io": self.to_file,
            "dask": self.to_dask,
            "ray": self.to_ray,
            "sparse": self.to_csr,
            "csr": self.to_csr,
        }

    def to_list(self, X, **kwargs):
        raise NotImplementedError()

    def to_dict(self, X, **kwargs):
        raise NotImplementedError()

    def to_dicts(self, X, **kwargs):
        raise NotImplementedError()

    def to_numpy(self, X, **kwargs):
        raise NotImplementedError()

    def to_pandas(self, X, **kwargs):
        raise NotImplementedError()

    def to_series(self, X, **kwargs):
        raise NotImplementedError()

    def to_polars(self, X, **kwargs):
        raise NotImplementedError()

    def to_arrow(self, X, **kwargs):
        raise NotImplementedError()

    def to_dataset(self, X, **kwargs):
        raise NotImplementedError()

    def to_iterabledataset(self, X, **kwargs):
        raise NotImplementedError()

    def to_file(self, X, **kwargs):
        raise NotImplementedError()

    def to_dask(self, X, **kwargs):
        raise NotImplementedError()

    def to_ray(self, X, **kwargs):
        raise NotImplementedError()

    def to_csr(self, X, **kwargs):
        raise NotImplementedError()

    def _check_first_n_rows(
        self,
        X: Union[List[List[Any]], List[Dict[str, List[Any]]], Dict[str, List[Any]]],
        input_columns: Optional[Union[List[str], List[int]]] = None,
        n: int = 100,
    ):
        if input_columns is None:
            cols = set(self.get_column_names(X, generate_cols=True))
        else:
            cols = set(input_columns)
        if isinstance(X, (list, np.ndarray)):
            if isinstance(X[0], (list, tuple, np.ndarray)):
                if not isinstance(next(iter(cols)), int):
                    if len(X[0]) != len(cols):
                        raise ValueError(
                            "Column indices must be integers when input is a list of lists or ndarrays"
                        )
                    cols = set(range(len(X[0])))
                dtypes = {
                    i: type(v)
                    for i, v in enumerate(X[0])
                    if i in cols and v is not None
                }
                null_dtypes = set(range(len(X[0]))) - set(dtypes.keys())
            elif isinstance(X[0], dict):
                dtypes = {
                    k: type(v) for k, v in X[0].items() if k in cols and v is not None
                }
                null_dtypes = set(X[0].keys()) - set(dtypes.keys())

            count = 0
            nrows = min(n, len(X))
            while len(null_dtypes) > 0 and count < nrows:
                count += 1
                to_rem = []
                for k in null_dtypes:
                    if X[count][k] is not None:
                        to_rem.append(k)
                        dtypes[k] = type(X[count][k])
                null_dtypes -= set(to_rem)
        elif isinstance(X, dict):
            if isinstance(X[next(iter(X))], (list, tuple, np.ndarray)):
                dtypes = {
                    k: type(v[0])
                    for k, v in X.items()
                    if k in cols and v[0] is not None
                }
                null_dtypes = set(X.keys()) - set(dtypes.keys())
                count = 0
                nrows = min(n, len(next(iter(X.values()))))
                while len(null_dtypes) > 0 and count < nrows:
                    to_rem = []
                    for k in null_dtypes:
                        if X[k][count] is not None:
                            to_rem.append(k)
                            dtypes[k] = type(X[k][count])
                    count += 1
                    null_dtypes -= set(to_rem)
            else:
                dtypes = {
                    k: type(v) for k, v in X.items() if k in cols and v is not None
                }
                null_dtypes = set(X.keys()) - set(dtypes.keys())
        else:
            raise ValueError(f"Cannot check first n rows for type {type(X)}")

        return dtypes

    def _check_column(
        self,
        X,
        columns: Optional[Union[str, List[str], int, List[int]]] = None,
        str_only=False,
        as_list=True,
        convert=True,
    ) -> Optional[Union[str, List[str], int, List[int]]]:
        """Check if the column is a string or integer and convert it to the appropriate format if necessary.

        Args:
            X (Any): the input data
            columns (Union[str, List[str], int, List[int]], *optional*): the column name or index
            str_only (bool, optional): whether to return only strings. Defaults to False.

        Raises:
            ValueError: _description_

        Returns:
            Union[str, List[str], int, List[int]]: the column name or index
        """
        if pdt.is_list_like(columns) or isinstance(columns, (slice, range)):
            if (
                columns
                and str_only
                and (
                    isinstance(columns, (slice, range))
                    or pdt.is_integer_dtype(type(columns[0]))
                )
            ):
                if convert:
                    cols = self.get_column_names(X)
                    if isinstance(columns, slice):
                        columns = cols[columns]
                    else:
                        columns = [cols[i] for i in columns]
                else:
                    columns = [f"__column_{i}__" for i in columns]
            if not as_list:
                if len(columns) > 1:
                    raise ValueError("Column must be a single column name")
                columns = columns[0]
        else:
            if pdt.is_integer_dtype(type(columns)) and str_only:
                if convert:
                    columns = self.get_column_names(X)[columns]
                else:
                    raise ValueError("Column must be a string, not an integer")
            if as_list and columns is not None:
                columns = [columns]
        return columns

    def append_column(self, X, name, column):
        raise NotImplementedError()

    def add_column(self, X, index, name, column):
        raise NotImplementedError()

    def rename_column(self, X, name, new_name):
        raise NotImplementedError()

    def rename_columns(self, X, mapping):
        raise NotImplementedError()

    def select_columns(
        self,
        X,
        columns=None,
        **kwargs,
    ):
        raise NotImplementedError()

    def select_column(self, X, column, **kwargs):
        raise NotImplementedError()

    def set_column(self, X, column, value):
        raise NotImplementedError()

    def set_column_names(self, X, names, new_fingerprint=None):
        raise NotImplementedError()

    def to_frame(self, X, name=None):
        raise NotImplementedError()

    def select_rows(self, X, indices, **kwargs):
        raise NotImplementedError()

    def select_row(self, X, index):
        raise NotImplementedError()

    def unique(self, X, column=None):
        raise NotImplementedError()

    def replace(self, X, column=None, mapping={}):
        raise NotImplementedError()

    def nunique(self, X, column=None):
        raise NotImplementedError()

    def argmax(self, X, axis=0):
        raise NotImplementedError()

    def ge(self, X, value=None, column=None):
        raise NotImplementedError()

    def gt(self, X, value=None, column=None):
        raise NotImplementedError()

    def le(self, X, value=None, column=None):
        raise NotImplementedError()

    def lt(self, X, value=None, column=None):
        raise NotImplementedError()

    def eq(self, X, value=None, column=None):
        raise NotImplementedError()

    def ne(self, X, value=None, column=None):
        raise NotImplementedError()

    def get_column_names(self, X, generate_cols=False):
        raise NotImplementedError()

    def get_shape(self, X):
        raise NotImplementedError()

    def iter(self, X, batch_size, drop_last_batch=False):
        raise NotImplementedError()

    def concat(self, tables, axis=0, **kwargs):
        raise NotImplementedError()

    def is_array_like(self, X, min_row=10, min_col=10):
        return False

    def get_dtypes(self, X, columns=None):
        raise NotImplementedError()

    def is_categorical(self, X, column, threshold=None):
        raise NotImplementedError()
