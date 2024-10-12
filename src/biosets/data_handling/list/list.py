import numpy as np
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils import logging
from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import (
    get_kwargs,
    np_array_kwargs,
    pa_table_from_pandas_kwargs,
)

from ..base import BaseDataConverter

logger = logging.get_logger(__name__)


class ListConverter(BaseDataConverter):
    dtype = "list"

    def _is_fatten(self, X):
        return not pdt.is_list_like(X[0])

    def to_list(self, X, **kwargs):
        return X

    def to_dict(self, X, **kwargs):
        X = self.to_frame(X)

        names = kwargs.get("names", None)
        if names is None:
            if not isinstance(X[0], (list, tuple)):
                num_cols = 1
                X = [[x] for x in X]
            else:
                num_cols = len(X[0])
            names = [f"{i}" for i in range(num_cols)]
        return {name: [x[i] for x in X] for i, name in enumerate(names)}

    def to_dicts(self, X, **kwargs):
        X = self.to_frame(X)
        names = kwargs.get("names", None)
        if names is None:
            if not isinstance(X[0], (list, tuple)):
                pass
            else:
                len(X[0])
            names = [f"{i}" for i in range(len(X[0]))]
        return [dict(zip(names, x)) for x in X]

    def to_numpy(self, X, **kwargs):
        return np.array(X, **np_array_kwargs(kwargs))

    def to_pandas(self, X, **kwargs):
        if self._is_fatten(X):
            return pd.Series(X, **get_kwargs(kwargs, pd.Series.__init__))
        return pd.DataFrame(X, **get_kwargs(kwargs, pd.DataFrame.__init__))

    def to_polars(self, X, **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        if self._is_fatten(X):
            return pl.Series(X, **get_kwargs(kwargs, pl.Series.__init__))
        return pl.from_records(X, **get_kwargs(kwargs, pl.from_records))

    def to_arrow(self, X, **kwargs):
        if self._is_fatten(X):
            return pa.array(X)
        return pa.Table.from_pandas(
            self.to_pandas(X, **kwargs), **pa_table_from_pandas_kwargs(kwargs)
        )

    def to_dataset(self, X, **kwargs):
        from biosets import Dataset

        input = self.to_pandas(X, **kwargs)
        if isinstance(input, pd.Series):
            input = input.to_frame()
        return Dataset.from_pandas(input, **get_kwargs(kwargs, Dataset.from_pandas))

    def to_iterabledataset(self, X, **kwargs):
        def gen(**gen_kwargs):
            for x in X:
                yield x

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X, **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, dd.from_pandas)
        )

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_items(X, **get_kwargs(kwargs, ray.data.from_items))

    def to_csr(self, X, **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(self.to_numpy(X), **get_kwargs(kwargs, csr_matrix.__init__))

    def append_column(self, X, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        if self._is_fatten(X):
            X = np.array(X).reshape(-1, 1).tolist()
        if self._is_fatten(column):
            column = np.array(column).reshape(-1, 1).tolist()
        return np.concatenate((np.array(X), np.array(column)), axis=1).tolist()

    def add_column(self, X, index, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        if self._is_fatten(X):
            X = np.array(X).reshape(-1, 1).tolist()

        if not self._is_fatten(column):
            column = np.array(column).flatten()
        return np.insert(np.array(X), index, column, axis=1).tolist()

    def rename_column(self, X, name, new_name):
        logger.warning_once(
            "rename_column is not supported for list format. Returning the original data."
        )
        return X

    def rename_columns(self, X, mapping):
        logger.warning_once(
            "rename_columns is not supported for list format. Returning the original data."
        )
        return X

    def select_columns(self, X, columns=None, **kwargs):
        if columns is None:
            return X
        if not columns:
            return []

        columns = np.array(
            self._check_column(X, columns, str_only=False, as_list=True, convert=False)
        )
        if self._is_fatten(X):
            X = np.array(X).reshape(-1, 1).tolist()
        return np.array(X)[:, columns].tolist()

    def select_column(self, X, column, **kwargs):
        column = self._check_column(
            X, column, str_only=False, as_list=False, convert=False
        )
        if self._is_fatten(X) and column == 0:
            return X
        return np.array(X)[:, column].flatten().tolist()

    def set_column(self, X, column, values):
        column = self._check_column(
            X, column, str_only=False, as_list=False, convert=False
        )
        X = np.array(X)
        if self._is_fatten(X):
            return np.array(values).flatten().tolist()
        if not self._is_fatten(values):
            values = np.array(values).flatten()
        X[:, column] = values
        return X.tolist()

    def select_rows(self, X, indices, **kwargs):
        return [X[i] for i in indices]

    def select_row(self, X, index):
        return X[index]

    def unique(self, X, column=None):
        return list(set(self.select_column(X, column)))

    def replace(self, X, column=None, mapping={}):
        if not self._is_fatten(X):
            return self.set_column(
                X, column, [mapping.get(x[column], x[column]) for x in X]
            )
        return self.set_column(X, column, [mapping.get(x, x) for x in X])

    def argmax(self, X, axis=0):
        X = np.array(X)
        return np.argmax(X, axis=axis).tolist()

    def nunique(self, X, column=None):
        return len(self.unique(X, column))

    def get_column_names(self, X, generate_cols=False):
        if generate_cols:
            x_dims = self.get_shape(X)
            if len(x_dims) == 1:
                return [0]
            return list(range(x_dims[1]))
        return None

    def set_column_names(self, X, names, new_fingerprint=None):
        return X

    def get_shape(self, X):
        if self._is_fatten(X):
            return (len(X),)
        return (len(X), len(X[0]))

    def to_frame(self, X, name=None):
        if not self._is_fatten(X):
            return X
        return np.array(X).reshape(-1, 1).tolist()

    def iter(self, X, batch_size, drop_last_batch=False):
        total_length = self.get_shape(X)[0]
        # Calculate the number of batches
        num_batches = total_length // batch_size
        if not drop_last_batch or total_length % batch_size == 0:
            num_batches += 1

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            yield X[start_idx:end_idx]

    def concat(self, tables, axis=0, **kwargs):
        return np.concatenate(tables, axis=axis).tolist()

    def is_array_like(self, X, min_row=10, min_col=10):
        # check first 100 rows to see if all have the same length
        valid = True
        if self._is_fatten(X):
            return False
        length = len(X[0])
        if length <= min_col or len(X) <= min_row:
            return False
        for i in range(min(100, len(X))):
            if len(X[i]) != length:
                valid = False
                break
        return valid

    def get_dtypes(self, X, columns=None):
        full_name_mapper = {
            "str": "string",
            "int": "int64",
        }
        if self._is_fatten(X):
            return {0: full_name_mapper.get(type(X[0]).__name__, type(X[0]).__name__)}
        X = np.array(X)
        x_dims = X.shape
        dtype = X.dtype
        if pd.api.types.is_string_dtype(dtype):
            dtype = "string"
        if len(x_dims) == 1:
            return {0: str(dtype)}
        return {i: str(dtype) for i in range(x_dims[1])}

    def is_categorical(self, X, column, threshold=None):
        arr = self.select_column(X, column)
        if isinstance(arr[0], str):
            return True
        if threshold:
            n_unique = self.nunique(X, column)
            return n_unique / self.get_shape(X)[0] < threshold
        return False
