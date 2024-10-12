import numpy as np
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils import logging
from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import (
    get_kwargs,
    pa_table_from_arrays_kwargs,
)

from ..base import BaseDataConverter

logger = logging.get_logger(__name__)


class NumPyConverter(BaseDataConverter):
    """
    A class that provides methods to convert data to and from NumPy arrays.
    """

    dtype = "np.ndarray"
    supports_named_columns = False

    def to_list(self, X: np.ndarray, **kwargs):
        return X.tolist()

    def to_dict(self, X: np.ndarray, names=None, **kwargs):
        if names is None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            names = [f"{i}" for i in range(X.shape[1])]
        return {name: X[:, i].tolist() for i, name in enumerate(names)}

    def to_dicts(self, X: np.ndarray, names=None, **kwargs):
        if names is None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            names = [f"{i}" for i in range(X.shape[1])]
        return [dict(zip(names, x)) for x in X]

    def to_numpy(self, X: np.ndarray, **kwargs):
        return X

    def to_pandas(self, X: np.ndarray, **kwargs):
        if "columns" not in kwargs:
            if X.ndim == 1:
                shape = (X.shape[0], 1)
                X = X.reshape(-1, 1)
            else:
                shape = X.shape
            kwargs["columns"] = [f"col_{i}" for i in range(shape[1])]
        return pd.DataFrame(X, **get_kwargs(kwargs, pd.DataFrame.__init__))

    def to_series(self, X: np.ndarray, **kwargs):
        if len(X.shape) == 1:
            return pd.Series(X, **get_kwargs(kwargs, pd.Series.__init__))
        else:
            raise ValueError("Cannot convert 2D array to series")

    def to_polars(self, X: np.ndarray, **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.DataFrame(X, **get_kwargs(kwargs, pl.DataFrame.__init__))

    def to_arrow(self, X: np.ndarray, **kwargs):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        names = kwargs.get("names", None)
        if names is None:
            names = [f"col_{i}" for i in range(X.shape[1])]
        return pa.Table.from_arrays(
            X.T, names=names, **pa_table_from_arrays_kwargs(kwargs)
        )

    def to_dataset(self, X: np.ndarray, **kwargs):
        from biosets import Dataset

        return Dataset(
            self.to_arrow(X, **kwargs), **get_kwargs(kwargs, Dataset.__init__)
        )

    def to_iterabledataset(self, X: np.ndarray, **kwargs):
        def gen(**gen_kwargs):
            for x in self.to_dicts(X, **gen_kwargs):
                yield x

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: np.ndarray, **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.array as da

        return da.from_array(X, **get_kwargs(kwargs, da.from_array))

    def to_ray(self, X: np.ndarray, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_numpy(X, **get_kwargs(kwargs, ray.data.from_numpy))

    def to_csr(self, X: np.ndarray, **kwargs):
        from scipy.sparse import csr_matrix

        return csr_matrix(X)

    def _check_column(self, columns, as_list=True):
        if isinstance(columns, (list, np.ndarray)):
            if len(columns) and isinstance(columns[0], (str, np.str_)):
                raise ValueError("Column names are not supported for NumPy arrays")
            if not as_list and len(columns) > 1:
                raise ValueError("Only one column can be selected")
            columns = np.array(columns)
        if isinstance(columns, (str, np.str_)):
            raise ValueError("Column names are not supported for NumPy arrays")
        return columns

    def append_column(self, X: np.ndarray, name, column):
        if len(column.shape) == 1:
            column = column[:, None]
        if len(X.shape) == 1:
            return np.hstack([X[:, None], column])
        return np.hstack([X, column])

    def add_column(self, X: np.ndarray, index, name, column):
        if len(column.shape) == 1:
            column = column[:, None]
        if len(X.shape) == 1:
            X = X[:, None]
        return np.hstack([X[:, :index], column, X[:, index:]])

    def select_rows(self, X: np.ndarray, indices, **kwargs):
        return X[np.array(indices)]

    def select_row(self, X: np.ndarray, index):
        return X[index]

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

    def select_columns(self, X: np.ndarray, columns=None, feature_type=None, **kwargs):
        if columns is None:
            return X
        if not columns:
            return np.array([])

        columns = self._check_column(columns)
        if columns is None:
            return X
        return X[:, columns]

    def select_column(self, X: np.ndarray, column, **kwargs):
        column = self._check_column(column, as_list=False)
        if len(X.shape) == 1:
            return X
        return X[:, column]

    def ge(self, X: np.ndarray, value, column=None):
        if column is not None:
            return self.select_column(X, column) >= value
        return X >= value

    def gt(self, X: np.ndarray, value, column=None):
        if column is not None:
            return self.select_column(X, column) > value
        return X > value

    def le(self, X: np.ndarray, value, column=None):
        if column is not None:
            return self.select_column(X, column) <= value
        return X <= value

    def lt(self, X: np.ndarray, value, column=None):
        if column is not None:
            return self.select_column(X, column) < value
        return X < value

    def eq(self, X: np.ndarray, value, column=None):
        if column is not None:
            return self.select_column(X, column) == value
        return X == value

    def ne(self, X: np.ndarray, value, column=None):
        if column is not None:
            return self.select_column(X, column) != value
        return X != value

    def unique(self, X: np.ndarray, column=None):
        if len(X.shape) == 1:
            return np.unique(X)
        return np.unique(self.select_column(X, column))

    def nunique(self, X: np.ndarray, column=None):
        return len(self.unique(X, column))

    def replace(self, X: np.ndarray, column=None, mapping={}):
        if len(X.shape) == 1:
            return np.array([mapping.get(x, x) for x in X])
        return np.array([mapping.get(x, x) for x in self.select_column(X, column)])

    def argmax(self, X: np.ndarray, axis=0):
        if len(X.shape) == 1:
            return np.argmax(X)
        return np.argmax(X, axis=axis)

    def set_column(self, X: np.ndarray, column, value):
        if len(X.shape) == 1:
            return value
        else:
            column = self._check_column(column, as_list=False)
            X[:, column] = value
        return X

    def set_column_names(self, X: np.ndarray, names, new_fingerprint=None):
        return X

    def get_column_names(self, X: np.ndarray, generate_cols=False):
        if generate_cols:
            x_dims = X.shape
            if len(x_dims) == 1:
                return [0]
            return list(range(x_dims[1]))
        return None

    def get_column_names_by_feature_type(self, X: np.ndarray, feature_type):
        raise ValueError("NumPy does not support feature type selection")

    def get_shape(self, X: np.ndarray):
        return X.shape

    def to_frame(self, X: np.ndarray, name=None):
        if X.ndim == 1:
            return X[:, None]
        return X

    def iter(self, X: np.ndarray, batch_size, drop_last_batch=False):
        for i in range(0, len(X), batch_size):
            yield X[i : i + batch_size]

    def concat(self, tables, axis=0, **kwargs):
        return np.concatenate(tables, axis=axis)

    def is_array_like(self, X: np.ndarray, min_row=10, min_col=10):
        return True

    def get_dtypes(self, X: np.ndarray, columns=None):
        columns = self._check_column(columns)
        x_dims = X.shape
        dtype = X.dtype
        if pd.api.types.is_string_dtype(dtype):
            dtype = "string"
        elif pd.api.types.is_float_dtype(dtype):
            dtype = "float"
        if len(x_dims) == 1:
            return {0: str(dtype)}
        return {i: str(dtype) for i in range(x_dims[1])}

    def get_numeric_features(self, X: np.ndarray):
        x_dims = X.shape
        if pdt.is_numeric_dtype(X):
            if len(x_dims) == 1:
                return [0]
            return list(range(X.shape[1]))

    def is_categorical(self, X: np.ndarray, column, threshold=None):
        column = self._check_column(column, as_list=False)
        if len(X.shape) == 1:
            return False
        if pd.api.types.is_string_dtype(X) or pd.api.types.is_object_dtype(X):
            return True
        if threshold:
            n_unique = self.nunique(X, column)
            return n_unique / X.shape[0] < threshold
        return False

    def get_categorical_features(self, X: np.ndarray):
        x_dims = X.shape
        if pd.api.types.is_string_dtype(X) or pd.api.types.is_object_dtype(X):
            if len(x_dims) == 1:
                return [0]
            return list(range(X.shape[1]))
