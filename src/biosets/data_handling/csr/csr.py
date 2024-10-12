from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import (
    get_kwargs,
    pa_table_from_pandas_kwargs,
)

from ..base import BaseDataConverter

if TYPE_CHECKING:
    import scipy.sparse as sp


class CSRConverter(BaseDataConverter):
    """
    A class that provides methods to convert data to and from CSR (Compressed Sparse Row) format.
    """

    dtype = "csr_matrix"

    def to_list(self, X: "sp.csr_matrix", **kwargs):
        return self.to_numpy(X, **kwargs).tolist()

    def to_dict(self, X: "sp.csr_matrix", names=None, **kwargs):
        if names is None:
            names = [f"{i}" for i in range(X.shape[1])]
        return {
            name: X.getcol(i).toarray().flatten().tolist()
            for i, name in enumerate(names)
        }

    def to_dicts(self, X: "sp.csr_matrix", names=None, **kwargs):
        if names is None:
            names = [f"{i}" for i in range(X.shape[1])]
        return [dict(zip(names, row.toarray().flatten())) for row in X]

    def to_numpy(self, X: "sp.csr_matrix", **kwargs):
        return X.toarray(**get_kwargs(kwargs, X.toarray))

    def to_pandas(self, X: "sp.csr_matrix", **kwargs):
        return pd.DataFrame(
            self.to_numpy(X, **kwargs), **get_kwargs(kwargs, pd.DataFrame.__init__)
        )

    def to_series(self, X: "sp.csr_matrix", **kwargs):
        if X.shape[1] == 1:
            return pd.Series(
                self.to_numpy(X, **kwargs).flatten(),
                **get_kwargs(kwargs, pd.Series.__init__),
            )
        else:
            raise ValueError("Cannot convert 2D CSR matrix to series")

    def to_polars(self, X: "sp.csr_matrix", **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.DataFrame(
            self.to_numpy(X, **kwargs), **get_kwargs(kwargs, pl.DataFrame.__init__)
        )

    def to_arrow(self, X: "sp.csr_matrix", **kwargs):
        return pa.Table.from_pandas(
            self.to_pandas(X, **kwargs), **pa_table_from_pandas_kwargs(kwargs)
        )

    def to_dataset(self, X: "sp.csr_matrix", **kwargs):
        from biosets import Dataset

        return Dataset(
            self.to_arrow(X, **kwargs), **get_kwargs(kwargs, Dataset.__init__)
        )

    def to_iterabledataset(self, X: "sp.csr_matrix", **kwargs):
        def gen(**gen_kwargs):
            for x in self.to_dicts(X, **gen_kwargs):
                yield x

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: "sp.csr_matrix", **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.array as da

        return da.from_array(X, **get_kwargs(kwargs, da.from_array))

    def to_ray(self, X: "sp.csr_matrix", **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_numpy(
            self.to_numpy(X, **kwargs), **get_kwargs(kwargs, ray.data.from_numpy)
        )

    def append_column(self, X, name, column):
        requires_backends(self.append_column, "scipy")
        import scipy.sparse as sp

        if not isinstance(column, sp.csr_matrix):
            if not isinstance(column, np.ndarray):
                column = np.array(column)
            if column.ndim == 1:
                column = column[:, None]
            if column.shape[0] != X.shape[0]:
                raise ValueError(
                    "The new column must have the same number of rows as the input matrix."
                )

            # Convert the new column to a CSR matrix
            column = sp.csr_matrix(column)

        return sp.hstack([X, column], format="csr")

    def add_column(self, X: "sp.csr_matrix", index, name, column):
        requires_backends(self.add_column, "scipy")
        import scipy.sparse as sp

        if not isinstance(column, sp.csr_matrix):
            if not isinstance(column, np.ndarray):
                column = np.array(column)
            if column.ndim == 1:
                column = column[:, None]
            if column.shape[0] != X.shape[0]:
                raise ValueError(
                    "The new column must have the same number of rows as the input matrix."
                )

            # Convert the new column to a CSR matrix
            column = sp.csr_matrix(column)

        left_part = X[:, :index] if index > 0 else None
        right_part = X[:, index:] if index < X.shape[1] else None

        if left_part is not None and right_part is not None:
            X = sp.hstack([left_part, column, right_part], format="csr")
        elif left_part is not None:
            X = sp.hstack([left_part, column], format="csr")
        elif right_part is not None:
            X = sp.hstack([column, right_part], format="csr")
        else:
            X = column

        return X

    def select_rows(self, X: "sp.csr_matrix", indices, **kwargs):
        return X[indices]

    def select_row(self, X: "sp.csr_matrix", index):
        return X[index]

    def _check_column(self, columns, limit=None):
        if isinstance(columns, (list, np.ndarray)):
            if len(columns) and isinstance(columns[0], (str, np.str_)):
                raise ValueError("Column names are not supported for NumPy arrays")
            if limit and len(columns) > limit:
                raise ValueError("Too many columns selected.")
            columns = np.array(columns)
        if isinstance(columns, (str, np.str_)):
            raise ValueError("Column names are not supported for NumPy arrays")
        return columns

    def select_columns(self, X: "sp.csr_matrix", columns=None, **kwargs):
        requires_backends(self.select_columns, "scipy")
        import scipy.sparse as sp

        if columns is None:
            return X
        if not columns:
            return sp.csr_matrix([])
        columns = self._check_column(columns)
        if columns is None:
            return X
        return X[:, columns]

    def select_column(self, X: "sp.csr_matrix", column, **kwargs):
        column = self._check_column(column, 1)
        if isinstance(column, int):
            return X.getcol(column)
        raise ValueError("Column must be an integer index")

    def unique(self, X: "sp.csr_matrix", column=None):
        if column is not None:
            return np.unique(X.getcol(column).toarray())
        return np.unique(self.to_numpy(X))

    def nunique(self, X: "sp.csr_matrix", column=None):
        return len(self.unique(X, column))

    def replace(self, X: "sp.csr_matrix", column=None, mapping={}):
        requires_backends(self.replace, "scipy")
        import scipy.sparse as sp

        arr = self.to_numpy(X)
        if column is not None:
            col_data = arr[:, column]
            new_col_data = np.array([mapping.get(item, item) for item in col_data])
            arr[:, column] = new_col_data
        else:
            for i in range(arr.shape[1]):
                col_data = arr[:, i]
                new_col_data = np.array([mapping.get(item, item) for item in col_data])
                arr[:, i] = new_col_data
        return sp.csr_matrix(arr)

    def argmax(self, X: "sp.csr_matrix", axis=0):
        if axis == 0:
            # For column-wise max, we find the argmax of each column
            # We convert to CSC.
            X_csc = X.tocsc()
            max_indices = np.zeros(X.shape[1], dtype=int)
            for col in range(X.shape[1]):
                col_array = X_csc.getcol(col).toarray().ravel()
                max_indices[col] = col_array.argmax()
            return max_indices
        elif axis == 1:
            # For row-wise max, CSR format supports efficient row operations
            max_indices = np.zeros(X.shape[0], dtype=int)
            for row in range(X.shape[0]):
                row_array = X.getrow(row).toarray().ravel()
                max_indices[row] = row_array.argmax()
            return max_indices
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows)")

    def ge(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X.getcol(column) >= value
        return X >= value

    def gt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X.getcol(column) > value
        return X > value

    def le(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X.getcol(column) <= value
        return X <= value

    def lt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X.getcol(column) < value
        return X < value

    def eq(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X.getcol(column) == value
        return X == value

    def ne(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X.getcol(column) != value
        return X != value

    def set_column(self, X: "sp.csr_matrix", column, value):
        requires_backends(self.set_column, "scipy")
        import scipy.sparse as sp

        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.ndim == 1:
            value = value[:, None]
        if value.shape[0] != X.shape[0]:
            raise ValueError(
                "Value array must have the same number of rows as the input matrix."
            )

        # Create a new sparse column matrix
        new_col = sp.csr_matrix(value)

        # Replace the specified column with the new sparse column
        left_part = X[:, :column] if column > 0 else None
        right_part = X[:, column + 1 :] if column < X.shape[1] - 1 else None

        if left_part is not None and right_part is not None:
            X = sp.hstack([left_part, new_col, right_part], format="csr")
        elif left_part is not None:
            X = sp.hstack([left_part, new_col], format="csr")
        elif right_part is not None:
            X = sp.hstack([new_col, right_part], format="csr")
        else:
            X = new_col  # Only when X has exactly one column

        return X

    def set_column_names(self, X: "sp.csr_matrix", names, new_fingerprint=None):
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

    def get_shape(self, X: "sp.csr_matrix"):
        return X.shape

    def to_frame(self, X: "sp.csr_array", **kwargs):
        requires_backends(self.to_frame, "scipy")
        import scipy.sparse as sp

        return sp.csr_matrix(X)

    def iter(self, X: "sp.csr_matrix", batch_size, drop_last_batch=False):
        for i in range(0, len(X), batch_size):
            yield X[i : i + batch_size]

    def concat(self, tables, axis=0, **kwargs):
        requires_backends(self.concat, "scipy")
        import scipy.sparse as sp

        if axis == 0:
            return sp.vstack(tables, format="csr")
        elif axis == 1:
            return sp.hstack(tables, format="csr")
        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns)")

    def is_array_like(self, X: np.ndarray, min_row=10, min_col=10):
        return True

    def get_dtypes(self, X: np.ndarray, columns=None):
        columns = self._check_column(columns)
        x_dims = X.shape
        dtype = X.dtype
        if pd.api.types.is_string_dtype(dtype):
            dtype = "string"
        if len(x_dims) == 1:
            return {0: str(dtype)}
        return {i: str(dtype) for i in range(x_dims[1])}
