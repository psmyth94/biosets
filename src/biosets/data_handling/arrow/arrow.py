from typing import List, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset, IterableDataset
from datasets.features.features import _ArrayXDExtensionType
from datasets.formatting.formatting import (
    _is_array_with_nulls,
    _is_zero_copy_only,
)

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import (
    get_kwargs,
    pa_array_to_pandas_kwargs,
    pa_table_to_pandas_kwargs,
)

from ..base import BaseDataConverter


class ArrowConverter(BaseDataConverter):
    dtype = "pa.Table"
    supports_named_columns = True

    def to_numpy(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        if isinstance(X, (pa.ChunkedArray, pa.Array)):
            return self._arrow_array_to_numpy(X)
        return np.vstack([self._arrow_array_to_numpy(col) for col in X.columns]).T

    def to_list(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        if isinstance(X, (pa.ChunkedArray, pa.Array)):
            return self._arrow_array_to_list(X)
        return [self._arrow_array_to_list(col) for col in X.columns]

    def _arrow_array_to_list(self, pa_array: pa.Array) -> List:
        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(
                    pa_array.type.storage_dtype, unnest=True
                )
                array: List = [
                    row
                    for chunk in pa_array.chunks
                    for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
            else:
                zero_copy_only = _is_zero_copy_only(pa_array.type) and all(
                    not _is_array_with_nulls(chunk) for chunk in pa_array.chunks
                )
                array: List = [
                    row
                    for chunk in pa_array.chunks
                    for row in chunk.to_numpy(zero_copy_only=zero_copy_only)
                ]
        else:
            if isinstance(pa_array.type, _ArrayXDExtensionType):
                # don't call to_pylist() to preserve dtype of the fixed-size array
                zero_copy_only = _is_zero_copy_only(
                    pa_array.type.storage_dtype, unnest=True
                )
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only)
            else:
                zero_copy_only = _is_zero_copy_only(
                    pa_array.type
                ) and not _is_array_with_nulls(pa_array)
                array: List = pa_array.to_numpy(zero_copy_only=zero_copy_only).tolist()
        return array

    def _arrow_array_to_numpy(self, pa_array: pa.Array) -> np.ndarray:
        """This method is adapted from datasets.formatting.formatting.NumpyArrowExtractor._arrow_array_to_numpy

        Args:
            pa_array (pa.Array): _description_

        Returns:
            np.ndarray: _description_
        """
        array = self._arrow_array_to_list(pa_array)
        if len(array) > 0:
            if any(
                (
                    isinstance(x, np.ndarray)
                    and (x.dtype == object or x.shape != array[0].shape)
                )
                or (isinstance(x, float) and np.isnan(x))
                for x in array
            ):
                return np.array(array, copy=False, dtype=object)
        return np.array(array, copy=False)

    def to_pandas(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        if isinstance(X, (pa.ChunkedArray, pa.Array)):
            return X.to_pandas(**pa_array_to_pandas_kwargs(kwargs))
        return X.to_pandas(**pa_table_to_pandas_kwargs(kwargs))

    def to_polars(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.from_arrow(X, **get_kwargs(kwargs, pl.from_arrow))

    def to_arrow(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        # Since X is already an Arrow table, return it as is
        return X

    def to_dict(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        if isinstance(X, (pa.ChunkedArray, pa.Array)):
            name = kwargs.get("name", "data")
            return {name: X.to_pylist()}
        return X.to_pydict()

    def to_dicts(
        self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs
    ) -> List[dict]:
        if isinstance(X, (pa.ChunkedArray, pa.Array)):
            return [{kwargs.get("name", "data"): X.to_pylist()}]

        return X.to_pylist()

    def to_dataset(self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs):
        return Dataset(X, **get_kwargs(kwargs, Dataset.__init__))

    def to_iterabledataset(
        self, X: Union[pa.Array, pa.ChunkedArray, pa.Table], **kwargs
    ):
        def gen(X):
            for row in X.to_pylist():
                yield row

        return IterableDataset.from_generator(gen)

    def to_dask(self, X, **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        dd.from_pandas(X.to_pandas(), **get_kwargs(kwargs, dd.from_pandas))

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_arrow(X, **get_kwargs(kwargs, ray.data.from_arrow))

    def to_csr(self, X, **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(self.to_numpy(X), **get_kwargs(kwargs, csr_matrix.__init__))

    def append_column(self, X, name, column):
        name = self._check_column(X, name, str_only=True, as_list=False, convert=False)
        if not isinstance(column, (pa.Array, pa.ChunkedArray)):
            if isinstance(column, pa.Table) and len(column.column_names) == 1:
                column = column.column(column.column_names[0])
            else:
                column = pa.array(column)

        return X.append_column(name, column)

    def add_column(self, X, index, name, column):
        name = self._check_column(X, name, str_only=True, as_list=False, convert=False)
        return X.add_column(
            index, pa.field(name, pa.array(column).type), pa.array(column)
        )

    def rename_column(self, X: pa.Table, name, new_name):
        name = self._check_column(X, name, str_only=True, as_list=False, convert=False)
        new_name = self._check_column(
            X, new_name, str_only=True, as_list=False, convert=False
        )
        new_columns = [c if c != name else new_name for c in X.column_names]
        return X.rename_column(name, new_columns)

    def rename_columns(self, X, mapping):
        if isinstance(mapping, list):
            if len(mapping) == len(X.column_names):
                return X.rename_columns(mapping)
            raise ValueError(
                "Length of mapping must be equal to the number of columns in the "
                "table. Provide either a dictionary or a list of the same length as "
                "the number of columns."
            )
        new_columns = [mapping.get(c, c) for c in X.column_names]
        return X.rename_columns(new_columns)

    def select_columns(self, X, columns=None, feature_type=None, **kwargs):
        if columns is None:
            return X
        if not columns:
            return []

        if isinstance(X, (pa.Array, pa.ChunkedArray)):
            columns = self._check_column(
                X, columns, str_only=True, as_list=False, convert=True
            )
            return pa.table({columns: X})
        columns = self._check_column(
            X, columns, str_only=True, as_list=True, convert=True
        )
        if columns:
            return X.select(columns)
        if feature_type:
            from datasets import Features

            from biosets.integration import DatasetsPatcher

            with DatasetsPatcher():
                features = Features.from_arrow_schema(X.schema)
            return X.select(
                [f for f, v in features.items() if isinstance(v, feature_type)]
            )
        return X

    def select_column(self, X, column=None, **kwargs):
        if column is None:
            if isinstance(X, pa.Table) and X.shape[1] == 1:
                return X.column(X.column_names[0])
            return X
        column = self._check_column(
            X, column, str_only=False, as_list=False, convert=True
        )
        if isinstance(X, (pa.Array, pa.ChunkedArray)):
            return X
        return X.column(column)

    def set_column(self, X, column, value):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        col_pos = X.column_names.index(column) if isinstance(column, str) else column
        return X.set_column(col_pos, column, value)

    def set_column_names(self, X, names, new_fingerprint=None):
        names = self._check_column(X, names, str_only=True, as_list=True, convert=False)
        if isinstance(X, (pa.Array, pa.ChunkedArray)):
            return X
        return X.rename_columns(names)

    def select_rows(self, X, indices, **kwargs):
        if not isinstance(indices, (pa.Array, pa.ChunkedArray)):
            indices = pa.array(indices)
        return X.take(indices)

    def select_row(self, X, index):
        return X.slice(index, 1)

    def unique(self, X, column=None):
        return self.select_column(X, column).unique()

    def replace(self, X, column=None, mapping={}):
        col = self.select_column(X, column)
        new_col = pa.array([mapping.get(v.as_py(), v.as_py()) for v in col])
        return self.set_column(X, column, new_col)

    def nunique(self, X, column=None):
        return len(self.unique(X, column))

    def argmax(self, X, axis=0):
        return np.argmax(self.to_numpy(X), axis=axis)

    def ge(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return pc.greater_equal(X, value)

    def gt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return pc.greater(X, value, column=None)

    def le(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return pc.less_equal(X, value, column=None)

    def lt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return pc.less(X, value, column=None)

    def eq(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return pc.equal(X, value, column=None)

    def ne(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return pc.not_equal(X, value, column=None)

    def get_column_names(self, X, generate_cols=False):
        if isinstance(X, (pa.Array, pa.ChunkedArray)):
            return ["0"]
        return X.column_names

    def get_shape(self, X):
        if isinstance(X, (pa.Array, pa.ChunkedArray)):
            return (len(X),)
        return X.shape

    def to_frame(self, X, name=None):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=True)
        if isinstance(X, (pa.Array, pa.ChunkedArray)):
            return pa.table({name: X})
        return X

    def iter(self, X, batch_size, drop_last_batch=False):
        total_length = self.get_shape(X)[0]
        # Calculate the number of batches
        num_batches = total_length // batch_size
        if not drop_last_batch or total_length % batch_size == 0:
            num_batches += 1

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            yield X.slice(start_idx, batch_size)

    def concat(self, tables, axis=0, **kwargs):
        if axis == 0:
            return pa.concat_tables(tables, promote="default")
        out = tables[0]
        for i, t in enumerate(tables[1:]):
            if isinstance(t, (pa.Array, pa.ChunkedArray)):
                out = self.append_column(out, str(i), t)
            else:
                for col in t.column_names:
                    out = self.append_column(out, col, t.column(col))
        return out

    def is_array_like(self, X, min_row=10, min_col=10):
        return True

    def get_dtypes(self, X, columns=None):
        dtype_map = {
            "double": "float64",
        }

        return {
            s.name: dtype_map.get(str(s.type), str(s.type))
            for s in self.select_columns(X, columns).schema
        }

    def get_numerical_features(self, X):
        dtypes = {s.name: s.type for s in X.schema}
        return [
            k
            for k, dtype in dtypes.items()
            if pa.types.is_floating(dtype) or pa.types.is_integer(dtype)
        ]

    def is_categorical(self, X, column, threshold=None):
        col = self.select_column(X, column)
        num_rows = len(col)
        if pa.types.is_dictionary(col.type) or pa.types.is_string(col.type):
            return True
        if threshold and pa.types.is_integer(col.type):
            return self.nunique(X, column) / num_rows < threshold
        return False

    def get_categorical_features(self, X):
        dtypes = {s.name: s.type for s in X.schema}
        return [
            k
            for k, dtype in dtypes.items()
            if pa.types.is_dictionary(dtype) or pa.types.is_string(dtype)
        ]
