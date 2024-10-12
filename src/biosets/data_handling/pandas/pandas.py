from typing import Union

import numpy as np
import pandas as pd
import pandas.api.types as pdt
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import (
    get_kwargs,
    pa_table_from_pandas_kwargs,
)

from ..base import BaseDataConverter


class PandasConverter(BaseDataConverter):
    dtype = "pd.DataFrame"
    supports_named_columns = True

    def to_numpy(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        return X.values

    def to_pandas(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        if isinstance(X, pd.Series):
            return X.to_frame(**get_kwargs(kwargs, X.to_frame))
        return X

    def to_series(self, X, **kwargs):
        if X.shape[1] == 1:
            return X.iloc[:, 0]
        if X.shape[0] == 1:
            return X.iloc[0, :]
        raise ValueError("Cannot convert multi-dimensional dataframe to series")

    def to_polars(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.from_pandas(X, **get_kwargs(kwargs, pl.from_pandas))

    def to_list(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        return X.values.tolist()

    def to_dict(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        return X.to_dict("list", **get_kwargs(kwargs, X.to_dict))

    def to_dicts(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        return X.to_dict("records", **get_kwargs(kwargs, X.to_dict))

    def to_arrow(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        if isinstance(X, pd.Series):
            return pa.Table.from_pandas(
                X.to_frame(), **pa_table_from_pandas_kwargs(kwargs)
            )
        return pa.Table.from_pandas(X, **pa_table_from_pandas_kwargs(kwargs))

    def to_dataset(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        from biosets import Dataset

        if isinstance(X, pd.Series):
            X = X.to_frame(**get_kwargs(kwargs, X.to_frame))
        return Dataset.from_pandas(X, **get_kwargs(kwargs, Dataset.from_pandas))

    def to_iterabledataset(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        def gen(**gen_kwargs):
            for _, row in X.iterrows():
                yield row.to_dict()

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: Union[pd.DataFrame, pd.Series], **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(X, **get_kwargs(kwargs, dd.from_pandas))

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_pandas(X, **get_kwargs(kwargs, ray.data.from_pandas))

    def to_csr(self, X, **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(X)

    def append_column(self, X, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False)
        if isinstance(column, (pd.Series, pd.DataFrame)):
            if isinstance(column, pd.Series):
                column.name = name
            else:
                column.columns = [name]
            column.index = X.index
            return X.join(column)
        return pd.concat(
            X.join(pd.DataFrame({name: column}, index=X.index)), copy=False
        )

    def add_column(self, X, index, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False)
        if isinstance(column, (pd.Series, pd.DataFrame)):
            if isinstance(column, pd.Series):
                column.name = name
            else:
                column.columns = [name]
            column.index = X.index
            return X.join(column)
        if isinstance(X, pd.Series):
            X = X.to_frame()
        X.insert(index, name, column)
        return X

    def rename_column(self, X, name, new_name):
        name = self._check_column(X, name, str_only=False, as_list=False)
        new_name = self._check_column(X, new_name, str_only=False, as_list=False)
        return X.rename(columns={name: new_name})

    def rename_columns(self, X, mapping):
        mapping = {k: mapping.get(k, k) for k in X.columns}
        return X.rename(columns=mapping)

    def select_columns(
        self, X, columns=None, feature_type=None, **kwargs
    ) -> pd.DataFrame:
        if columns is None:
            return X
        if isinstance(columns, list) and len(columns) == 0:
            return pd.DataFrame()
        if isinstance(X, pd.Series):
            column = self._check_column(
                X, columns, str_only=True, as_list=False, convert=True
            )
            return X.to_frame(column)
        columns = self._check_column(
            X, columns, str_only=False, as_list=True, convert=False
        )
        if pdt.is_list_like(columns) and all(
            pdt.is_integer_dtype(type(c)) for c in columns
        ):
            return X.iloc[:, columns]
        return X.loc[:, columns]

    def select_column(self, X, column=None, **kwargs) -> pd.Series:
        if column is None:
            if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
                return X.iloc[:, 0]
            return X
        column = self._check_column(
            X, column, str_only=False, as_list=False, convert=False
        )
        if isinstance(X, pd.Series):
            return (
                X if column is None or column == X.name or column == 0 else pd.Series()
            )
        if isinstance(column, int):
            return X.iloc[:, column]
        return X.loc[:, column]

    def set_column(self, X, column, value):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        X[column] = value
        return X

    def set_column_names(self, X, names, new_fingerprint=None):
        names = self._check_column(X, names, str_only=True, as_list=True, convert=False)
        if isinstance(X, pd.Series):
            X.name = names[0]
            return X
        X.columns = names
        return X

    def to_frame(self, X, name=None):
        if isinstance(X, pd.Series):
            name = self._check_column(
                X, name, str_only=True, as_list=False, convert=False
            )
            return X.to_frame(name)
        return X

    def select_rows(self, X, indices, **kwargs):
        return X.iloc[indices]

    def select_row(self, X, index):
        return X.iloc[index]

    def unique(self, X, column=None):
        return self.select_column(X, column).unique()

    def replace(self, X, column=None, mapping={}):
        return self.set_column(
            X, column, self.select_column(X, column).replace(mapping)
        )

    def nunique(self, X, column=None):
        return len(self.unique(X, column))

    def argmax(self, X, axis=0):
        return np.argmax(self.to_numpy(X), axis=axis)

    def ge(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return self.select_column(X, column) >= value
        return X >= value

    def gt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return self.select_column(X, column) > value
        return X > value

    def le(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return self.select_column(X, column) <= value
        return X <= value

    def lt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return self.select_column(X, column) < value
        return X < value

    def eq(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return self.select_column(X, column) == value
        return X == value

    def ne(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return self.select_column(X, column) != value
        return X != value

    def get_column_names(self, X, generate_cols=False):
        if isinstance(X, pd.Series):
            return [X.name]
        return X.columns.tolist()

    def get_shape(self, X):
        return X.shape

    def iter(self, X, batch_size, drop_last_batch=False):
        total_length = self.get_shape(X)[0]
        num_batches = total_length // batch_size
        if not drop_last_batch or total_length % batch_size == 0:
            num_batches += 1

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            yield X.iloc[start_idx:end_idx]

    def concat(self, tables, axis=0, **kwargs):
        if "ignore_index" not in kwargs:
            kwargs["ignore_index"] = True
            for i in range(len(tables)):
                if isinstance(tables[i], (pd.Series, pd.DataFrame)):
                    tables[i] = tables[i].reset_index(drop=True)
                if isinstance(tables[i], pd.Series):
                    tables[i] = tables[i].to_frame()
        if "copy" not in kwargs:
            kwargs["copy"] = False

        out = pd.concat(tables, axis=axis, **get_kwargs(kwargs, pd.concat))
        out.columns = [str(c) for tbl in tables for c in tbl.columns]
        return out

    def is_array_like(self, X):
        return True

    def get_dtypes(self, X, columns=None):
        dtype_map = {
            "object": "string",
            "double": "float64",
            "float": "float32",
            "int": "int64",
        }

        if columns is None:
            columns = self.get_column_names(X)
        dtypes = self.select_columns(X, columns).dtypes
        if isinstance(dtypes, pd.Series):
            return {
                k: dtype_map.get(str(v), str(v)) for k, v in dtypes.to_dict().items()
            }
        if isinstance(columns, list):
            columns = columns[0]
        return {columns: dtype_map.get(str(dtypes), str(dtypes))}

    def is_categorical(self, X, column, threshold=None):
        ser = self.select_column(X, column)

        if isinstance(ser.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(
            ser.dtype
        ):
            return True
        if threshold and pd.api.types.is_integer_dtype(ser.dtype):
            return ser.nunique() / ser.shape[0] < threshold
        return False
