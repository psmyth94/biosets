from typing import TYPE_CHECKING, Union

import numpy as np
from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import get_kwargs

from ..base import BaseDataConverter

if TYPE_CHECKING:
    import polars as pl


class PolarsConverter(BaseDataConverter):
    dtype = "pl.DataFrame"
    supports_named_columns = True

    def to_numpy(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return X.to_numpy(**get_kwargs(kwargs, X.to_numpy))

    def to_pandas(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return X.to_pandas(**get_kwargs(kwargs, X.to_pandas))

    def to_polars(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return X

    def to_list(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return X.to_numpy(**get_kwargs(kwargs, X.to_numpy)).tolist()

    def to_records(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return [list(v.values()) for v in X.to_dicts()]

    def to_dict(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return {name: X[name].to_list() for name in X.columns}

    def to_dicts(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return X.to_dicts()

    def to_arrow(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        return X.to_arrow()

    def to_dataset(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        from biosets import Dataset

        requires_backends(self.to_dataset, "polars")
        import polars as pl

        if isinstance(X, pl.Series):
            return Dataset.from_pandas(
                self.to_pandas(X.to_frame(), **kwargs),
                **get_kwargs(kwargs, Dataset.from_pandas),
            )

        return Dataset.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, Dataset.from_pandas)
        )

    def to_iterabledataset(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        def gen(**gen_kwargs):
            for row in X.to_dicts():
                yield row

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, dd.from_pandas)
        )

    def to_ray(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_arrow(
            self.to_arrow(X, **kwargs), **get_kwargs(kwargs, ray.data.from_arrow)
        )

    def to_csr(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(X.to_numpy(), **get_kwargs(kwargs, csr_matrix.__init__))

    def append_column(
        self, X: Union["pl.DataFrame", "pl.Series"], name, column
    ) -> "pl.DataFrame":
        requires_backends(self.append_column, "polars")
        import polars as pl

        name = self._check_column(X, name, str_only=False, as_list=False, convert=True)
        if isinstance(column, pl.Series):
            return X.insert_column(len(X.columns), column)
        return X.insert_column(len(X.columns), pl.Series(name, column))

    def add_column(
        self, X: Union["pl.DataFrame", "pl.Series"], index, name, column
    ) -> "pl.DataFrame":
        requires_backends(self.add_column, "polars")
        import polars as pl

        name = self._check_column(X, name, str_only=False, as_list=False, convert=True)
        if isinstance(column, pl.Series):
            return X.insert_column(index, column)
        return X.insert_column(index, pl.Series(name, column))

    def rename_column(
        self, X: Union["pl.DataFrame", "pl.Series"], name, new_name
    ) -> "pl.DataFrame":
        name = self._check_column(X, name, str_only=False, as_list=False, convert=True)
        new_name = self._check_column(
            X, new_name, str_only=False, as_list=False, convert=True
        )
        new_columns = {k: k if k != name else new_name for k in X.columns}
        return X.rename(new_columns)

    def rename_columns(
        self, X: Union["pl.DataFrame", "pl.Series"], mapping
    ) -> "pl.DataFrame":
        mapping = {k: mapping.get(k, k) for k in X.columns}
        return X.rename(mapping)

    def select_columns(
        self,
        X: Union["pl.DataFrame", "pl.Series"],
        columns=None,
        **kwargs,
    ) -> "pl.DataFrame":
        requires_backends(self.select_columns, "polars")
        import polars as pl

        if columns is None:
            return X
        if not columns:
            return pl.DataFrame()

        if isinstance(X, pl.Series):
            column = self._check_column(
                X, columns, str_only=True, as_list=False, convert=True
            )
            return self.to_frame(X, column)

        columns = self._check_column(
            X, columns, str_only=True, as_list=True, convert=True
        )
        if columns is None:
            return X
        return X.select(columns)

    def select_column(
        self, X: Union["pl.DataFrame", "pl.Series"], column, **kwargs
    ) -> "pl.Series":
        requires_backends(self.select_column, "polars")
        import polars as pl

        if column is None:
            if isinstance(X, pl.DataFrame) and X.shape[1] == 1:
                return X.get_column(X.columns[0])
            return X
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=True
        )
        if isinstance(X, pl.Series):
            return X if column == X.name else pl.Series()
        return X.get_column(column)

    def set_column(self, X: Union["pl.DataFrame", "pl.Series"], column, value):
        requires_backends(self.set_column, "polars")
        import polars as pl

        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if isinstance(X, pl.Series):
            if not isinstance(value, pl.Series):
                value = pl.Series(value)
            return value.rename(column)

        return X.with_column(pl.Series(value).alias(column))

    def set_column_names(
        self, X: Union["pl.DataFrame", "pl.Series"], names, new_fingerprint=None
    ):
        requires_backends(self.set_column_names, "polars")
        import polars as pl

        names = self._check_column(X, names, str_only=True, as_list=True, convert=False)
        if isinstance(X, pl.Series):
            if isinstance(names, list) and len(names) == 1:
                return X.rename(names[0])
            return X.rename(names)

        return X.rename(
            {old_name: new_name for old_name, new_name in zip(X.columns, names)}
        )

    def to_frame(self, X: Union["pl.DataFrame", "pl.Series"], **kwargs):
        import polars as pl

        name = kwargs.pop("name", None)
        name = self._check_column(X, name, str_only=False, as_list=False, convert=True)
        if isinstance(X, pl.Series):
            return X.to_frame(name=name)
        return X

    def select_rows(
        self, X: Union["pl.DataFrame", "pl.Series"], indices, new_fingerprint=None
    ):
        return X[indices]

    def select_row(self, X: Union["pl.DataFrame", "pl.Series"], index):
        return X.slice(index, 1)

    def unique(self, X: Union["pl.DataFrame", "pl.Series"], column=None):
        return self.select_column(X, column).unique()

    def replace(self, X: Union["pl.DataFrame", "pl.Series"], column=None, mapping={}):
        return self.set_column(
            X, column, self.select_column(X, column).replace(mapping)
        )

    def nunique(self, X: Union["pl.DataFrame", "pl.Series"], column=None):
        return len(self.unique(X, column))

    def argmax(self, X: Union["pl.DataFrame", "pl.Series"], axis=0):
        requires_backends(self.argmax, "polars")
        import polars as pl

        if isinstance(X, pl.Series):
            if axis == 0:
                return X.arg_max()
            else:
                return 0
        if axis == 1:
            return X.with_columns(pl.col("*").arg_max())
        else:
            return np.argmax(X.to_numpy(), axis=axis)

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

    def get_column_names(
        self, X: Union["pl.DataFrame", "pl.Series"], generate_cols=False
    ):
        requires_backends(self.get_column_names, "polars")
        import polars as pl

        if isinstance(X, pl.Series):
            return [X.name]
        return list(X.columns)

    def get_shape(self, X: Union["pl.DataFrame", "pl.Series"]):
        return X.shape

    def iter(
        self, X: Union["pl.DataFrame", "pl.Series"], batch_size, drop_last_batch=False
    ):
        total_length = self.get_shape(X)[0]
        num_batches = total_length // batch_size
        if not drop_last_batch or total_length % batch_size == 0:
            num_batches += 1

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            yield X.slice(start_idx, batch_size)

    def concat(self, tables, axis=0, **kwargs):
        requires_backends(self.concat, "polars")
        import polars as pl

        if axis == 0:
            return pl.concat(tables, how="vertical_relaxed")
        else:
            for i, table in enumerate(tables):
                if i == 0:
                    continue
                if isinstance(table, pl.Series):
                    tables[i] = self.to_frame(table, **kwargs)
            return pl.concat(tables, how="horizontal")

    def is_array_like(
        self, X: Union["pl.DataFrame", "pl.Series"], min_row=10, min_col=10
    ):
        return True

    def get_dtypes(self, X: Union["pl.DataFrame", "pl.Series"], columns=None):
        requires_backends(self.get_dtypes, "polars")
        import polars as pl

        if isinstance(X, pl.Series):
            schema = {X.name: X.dtype}
        else:
            schema = {k: v for k, v in self.select_columns(X, columns).schema.items()}

        dtypes = {}
        for k, v in schema.items():
            dtype = str(v).lower()
            if dtype == "utf8":
                dtypes[k] = "string"
            elif dtype == "boolean":
                dtypes[k] = "bool"
            elif dtype == "double":
                dtypes[k] = "float64"
            elif "datetime" in dtype:
                if "'ns'" in dtype:
                    dtypes[k] = "datetime64[ns]"
                elif "'us'" in dtype:
                    dtypes[k] = "datetime64[us]"
                elif "'ms'" in dtype:
                    dtypes[k] = "datetime64[ms]"
                elif "'s'" in dtype:
                    dtypes[k] = "datetime64[s]"
                else:
                    dtypes[k] = "datetime64[us]"

            else:
                dtypes[k] = dtype
        return dtypes

    def is_categorical(
        self, X: Union["pl.DataFrame", "pl.Series"], column, threshold=None
    ):
        requires_backends(self.is_categorical, "polars")
        from polars.datatypes import Categorical, IntegerType, Object

        ser = self.select_column(X, column)
        if isinstance(ser.dtype, Categorical) or isinstance(ser.dtype, Object):
            return True
        if threshold and isinstance(ser.dtype, IntegerType):
            return self.nunique(ser) / ser.len() < threshold
        return False
