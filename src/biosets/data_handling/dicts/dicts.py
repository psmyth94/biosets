from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import (
    get_kwargs,
    np_array_kwargs,
    pa_table_from_pandas_kwargs,
)

from ..base import BaseDataConverter


class DictsConverter(BaseDataConverter):
    dtype = "List[Dict[str, Any]]"
    supports_named_columns = True

    def to_list(self, X: List[Dict[str, Any]], **kwargs):
        return X

    def to_dict(self, X: List[Dict[str, Any]], **kwargs):
        names = kwargs.get("names", None)
        if names is None:
            if len(X) == 0:
                raise ValueError("Cannot automatically select names for empty list")
            names = list(X[0].keys())
        return {name: [x[name] for x in X] for name in names}

    def to_dicts(self, X: List[Dict[str, Any]], **kwargs):
        return X

    def to_numpy(self, X: List[Dict[str, Any]], **kwargs):
        return np.array(self.to_list(X, **kwargs), **np_array_kwargs(kwargs))

    def to_pandas(self, X: List[Dict[str, Any]], **kwargs):
        return pd.DataFrame(X, **get_kwargs(kwargs, pd.DataFrame.__init__))

    def to_series(self, X, **kwargs):
        if len(X) == 1:
            return pd.Series(X[0], **get_kwargs(kwargs, pd.Series.__init__))
        else:
            raise ValueError("Cannot convert list of dictionaries to series")

    def to_polars(self, X: List[Dict[str, Any]], **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.from_dicts(X, **get_kwargs(kwargs, pl.from_dicts))

    def to_arrow(self, X: List[Dict[str, Any]], **kwargs):
        return pa.Table.from_pandas(
            self.to_pandas(X, **kwargs), **pa_table_from_pandas_kwargs(kwargs)
        )

    def to_dataset(self, X: List[Dict[str, Any]], **kwargs):
        from biosets import Dataset

        return Dataset.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, Dataset.from_pandas)
        )

    def to_iterabledataset(self, X: List[Dict[str, Any]], **kwargs):
        def gen(**gen_kwargs):
            for x in X:
                yield x

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: List[Dict[str, Any]], **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, dd.from_pandas)
        )

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_arrow(
            pa.Table.from_pylist(X), **get_kwargs(kwargs, ray.data.from_arrow)
        )

    def to_csr(self, X, **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(self.to_numpy(X), **get_kwargs(kwargs, csr_matrix.__init__))

    def append_column(self, X, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        return [{name: column[i], **X[i]} for i in range(len(X))]

    def add_column(self, X, index, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        left_cols = set(list(X[0].keys())[:index])
        right_cols = set(list(X[0].keys())[index:])
        return [
            {
                **{k: v for k, v in X[i].items() if k in left_cols},
                **{name: column[i]},
                **{k: v for k, v in X[i].items() if k in right_cols},
            }
            for i in range(len(X))
        ]

    def select_rows(self, X, indices, **kwargs):
        return [X[i] for i in indices]

    def select_row(self, X, index):
        return X[index]

    def rename_column(self, X, name, new_name):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        new_name = self._check_column(
            X, new_name, str_only=False, as_list=False, convert=False
        )
        return [{new_name if k == name else k: v for k, v in x.items()} for x in X]

    def rename_columns(self, X, mapping):
        return [{mapping.get(k, k): v for k, v in x.items()} for x in X]

    def select_columns(self, X, columns=None, feature_type=None, **kwargs):
        if columns is None:
            return X
        if not columns:
            return []

        columns = self._check_column(
            X, columns, str_only=True, as_list=True, convert=True
        )
        return [{k: x[k] for k in columns} for x in X]

    def select_column(self, X, column, **kwargs):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=True
        )
        return [x[column] for x in X]

    def unique(self, X, column=None):
        return list(set(self.select_column(X, column)))

    def replace(self, X, column=None, mapping={}):
        return self.set_column(
            X, column, [mapping.get(x[column], x[column]) for x in X]
        )

    def nunique(self, X, column=None):
        return len(self.unique(X, column))

    def argmax(self, X, axis=0):
        if axis == 0:
            return {k: np.argmax([x[k] for x in X]) for k in X[0].keys()}
        else:
            cols = []
            for x in X:
                for k, v in x.items():
                    if len(cols) <= k:
                        cols.append([])
                    cols[k].append(v)
            return {k: np.argmax(v, axis=axis) for k, v in enumerate(cols)}

    def ge(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return [x[column] >= value for x in X]
        raise ValueError("A column must be specified.")

    def gt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return [x[column] > value for x in X]
        raise ValueError("A column must be specified.")

    def le(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return [x[column] <= value for x in X]
        raise ValueError("A column must be specified.")

    def lt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return [x[column] < value for x in X]
        raise ValueError("A column must be specified.")

    def eq(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return [x[column] == value for x in X]
        raise ValueError("A column must be specified.")

    def ne(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return [x[column] != value for x in X]
        raise ValueError("A column must be specified.")

    def set_column(self, X, column, value):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        for x in X:
            x[column] = value
        return X

    def set_column_names(self, X, names, new_fingerprint=None):
        names = self._check_column(X, names, str_only=True, as_list=True, convert=False)
        return [{name: x[i] for i, name in enumerate(names)} for x in X]

    def get_column_names(self, X, generate_cols=False):
        return list(X[0].keys())

    def get_shape(self, X):
        return (len(X), len(X[0]))

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
        if axis == 0:
            return [x for table in tables for x in table]

        out = tables[0]
        if not isinstance(out[0], (list, dict)):
            out = [[x] for x in out]

        for i, t in enumerate(tables[1:]):
            if isinstance(t, list):
                for j, x in enumerate(t):
                    if isinstance(x, dict):
                        out[j].update(x)
                    elif isinstance(x, list):
                        out[j].extend(x)
                    else:
                        out[j] += x
            else:
                for j, x in enumerate(t):
                    out[j].append(x)
        return out

    def is_array_like(self, X, min_row=10, min_col=10):
        # check first 100 rows to see if all have the same keys
        valid = True
        cols = set(X[0].keys())

        # check if the number of columns and rows are greater than the minimum required to be considered array-like
        # this is to avoid false positives for small lists of dictionaries
        if len(cols) <= min_col or len(X) <= min_row:
            return False

        for i in range(min(100, len(X))):
            if cols - set(X[i].keys()):
                valid = False
                break
        return valid

    def get_dtypes(self, X, columns=None):
        columns = self._check_column(
            X, columns, str_only=True, as_list=True, convert=True
        )
        full_name_mapper = {
            "str": "string",
            "int": "int64",
        }
        dtypes = self._check_first_n_rows(X, input_columns=columns, n=1000)
        return {
            k: full_name_mapper.get(v.__name__, v.__name__) for k, v in dtypes.items()
        }

    def is_categorical(self, X, column, threshold=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=True
        )
        if pd.api.types.is_string_dtype(self.get_dtypes(X, columns=[column])[column]):
            return True
        if threshold:
            n_unique = self.nunique(X, column)
            return n_unique / len(X) < threshold
        return False
