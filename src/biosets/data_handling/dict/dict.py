from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import get_kwargs, np_array_kwargs

from ..arrow.arrow import ArrowConverter
from ..base import BaseDataConverter, get_data_format
from ..csr.csr import CSRConverter
from ..dask.dask import DaskConverter
from ..dataset.dataset import DatasetConverter
from ..iterabledataset.iterabledataset import IterableDatasetConverter
from ..list.list import ListConverter
from ..numpy.numpy import NumPyConverter
from ..pandas.pandas import PandasConverter
from ..polars.polars import PolarsConverter
from ..ray.ray import RayConverter

_FORMAT_TO_CONVERTER = {
    "np": NumPyConverter(),
    "numpy": NumPyConverter(),
    "list": ListConverter(),
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
    "ray": RayConverter(),
    "sparse": CSRConverter(),
    "csr": CSRConverter(),
    "dask": DaskConverter(),
}


class DictConverter(BaseDataConverter):
    dtype = "dict"
    supports_named_columns = True

    def to_list(self, X: Dict[str, list], **kwargs):
        first_entry = next(iter(X.values()))
        ncol = len(first_entry)
        return [[X[k][i] for k in X] for i in range(ncol)]

    def to_dict(self, X: Dict[str, list], **kwargs):
        # This method is trivial for a DictConverter as X is already a dict
        return X

    def to_dicts(self, X: Dict[str, list], **kwargs):
        # Returns a list of dictionaries if the values are themselves dicts
        first_entry = next(iter(X.values()))
        return [{k: v[i] for k, v in X.items()} for i in range(len(first_entry))]

    def to_numpy(self, X: Dict[str, list], **kwargs):
        return np.array(self.to_list(X, **kwargs), **np_array_kwargs(kwargs))

    def to_pandas(self, X: Dict[str, list], **kwargs):
        return pd.DataFrame(X, **get_kwargs(kwargs, pd.DataFrame.__init__))

    def to_polars(self, X: Dict[str, list], **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.DataFrame(X, **get_kwargs(kwargs, pl.DataFrame.__init__))

    def to_arrow(self, X: Dict[str, list], **kwargs):
        return pa.table(X)

    def to_dataset(self, X: Dict[str, list], **kwargs):
        from biosets import Dataset

        return Dataset.from_dict(X, **get_kwargs(kwargs, Dataset.from_dict))

    def to_iterabledataset(self, X: Dict[str, list], **kwargs):
        def gen(**gen_kwargs):
            for key, value in X.items():
                yield {key: value}

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: Dict[str, list], **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_dict(X, **get_kwargs(kwargs, dd.from_dict))

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_arrow(
            pa.table(X), **get_kwargs(kwargs, ray.data.from_arrow)
        )

    def to_csr(self, X, **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(self.to_numpy(X), **get_kwargs(kwargs, csr_matrix.__init__))

    def append_column(self, X, name, column):
        name = self._check_column(X, name, str_only=True, as_list=False, convert=False)
        X[name] = column
        return X

    def ge(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X[column] >= value
        raise ValueError("Column must be specified for dict format")

    def gt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X[column] > value
        raise ValueError("Column must be specified for dict format")

    def le(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X[column] <= value
        raise ValueError("Column must be specified for dict format")

    def lt(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X[column] < value
        raise ValueError("Column must be specified for dict format")

    def eq(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X[column] == value
        raise ValueError("Column must be specified for dict format")

    def ne(self, X, value, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        if column is not None:
            return X[column] != value
        raise ValueError("Column must be specified for dict format")

    def add_column(self, X, index, name, column):
        name = self._check_column(X, name, str_only=True, as_list=False, convert=False)
        cols = list(X.keys())
        cols.insert(index, name)
        X[name] = column
        return {col: X[col] for col in cols}

    def rename_column(self, X, name, new_name):
        name = self._check_column(X, name, str_only=True, as_list=False, convert=False)
        new_name = self._check_column(
            X, new_name, str_only=True, as_list=False, convert=False
        )
        X[new_name] = X.pop(name)
        return X

    def rename_columns(self, X, mapping):
        for name, new_name in mapping.items():
            X = self.rename_column(X, name, new_name)
        return X

    def select_columns(self, X, columns=None, **kwargs):
        if columns is None:
            return X
        if not columns:
            return dict()

        columns = self._check_column(
            X, columns, str_only=True, as_list=True, convert=True
        )
        return {k: X[k] for k in columns}

    def select_column(self, X, column, **kwargs):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=True
        )
        return X[column]

    def set_column(self, X, column, value):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        X[column] = value
        return X

    def set_column_names(self, X, names, new_fingerprint=None):
        names = self._check_column(X, names, str_only=True, as_list=True, convert=False)
        return {k: v for k, v in zip(names, X.values())}

    def to_frame(self, X, name=None):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=True)
        if name is not None:
            return {name: next(iter(X.values()))}
        return X

    def select_rows(self, X, indices, **kwargs):
        format = get_data_format(next(iter(X.values())))
        sel_rows = _FORMAT_TO_CONVERTER[format].select_rows
        return {
            k: sel_rows(v, indices, new_fingerprint=kwargs.get("new_fingerprint", None))
            for k, v in X.items()
        }

    def select_row(self, X, index):
        format = get_data_format(next(iter(X.values())))
        sel_row = _FORMAT_TO_CONVERTER[format].select_row

        return {k: sel_row(v, index) for k, v in X.items()}

    def unique(self, X, column=None):
        arr = self.select_column(X, column)
        return _FORMAT_TO_CONVERTER[get_data_format(arr)].unique(arr)

    def replace(self, X, column=None, mapping={}):
        arr = self.select_column(X, column)
        return _FORMAT_TO_CONVERTER[get_data_format(arr)].replace(arr)

    def nunique(self, X, column=None):
        return len(self.unique(X, column))

    def argmax(self, X, axis=0):
        format = get_data_format(next(iter(X.values())))
        tbl = _FORMAT_TO_CONVERTER[format].concat(list(X.values()), axis=0)
        return _FORMAT_TO_CONVERTER[format].argmax(tbl, axis=axis)

    def get_column_names(self, X, generate_cols=False):
        return list(X.keys())

    def get_shape(self, X):
        return len(next(iter(X.values()))), len(X)

    def iter(self, X, batch_size, drop_last_batch=False):
        total_length = self.get_shape(X)[0]
        # Calculate the number of batches
        num_batches = total_length // batch_size
        if not drop_last_batch or total_length % batch_size == 0:
            num_batches += 1
        format = get_data_format(next(iter(X.values())))
        sel_rows = _FORMAT_TO_CONVERTER[format].select_rows
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = start_idx + batch_size
            yield {k: sel_rows(v, slice(start_idx, end_idx)) for k, v in X.items()}

    def concat(self, tables, axis=0, **kwargs):
        format = get_data_format(next(iter(tables[0].values())))
        concat = _FORMAT_TO_CONVERTER[format].concat
        if axis == 0:
            return {
                k: concat([table[k] for table in tables], axis=None) for k in tables[0]
            }
        out = tables[0]
        for t in tables[1:]:
            out.update(t)
        return out

    def is_array_like(self, X, min_row=10, min_col=10):
        if isinstance(next(iter(X.values())), (list, tuple, np.ndarray)):
            valid = True
            length = len(next(iter(X.values())))
            if length <= min_row or len(X) <= min_col:
                return False
            for v in X.values():
                if len(v) != length:
                    valid = False
                    break
            return valid

    def get_dtypes(self, X, columns=None):
        format = get_data_format(next(iter(X.values())))
        tbl = _FORMAT_TO_CONVERTER[format].concat(list(X.values()), axis=1)
        return _FORMAT_TO_CONVERTER[format].get_dtypes(tbl, columns=columns)

    def is_categorical(self, X, column, threshold=None):
        arr = self.select_column(X, column)
        format = get_data_format(arr)
        return _FORMAT_TO_CONVERTER[format].is_categorical(arr, threshold=threshold)
