from typing import Any, Dict

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

from ..dict import DictConverter


class RowDictConverter(DictConverter):
    dtype = "Dict[str, Any]"

    def to_list(self, X: Dict[str, Any], **kwargs):
        return [X]

    def to_dict(self, X: Dict[str, Any], **kwargs):
        return X

    def to_dicts(self, X: Dict[str, Any], **kwargs):
        return [X]

    def to_numpy(self, X: Dict[str, Any], **kwargs):
        return np.array([X], **np_array_kwargs(kwargs))

    def to_pandas(self, X: Dict[str, Any], **kwargs):
        return pd.DataFrame([X], **get_kwargs(kwargs, pd.DataFrame.__init__))

    def to_series(self, X: Dict[str, Any], **kwargs):
        return pd.Series(X, **get_kwargs(kwargs, pd.Series.__init__))

    def to_polars(self, X: Dict[str, Any], **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.DataFrame([X], **get_kwargs(kwargs, pl.DataFrame.__init__))

    def to_arrow(self, X: Dict[str, Any], **kwargs):
        return pa.Table.from_pandas(
            self.to_pandas(X, **kwargs), **pa_table_from_pandas_kwargs(kwargs)
        )

    def to_dataset(self, X: Dict[str, Any], **kwargs):
        from biosets import Dataset

        return Dataset.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, Dataset.from_pandas)
        )

    def to_iterabledataset(self, X: Dict[str, Any], **kwargs):
        def gen(**gen_kwargs):
            yield X

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_dask(self, X: Dict[str, Any], **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, dd.from_pandas)
        )

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_items(X, **get_kwargs(kwargs, ray.data.from_items))
