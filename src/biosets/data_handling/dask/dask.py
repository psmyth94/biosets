import sys
from typing import TYPE_CHECKING

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
    import dask.dataframe as dd


class DaskConverter(BaseDataConverter):
    dtype = "dd.DataFrame"

    def to_numpy(self, X: "dd.DataFrame", **kwargs):
        if "dask" in sys.modules:
            import dask.array as da

        if isinstance(X, da.Array):
            return X.compute()
        return X.compute().values

    def to_pandas(self, X: "dd.DataFrame", **kwargs):
        if "dask" in sys.modules:
            import dask.array as da

        if isinstance(X, da.Array):
            return pd.DataFrame(X.compute())
        return X.compute()

    def to_series(self, X: "dd.DataFrame", **kwargs):
        if len(X.columns) == 1:
            return X.iloc[:, 0].compute()
        if len(X.index) == 1:
            return X.compute().iloc[0, :]
        raise ValueError("Cannot convert multi-dimensional dataframe to series")

    def to_polars(self, X: "dd.DataFrame", **kwargs):
        if "polars" in sys.modules:
            import polars as pl

        return pl.from_pandas(X.compute(), **get_kwargs(kwargs, pl.from_pandas))

    def to_list(self, X: "dd.DataFrame", **kwargs):
        return X.compute().values.tolist()

    def to_dict(self, X: "dd.DataFrame", **kwargs):
        return X.compute().to_dict("list", **get_kwargs(kwargs, pd.DataFrame.to_dict))

    def to_dicts(self, X: "dd.DataFrame", **kwargs):
        return X.compute().to_dict(
            "records", **get_kwargs(kwargs, pd.DataFrame.to_dict)
        )

    def to_arrow(self, X: "dd.DataFrame", **kwargs):
        return pa.Table.from_pandas(X.compute(), **pa_table_from_pandas_kwargs(kwargs))

    def to_dataset(self, X: "dd.DataFrame", **kwargs):
        from biosets import Dataset

        return Dataset.from_pandas(
            X.compute(), **get_kwargs(kwargs, Dataset.from_pandas)
        )

    def to_iterabledataset(self, X: "dd.DataFrame", **kwargs):
        def gen(**gen_kwargs):
            for _, row in X.compute().iterrows():
                yield row.to_dict()

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_dask(X, **get_kwargs(kwargs, ray.data.from_dask))
