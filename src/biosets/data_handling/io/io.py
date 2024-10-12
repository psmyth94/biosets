from pathlib import Path

import pandas as pd
import pyarrow.csv as pac
import pyarrow.feather as pf
import pyarrow.json as paj
import pyarrow.parquet as pq
from datasets import DatasetInfo, Features, IterableDataset
from datasets.table import InMemoryTable

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import get_kwargs

from ..base import BaseDataConverter


class IOConverter(BaseDataConverter):
    dtype = "io"

    def get_file_extension(self, path, **kwargs):
        return Path(path).suffix

    def validate_file(self, ext, valid_ext, **kwargs):
        if ext not in valid_ext:
            raise ValueError(
                f"Invalid file extension {ext}. Must be one of {valid_ext}"
            )

    def to_arrow(self, path, **kwargs):
        valid_ext = [".arrow", ".feather", ".parquet", ".json", ".csv"]
        ext = self.get_file_extension(path, **kwargs)
        self.validate_file(ext, valid_ext, **kwargs)
        if ext == ".arrow":
            return InMemoryTable.from_file(path).table
        elif ext == ".feather":
            return pf.read_table(path)
        elif ext == ".parquet":
            return pq.read_table(path)
        elif ext == ".json":
            return paj.read_json(path)
        elif ext == ".csv":
            return pac.read_csv(path)

    def to_pandas(self, path, **kwargs):
        return self.to_arrow(path, **kwargs).to_pandas()

    def to_polars(self, path, **kwargs):
        requires_backends(self.to_polars, "polars")
        import polars as pl

        return pl.from_arrow(self.to_arrow(path, **kwargs))

    def to_numpy(self, path, **kwargs):
        return self.to_pandas(path, **kwargs).values

    def to_list(self, path, **kwargs):
        return self.to_numpy(path, **kwargs).tolist()

    def to_dict(self, path, **kwargs):
        return self.to_pandas(path, **kwargs).to_dict(
            "list", **get_kwargs(kwargs, pd.DataFrame.to_dict)
        )

    def to_dicts(self, path, **kwargs):
        return self.to_pandas(path, **kwargs).to_dict(
            "records", **get_kwargs(kwargs, pd.DataFrame.to_dict)
        )

    def to_dataset(self, path, **kwargs):
        from biosets import Dataset

        tbl = self.to_arrow(path, **kwargs)
        features = Features.from_arrow_schema(tbl.schema)
        info = DatasetInfo(features=features)
        return Dataset(tbl, info=info)

    def to_iterabledataset(self, path, **kwargs):
        if isinstance(path, str):
            path = [path]

        def gen(**gen_kwargs):
            for f in path:
                yield self.to_dict(f, **gen_kwargs)

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_file(self, path, **kwargs):
        return path

    def to_dask(self, path, **kwargs):
        valid_ext = [".arrow", ".parquet", ".json", ".csv", ".hdf5", "h5"]
        ext = self.get_file_extension(path, **kwargs)
        self.validate_file(ext, valid_ext, **kwargs)
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        if ext == ".arrow":
            return dd.from_pandas(
                self.to_pandas(path, **kwargs), **get_kwargs(kwargs, dd.from_pandas)
            )
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            return dd.read_hdf(path, **get_kwargs(kwargs, dd.read_hdf))
        elif ext == ".parquet":
            return dd.read_parquet(path, **get_kwargs(kwargs, dd.read_parquet))
        elif ext == ".json":
            return dd.read_json(path, **get_kwargs(kwargs, dd.read_json))
        elif ext == ".csv":
            return dd.read_csv(path, **get_kwargs(kwargs, dd.read_csv))
