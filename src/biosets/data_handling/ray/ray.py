from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa
from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import get_kwargs

from ..arrow import ArrowConverter

if TYPE_CHECKING:
    import ray.data.dataset


class RayConverter(ArrowConverter):
    dtype = "Dataset"

    def to_numpy(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        return np.concatenate(
            [super().to_numpy(x, **kwargs) for x in X.to_arrow_refs()]
        )

    def to_pandas(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_pandas, "ray")
        import ray.data.dataset

        return X.to_pandas(**get_kwargs(kwargs, ray.data.dataset.Dataset.to_pandas))

    def to_polars(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_polars, "ray")
        requires_backends(self.to_polars, "polars")
        import polars as pl
        import ray.data.dataset

        return pl.from_pandas(
            X.to_pandas(**get_kwargs(kwargs, ray.data.dataset.Dataset.to_pandas)),
            **get_kwargs(kwargs, pl.from_pandas),
        )

    def to_arrow(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        return pa.concat_tables([x for x in X.to_arrow_refs()], promote="default")

    def to_list(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_list, "ray")
        import ray.data.dataset

        return X.to_pandas(
            **get_kwargs(kwargs, ray.data.dataset.Dataset.to_pandas)
        ).to_dict("list")

    def to_dict(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_dict, "ray")
        import ray.data.dataset

        return X.to_pandas(
            **get_kwargs(kwargs, ray.data.dataset.Dataset.to_pandas)
        ).to_dict()

    def to_dicts(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_dicts, "ray")
        import ray.data.dataset

        return X.to_pandas(
            **get_kwargs(kwargs, ray.data.dataset.Dataset.to_pandas)
        ).to_dict("records")

    def to_dataset(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        from biosets import Dataset

        return Dataset(self.to_arrow(X, kwargs), **get_kwargs(kwargs, Dataset.__init__))

    def to_iterabledataset(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_dicts, "ray")
        import ray.data.dataset

        def gen():
            for row in X.iter_rows(
                **get_kwargs(kwargs, ray.data.dataset.Dataset.iter_rows)
            ):
                yield row

        return IterableDataset.from_generator(
            gen, **get_kwargs(kwargs, IterableDataset.from_generator)
        )

    def to_file(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        path: Union[Path, str],
        **kwargs,
    ):
        path = Path(path)
        ext = Path(path).suffix
        valid_ext = [".arrow", ".parquet", ".json", ".csv"]
        if ext not in valid_ext:
            raise ValueError(
                f"Invalid file extension {ext}. Must be one of {valid_ext}"
            )

        if ext == ".arrow":
            path = path.replace(ext, "")
            path.mkdir(parents=True, exist_ok=True)
            X.write_parquet(path, **get_kwargs(kwargs))
        elif ext == ".parquet":
            path.parent.mkdir(parents=True, exist_ok=True)
            X.write_parquet(path, **get_kwargs(kwargs))
        elif ext == ".json":
            path.parent.mkdir(parents=True, exist_ok=True)
            X.write_json(path, **get_kwargs(kwargs))
        elif ext == ".csv":
            path.parent.mkdir(parents=True, exist_ok=True)
            X.write_csv(path, **get_kwargs(kwargs))

    def to_dask(
        self,
        X: Union["ray.data.dataset.MaterializedDataset", "ray.data.dataset.Dataset"],
        **kwargs,
    ):
        requires_backends(self.to_dask, "dask")
        requires_backends(self.to_dask, "ray")
        import dask.dataframe as dd
        import ray.data.dataset

        return dd.from_pandas(
            X.to_pandas(**get_kwargs(kwargs, ray.data.dataset.Dataset.to_pandas)),
            **get_kwargs(kwargs, dd.from_pandas),
        )

    def to_ray(self, X, **kwargs):
        return X
