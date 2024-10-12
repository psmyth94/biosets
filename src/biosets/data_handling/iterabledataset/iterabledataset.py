from datasets import IterableDataset

from biosets.utils.import_util import requires_backends
from biosets.utils.inspect import get_kwargs

from ..dataset import DatasetConverter


class IterableDatasetConverter(DatasetConverter):
    dtype = "IterableDataset"

    streaming = False

    def to_numpy(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_numpy_in_memory(X, **kwargs)
        else:
            return self._to_numpy(X, **kwargs)

    def _to_numpy_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_numpy(self.to_dataset(X, **kwargs), **kwargs)

    def _to_numpy(self, X: IterableDataset, **kwargs):
        return X.with_format("numpy")

    def to_pandas(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_pandas_in_memory(X, **kwargs)
        else:
            return self._to_pandas(X, **kwargs)

    def _to_pandas_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_pandas(self.to_dataset(X, **kwargs), **kwargs)

    def _to_pandas(self, X: IterableDataset, **kwargs):
        return X.with_format("pandas")

    def to_polars(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_polars_in_memory(X, **kwargs)
        else:
            return self._to_polars(X, **kwargs)

    def _to_polars_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_polars(self.to_dataset(X, **kwargs), **kwargs)

    def _to_polars(self, X: IterableDataset, **kwargs):
        return X.with_format("polars")

    def to_file(self, X: IterableDataset, path, **kwargs):
        if self.streaming:
            return self._to_file_in_memory(X, path, **kwargs)
        else:
            return self._to_file(X, path, **kwargs)

    def _to_file_in_memory(self, X: IterableDataset, path, **kwargs):
        return super().to_file(self.to_dataset(X, **kwargs), path, **kwargs)

    def _to_file(self, X: IterableDataset, path, **kwargs):
        return NotImplemented

    def to_arrow(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_arrow_in_memory(X, **kwargs)
        else:
            return self._to_arrow(X, **kwargs)

    def _to_arrow_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_arrow(self.to_dataset(X, **kwargs), **kwargs)

    def _to_arrow(self, X: IterableDataset, **kwargs):
        return X.with_format("arrow")

    def to_dicts(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_dicts_in_memory(X, **kwargs)
        else:
            return self._to_dicts(X, **kwargs)

    def _to_dicts_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_dicts(self.to_dataset(X, **kwargs), **kwargs)

    def _to_dicts(self, X: IterableDataset, **kwargs):
        return NotImplemented

    def to_dict(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_dict_in_memory(X, **kwargs)
        else:
            return self._to_dict(X, **kwargs)

    def _to_dict_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_dict(self.to_dataset(X, **kwargs), **kwargs)

    def _to_dict(self, X: IterableDataset, **kwargs):
        return NotImplemented

    def to_list(self, X: IterableDataset, **kwargs):
        if self.streaming:
            return self._to_list_in_memory(X, **kwargs)
        else:
            return self._to_list(X, **kwargs)

    def _to_list_in_memory(self, X: IterableDataset, **kwargs):
        return super().to_list(self.to_dataset(X, **kwargs), **kwargs)

    def _to_list(self, X: IterableDataset, **kwargs):
        return NotImplemented

    def to_dataset(self, X: IterableDataset, **kwargs):
        from biosets import Dataset

        def gen(**gen_kwargs):
            for row in X:
                yield row

        return Dataset.from_generator(gen, **get_kwargs(kwargs, Dataset.from_generator))

    def to_iterabledataset(self, X: IterableDataset, **kwargs):
        return X

    def to_dask(self, X: IterableDataset, **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(
            self.to_pandas(X, **kwargs), **get_kwargs(kwargs, dd.from_pandas)
        )

    def to_ray(self, X, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_huggingface(
            X, **get_kwargs(kwargs, ray.data.from_huggingface)
        )
