import copy
from pathlib import Path
from typing import Union

import pyarrow as pa
from datasets import Dataset
from datasets.features.features import Value, generate_from_arrow_type
from datasets.table import Table

from biosets.utils.import_util import is_polars_available, requires_backends
from biosets.utils.inspect import get_kwargs, pa_table_from_pandas_kwargs

from ..arrow import ArrowConverter


class DatasetConverter(ArrowConverter):
    dtype = "Dataset"
    supports_named_columns = True

    def to_numpy(self, X: Dataset, **kwargs):
        return super().to_numpy(self.to_arrow(X), **kwargs)

    def to_pandas(self, X: Dataset, **kwargs):
        return X.to_pandas(**get_kwargs(kwargs, X.to_pandas))

    def to_polars(self, X: Dataset, **kwargs):
        return X.to_polars(**get_kwargs(kwargs, X.to_polars))

    def to_arrow(self, X: Dataset, **kwargs):
        if X._indices is not None:
            if is_polars_available():
                return self.to_polars(X, **kwargs).to_arrow()
            else:
                return pa.Table.from_pandas(
                    self.to_pandas(X, **kwargs), **pa_table_from_pandas_kwargs(kwargs)
                )
        return X.data.table

    def to_list(self, X: Dataset, **kwargs):
        return super().to_list(self.to_arrow(X), **kwargs)

    def to_dict(self, X: Dataset, **kwargs):
        return X.to_dict(**get_kwargs(kwargs, X.to_dict))

    def to_dicts(self, X: Dataset, **kwargs):
        return super().to_dicts(self.to_arrow(X), **kwargs)

    def to_dataset(self, X: Dataset, **kwargs):
        return X

    def to_iterabledataset(self, X: Dataset, **kwargs):
        return X.to_iterable_dataset(**get_kwargs(kwargs, X.to_iterable_dataset))

    def to_file(self, X: Dataset, path: Union[Path, str], **kwargs):
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
            return X.save_to_disk(path, **get_kwargs(kwargs, X.save_to_disk))
        elif ext == ".parquet":
            path.parent.mkdir(parents=True, exist_ok=True)
            return X.to_parquet(path, **get_kwargs(kwargs, X.to_parquet))
        elif ext == ".json":
            path.parent.mkdir(parents=True, exist_ok=True)
            return X.to_json(path, **get_kwargs(kwargs, X.to_json))
        elif ext == ".csv":
            path.parent.mkdir(parents=True, exist_ok=True)
            return X.to_csv(path, **get_kwargs(kwargs, X.to_csv))

    def to_dask(self, X: Dataset, **kwargs):
        requires_backends(self.to_dask, "dask")
        import dask.dataframe as dd

        return dd.from_pandas(X.to_pandas(), **get_kwargs(kwargs, dd.from_pandas))

    def to_ray(self, X: Dataset, **kwargs):
        requires_backends(self.to_ray, "ray")
        import ray.data

        return ray.data.from_huggingface(
            X, **get_kwargs(kwargs, ray.data.from_huggingface)
        )

    def to_csr(self, X: Dataset, **kwargs):
        requires_backends(self.to_csr, "scipy")
        from scipy.sparse import csr_matrix

        return csr_matrix(self.to_numpy(X), **get_kwargs(kwargs, csr_matrix.__init__))

    def append_column(self, X: Dataset, name, column):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        return X.add_column(name, column=column, new_fingerprint=X._fingerprint)

    def add_column(self, X: Dataset, index, name, column):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        return self.append_column(X, name, column)

    def select_rows(self, X: Dataset, indices, new_fingerprint=None):
        if isinstance(indices, Table):
            X._indices = indices
            return X
        return X.select(
            indices=indices,
            new_fingerprint=new_fingerprint or X._fingerprint,
        )

    def select_row(self, X: Dataset, index: Dataset):
        return X.select(index)

    def rename_column(self, X: Dataset, name, new_name):
        name = self._check_column(X, name, str_only=False, as_list=False, convert=False)
        new_name = self._check_column(
            X, new_name, str_only=False, as_list=False, convert=False
        )
        return X.rename_column(name, new_name)

    def rename_columns(self, X: Dataset, mapping):
        mapping = {k: mapping.get(k, k) for k in X.column_names}
        return X.rename_columns(mapping)

    def select_columns(self, X: Dataset, columns=None, feature_type=None, **kwargs):
        if columns:
            columns = self._check_column(
                X, columns, str_only=True, as_list=True, convert=True
            )
            new_fingerprint = kwargs.pop("new_fingerprint", None)
            keep_old_fingerprint = kwargs.pop("keep_old_fingerprint", True)
            if new_fingerprint is None and keep_old_fingerprint:
                new_fingerprint = X._fingerprint
            return X.select_columns(columns, new_fingerprint=new_fingerprint, **kwargs)
        elif feature_type:
            return X.select_columns(
                [k for k, v in X._info.features.items() if isinstance(v, feature_type)]
            )
        else:
            return X

    def select_column(self, X: Dataset, column, **kwargs):
        return self.select_columns(X, column, **kwargs)

    def unique(self, X: Dataset, column=None):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=True
        )
        if column is None:
            if X.num_columns == 1:
                column = X.column_names[0]
            else:
                raise ValueError(
                    "Column must be specified for Dataset with multiple columns"
                )
        return X.unique(column)

    def replace(self, X: Dataset, column=None, mapping={}):
        new_col = pa.array(
            [mapping.get(v.as_py(), v.as_py()) for v in self.to_arrow(X).column(column)]
        )
        return self.set_column(X, column, new_col)

    def nunique(self, X: Dataset, column=None):
        return len(self.unique(X, column))

    def argmax(self, X: Dataset, axis=0):
        return super().argmax(self.to_arrow(X), axis)

    def ge(self, X, value, column=None):
        return super().ge(self.to_arrow(X), value, column=column)

    def gt(self, X, value, column=None):
        return super().gt(self.to_arrow(X), value, column=column)

    def le(self, X, value, column=None):
        return super().le(self.to_arrow(X), value, column=column)

    def lt(self, X, value, column=None):
        return super().lt(self.to_arrow(X), value, column=column)

    def eq(self, X, value, column=None):
        return super().eq(self.to_arrow(X), value, column=column)

    def ne(self, X, value, column=None):
        return super().ne(self.to_arrow(X), value, column=column)

    def set_column(self, X: Dataset, column, value):
        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=False
        )
        col_pos = X.column_names.index(column)
        out = copy.deepcopy(X)
        out._data.table = out._data.table.set_column(col_pos, column, value)
        feat = generate_from_arrow_type(out._data.table.column(col_pos).type)
        out._info.features[column] = feat
        return out

    def set_column_names(self, X: Dataset, names, new_fingerprint=None):
        names = self._check_column(X, names, str_only=True, as_list=True, convert=False)
        return X.rename_columns(
            {old_name: new_name for old_name, new_name in zip(X.column_names, names)},
            new_fingerprint=new_fingerprint,
        )

    def get_column_names(self, X: Dataset, generate_cols=False):
        return X.column_names

    def get_column_names_by_feature_type(self, X: Dataset, feature_type):
        if feature_type:
            return [
                k for k, v in X._info.features.items() if isinstance(v, feature_type)
            ]
        return X.column_names

    def get_shape(self, X: Dataset):
        return X.shape

    def to_frame(self, X: Dataset, name=None):
        return X

    def iter(self, X: Dataset, batch_size, drop_last_batch=False):
        return X.iter(batch_size, drop_last_batch=drop_last_batch)

    def concat(self, tables, axis=0, **kwargs):
        from biosets import concatenate_datasets

        return concatenate_datasets(tables, axis=axis)

    def is_array_like(self, X: Dataset, min_row=10, min_col=10):
        return True

    def get_dtypes(self, X: Dataset, columns=None):
        columns = self._check_column(X, columns)
        if columns is None:
            return {k: str(v.pa_type) for k, v in X._info.features.items()}
        return {col: X._info.features[col].dtype for col in columns}

    def is_categorical(self, X: Dataset, column, threshold=None):
        from biosets.features import CATEGORICAL_FEATURES

        column = self._check_column(
            X, column, str_only=True, as_list=False, convert=True
        )
        feature = X._info.features[column]
        if isinstance(feature, CATEGORICAL_FEATURES):
            return True
        if isinstance(feature, Value):
            if feature.dtype == "string":
                return True
            if threshold and "int" in feature.dtype:
                n_unique = self.nunique(X, column)
                return n_unique / self.get_shape(X)[0] < threshold
            return False
        return False
