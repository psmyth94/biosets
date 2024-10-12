import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from biosets import Dataset, config
from biosets.data_handling import (
    DataHandler,
)
from biosets.data_handling.data_handling import _FORMAT_TO_CONVERTER
from biosets.utils.import_util import is_polars_available

from datasets import IterableDataset

pl = None
if is_polars_available():
    import polars as pl_

    pl = pl_


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_numpy_converter(format):
    try:
        assert DataHandler.to_format(np.array([1, 2, 3]), format) is not None
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_list_converter(format):
    try:
        assert DataHandler.to_format([1, 2, 3], format) is not None
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_dict_converter(format):
    try:
        assert DataHandler.to_format({"a": 1}, format) is not None
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_pandas_converter(format):
    try:
        assert DataHandler.to_format(pd.DataFrame({"a": [1]}), format) is not None
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_polars_converter(format):
    try:
        assert DataHandler.to_format(pl.DataFrame({"a": [1]}), format) is not None
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_arrow_converter(format):
    try:
        assert (
            DataHandler.to_format(pa.Table.from_pydict({"a": [1]}), format) is not None
        )
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_dataset_converter(format):
    try:
        kwargs = {}
        if format == "io":
            kwargs["path"] = Path(tempfile.mkdtemp()) / "test_dataset_converter.csv"

        assert (
            DataHandler.to_format(Dataset.from_dict({"a": [1]}), format, **kwargs)
            is not None
        )
    except NotImplementedError:
        pass


@pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
def test_iterable_dataset_converter(format):
    def gen():
        data = [{"a": 1}]
        for d in data:
            yield d

    try:
        kwargs = {}
        if format == "io":
            kwargs["path"] = Path(tempfile.mkdtemp()) / "test_dataset_converter.csv"
        assert (
            DataHandler.to_format(IterableDataset.from_generator(gen), format, **kwargs)
            is not None
        )
    except NotImplementedError:
        pass


# @pytest.mark.parametrize("format", _FORMAT_TO_CONVERTER.keys())
# def test_io_converter(format):
#     data = {"a": [1], "b": [2]}
#     base_dir = Path(os.path.abspath("./tests/data/test_io_converter"))
#     base_dir.mkdir(parents=True, exist_ok=True)
#
#     csv = "a,b\n1,2\n"
#     csv_path = base_dir / "test.csv"
#     with open(csv_path, "w") as f:
#         f.write(csv)
#
#     json = '{"a": 1, "b": 2}'
#     json_path = base_dir / "test.json"
#     with open(json_path, "w") as f:
#         f.write(json)
#
#     tbl = pa.Table.from_pydict(data)
#     parquet_path = base_dir / "test.parquet"
#     pq.write_table(tbl, parquet_path)
#
#     arrow_path = base_dir / "test.arrow"
#     with open(arrow_path, "wb") as f:
#         writer = pa.RecordBatchStreamWriter(f, tbl.schema)
#         writer.write_table(tbl)
#
#     try:
#         assert DataHandler.to_format(csv_path.as_posix(), format) is not None
#     except NotImplementedError:
#         pass
#
#     try:
#         assert DataHandler.to_format(json_path, format) is not None
#     except NotImplementedError:
#         pass
#
#     try:
#         assert DataHandler.to_format(parquet_path, format) is not None
#     except NotImplementedError:
#         pass
#
#     try:
#         assert DataHandler.to_format(arrow_path, format) is not None
#     except NotImplementedError:
#         pass
