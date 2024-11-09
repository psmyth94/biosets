import copy
import inspect
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Union

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
from biocore.utils.import_util import is_polars_available
from biocore.utils.inspect import get_kwargs
from datasets.features.features import require_storage_cast
from datasets.packaged_modules.csv.csv import Csv as _Csv
from datasets.packaged_modules.csv.csv import CsvConfig as HfCsvConfig

from biosets.utils import logging

if TYPE_CHECKING:
    from polars.type_aliases import CsvEncoding

logger = logging.get_logger(__name__)


@dataclass
class CsvConfig(datasets.BuilderConfig):
    batch_size: int = 50_000

    features: Optional[Optional[datasets.Features]] = None

    # read_csv_batched kwargs
    has_header: bool = True
    columns: Optional[Union[Sequence[int], Sequence[str]]] = None
    new_columns: Optional[Sequence[str]] = None
    separator: str = ","
    comment_prefix: Optional[str] = None
    quote_char: str = '"'
    skip_rows: int = 0
    null_values: Optional[Union[str, Sequence[str], Dict[str, str]]] = None
    missing_utf8_is_empty_string: bool = False
    ignore_errors: bool = False
    try_parse_dates: bool = False
    n_threads: Optional[int] = None
    infer_schema_length: int = 100
    n_rows: Optional[int] = None
    encoding: Union["CsvEncoding", str] = "utf8"
    low_memory: bool = False
    rechunk: bool = True
    skip_rows_after_header: int = 0
    row_count_name: Optional[str] = None
    row_count_offset: int = 0
    sample_size: int = 1024
    eol_char: str = "\n"
    raise_if_empty: bool = True

    HF_CSV_CONFIG: Optional[HfCsvConfig] = None
    hf_kwargs: Optional[Mapping[str, str]] = None
    use_polars: bool = True

    def __post_init__(self):
        if self.hf_kwargs is None:
            self.hf_kwargs = {}
        self.HF_CSV_CONFIG = HfCsvConfig(**self.hf_kwargs)

    @property
    def pl_scan_csv_kwargs(self):
        return {
            "has_header": self.has_header,
            "columns": self.columns,
            "new_columns": self.new_columns,
            "separator": self.separator,
            "comment_prefix": self.comment_prefix,
            "quote_char": self.quote_char,
            "skip_rows": self.skip_rows,
            "null_values": self.null_values,
            "missing_utf8_is_empty_string": self.missing_utf8_is_empty_string,
            "ignore_errors": self.ignore_errors,
            "try_parse_dates": self.try_parse_dates,
            "n_threads": self.n_threads,
            "infer_schema_length": self.infer_schema_length,
            "n_rows": self.n_rows,
            "encoding": self.encoding,
            "low_memory": self.low_memory,
            "rechunk": self.rechunk,
            "skip_rows_after_header": self.skip_rows_after_header,
            "row_index_name": self.row_count_name,
            "row_index_offset": self.row_count_offset,
            "sample_size": self.sample_size,
            "eol_char": self.eol_char,
            "raise_if_empty": self.raise_if_empty,
        }


class Csv(_Csv):
    BUILDER_CONFIG_CLASS = CsvConfig
    config: CsvConfig

    def __init__(self, *args, **kwargs):
        if not kwargs.get("hf_kwargs", None):
            biosets_args = [
                p.name
                for p in inspect.signature(CsvConfig.__init__).parameters.values()
                if p != p.VAR_KEYWORD
            ]
            kwargs["hf_kwargs"] = {}
            hf_kwargs = get_kwargs(copy.deepcopy(kwargs), HfCsvConfig.__init__)
            use_polars = kwargs.get("use_polars", True)
            for k, v in hf_kwargs.items():
                kwargs["hf_kwargs"][k] = v
                if k not in biosets_args:
                    kwargs.pop(k)
                    use_polars = False
            kwargs["use_polars"] = use_polars
        super().__init__(*args, **kwargs)

    def _generate_tables_pandas(self, files):
        schema = self.config.features.arrow_schema if self.config.features else None
        # dtype allows reading an int column as str
        dtype = (
            {
                name: dtype.to_pandas_dtype()
                if not require_storage_cast(feature)
                else object
                for name, dtype, feature in zip(
                    schema.names, schema.types, self.config.features.values()
                )
            }
            if schema is not None
            else None
        )
        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            csv_file_reader = pd.read_csv(
                file, iterator=True, dtype=dtype, **self.config.pd_read_csv_kwargs
            )
            try:
                for batch_idx, df in enumerate(csv_file_reader):
                    pa_table = pa.Table.from_pandas(df, preserve_index=False)
                    # Uncomment for debugging (will print the Arrow table size and elements)
                    # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                    # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                    yield (file_idx, batch_idx), self._cast_table(pa_table)
            except ValueError as e:
                logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                raise

    def _generate_tables(self, files):
        if not is_polars_available() or not getattr(self.config, "use_polars", None):
            if hasattr(self.config, "HF_CSV_CONFIG"):
                self.config = self.config.HF_CSV_CONFIG
            self.BUILDER_CONFIG_CLASS = HfCsvConfig
            for table in super()._generate_tables(files):
                yield table
        else:
            import polars as pl

            try:
                schema = (
                    self.config.features.arrow_schema if self.config.features else None
                )

                def parse_dtype(dtype):
                    if isinstance(dtype, type) and issubclass(dtype, np.generic):
                        dt = dtype()
                        if hasattr(dt, "item"):
                            return dtype().item().__class__
                        return object
                    return dtype

                # dtype allows reading an int column as str

                dtype = None
                if schema is not None:
                    schema_overrides = {}
                    for name, dtype, feature in zip(
                        schema.names, schema.types, self.config.features.values()
                    ):
                        if require_storage_cast(feature):
                            schema_overrides[name] = object
                        if dtype == pa.string() or dtype == pa.large_string():
                            schema_overrides[name] = str
                        else:
                            schema_overrides[name] = parse_dtype(
                                dtype.to_pandas_dtype()
                            )
                for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
                    csv_file_reader = pl.read_csv_batched(
                        file,
                        schema_overrides=schema_overrides,
                        **self.config.pl_scan_csv_kwargs,
                    )
                    try:
                        while True:
                            batches = csv_file_reader.next_batches(
                                self.config.batch_size
                            )
                            if not batches:
                                break
                            for batch_idx, df in enumerate(batches):
                                # Convert Polars DataFrame to Arrow Table
                                pa_table = df.to_arrow()
                                # Uncomment for debugging
                                # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                                # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                                yield (file_idx, batch_idx), self._cast_table(pa_table)

                    except ValueError as e:
                        logger.error(f"Error while reading file {file}: {e}")
                        raise e
            except Exception:
                self.config = self.config.HF_CSV_CONFIG
                self.BUILDER_CONFIG_CLASS = HfCsvConfig
                for table in super()._generate_tables(files):
                    yield table
