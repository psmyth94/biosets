import itertools
from dataclasses import dataclass
from typing import Optional

import datasets
import datasets.config
import pandas as pd
import pyarrow as pa
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
from datasets.utils.file_utils import xnumpy_load

from biosets.utils import logging, requires_backends

logger = logging.get_logger(__name__)


@dataclass
class SparseReaderConfig(datasets.BuilderConfig):
    batch_size: int = 50_000
    mmap_mode: Optional[str] = "r"
    allow_pickle: bool = False
    fix_imports: bool = True
    encoding: str = "ASCII"

    features: Optional[Optional[datasets.Features]] = None


class SparseReader(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = SparseReaderConfig
    config: SparseReaderConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN, gen_kwargs={"files": files}
                )
            ]
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files = [dl_manager.iter_files(file) for file in files]
            splits.append(
                datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})
            )
        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.config.features is not None:
            schema = self.config.features.arrow_schema
            if all(
                not require_storage_cast(feature)
                for feature in self.config.features.values()
            ):
                # cheaper cast
                pa_table = pa.Table.from_arrays(
                    [pa_table[field.name] for field in schema], schema=schema
                )
            else:
                # more expensive cast; allows str <-> int/float or str to Audio for example
                pa_table = table_cast(pa_table, schema)
        return pa_table

    def _generate_tables(self, files):
        requires_backends(self._cast_table, "scipy")
        import scipy.sparse as sp

        sp.load_npz

        for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
            with xnumpy_load(
                file,
                mmap_mode=self.config.mmap_mode,
                allow_pickle=self.config.allow_pickle,
                fix_imports=self.config.fix_imports,
                encoding=self.config.encoding,
            ) as loaded:
                sparse_format = loaded.get("format")
                if sparse_format is None:
                    raise ValueError(
                        f"The file {file} does not contain "
                        f"a sparse array or matrix."
                    )
                sparse_format = sparse_format.item()

                if not isinstance(sparse_format, str):
                    # Play safe with Python 2 vs 3 backward compatibility;
                    # files saved with SciPy < 1.0.0 may contain unicode or bytes.
                    sparse_format = sparse_format.decode("ascii")

                if loaded.get("_is_array"):
                    sparse_type = sparse_format + "_array"
                else:
                    sparse_type = sparse_format + "_matrix"

                try:
                    cls = getattr(sp, f"{sparse_type}")
                except AttributeError as e:
                    raise ValueError(f'Unknown format "{sparse_type}"') from e

                shape = loaded["shape"]
                for row_start in range(0, shape[0], self.config.batch_size):
                    row_end = min(row_start + self.config.batch_size, shape[0])
                    if sparse_format in ("csc", "csr", "bsr"):
                        indptr = loaded["indptr"]
                        lower_bound = indptr[row_start]
                        upper_bound = indptr[row_end]
                        batch_data = loaded["data"][lower_bound:upper_bound]
                        batch_indices = loaded["indices"][lower_bound:upper_bound]
                        batch_indptr = indptr[row_start : row_end + 1] - lower_bound
                        mat = cls(
                            (batch_data, batch_indices, batch_indptr),
                            shape=(row_end - row_start, shape[1]),
                        ).toarray()
                    elif sparse_format == "dia":
                        mat = cls(
                            (loaded["data"], loaded["offsets"]),
                            shape=(row_end - row_start, shape[1]),
                        ).toarray()
                    elif sparse_format == "coo":
                        mat = cls(
                            (loaded["data"], (loaded["row"], loaded["col"])),
                            shape=(row_end - row_start, shape[1]),
                        ).toarray()
                    else:
                        raise NotImplementedError(
                            f"Load is not implemented for "
                            f"sparse matrix of format {sparse_format}."
                        )
                    yield (
                        (file_idx, 0),
                        self._cast_table(
                            pa.Table.from_pandas(
                                pd.DataFrame(mat), preserve_index=False
                            )
                        ),
                    )
