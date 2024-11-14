import multiprocessing
import os
from functools import partial
from typing import List

import datasets.config
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from biocore.utils.import_util import is_polars_available
from biocore.utils.inspect import get_kwargs
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadManager
from datasets.features.features import Features
from datasets.utils.file_utils import is_relative_path, url_or_path_join, url_to_fs
from datasets.utils.logging import tqdm
from tqdm.contrib.concurrent import thread_map


class SchemaManager(DownloadManager):
    @classmethod
    def from_dl_manager(cls, dl_manager: DownloadManager) -> "SchemaManager":
        kwargs = dl_manager.__dict__.copy()
        init_kwargs = get_kwargs(kwargs, dl_manager.__init__)
        cls_instance = cls(**init_kwargs)
        for k, v in kwargs.items():
            if k not in init_kwargs:
                setattr(cls_instance, k, v)
        return cls_instance

    def _download_batched(
        self,
        url_or_filenames: List[str],
        download_config: DownloadConfig,
    ) -> List[str]:
        """
        Adapted from datasets.download.download_manager.DownloadManager._download_batched
        with the addition of schema gathering.
        """
        if len(url_or_filenames) >= 16:
            download_config = download_config.copy()
            download_config.disable_tqdm = True
            download_func = partial(
                self._download_single, download_config=download_config, get_schema=True
            )

            fs: fsspec.AbstractFileSystem
            path = str(url_or_filenames[0])
            if is_relative_path(path):
                # append the relative path to the base_path
                path = url_or_path_join(self._base_path, path)
            fs, path = url_to_fs(path, **download_config.storage_options)
            size = 0
            try:
                size = fs.info(path).get("size", 0)
            except Exception:
                pass
            max_workers = (
                datasets.config.HF_DATASETS_MULTITHREADING_MAX_WORKERS
                if size < (20 << 20)
                else 1
            )  # enable multithreading if files are small

            out = thread_map(
                download_func,
                url_or_filenames,
                desc=download_config.download_desc
                or "Downloading files and gathering schema",
                unit="files",
                position=multiprocessing.current_process()._identity[
                    -1
                ]  # contains the ranks of subprocesses
                if os.environ.get(
                    "HF_DATASETS_STACK_MULTIPROCESSING_DOWNLOAD_PROGRESS_BARS"
                )
                == "1"
                and multiprocessing.current_process()._identity
                else None,
                max_workers=max_workers,
                tqdm_class=tqdm,
            )
        else:
            out = [
                self._download_single(
                    url_or_filename, download_config=download_config, get_schema=True
                )
                for url_or_filename in url_or_filenames
            ]
        files, features = zip(*out)
        self.features = {}
        for feature in features:
            if feature is not None:
                self.features.update(feature)
        if len(self.features) == 0:
            self.features = None
        return list(files)

    def _gather_schema(self, file):
        if file.endswith(".parquet") or file.endswith(".pq"):
            with open(file, "rb") as f:
                return Features.from_arrow_schema(pq.read_schema(f))
        elif file.endswith(".arrow") or file.endswith(".feather"):
            with open(file, "rb") as f:
                return Features.from_arrow_schema(pa.ipc.read_schema(f))
        elif is_polars_available():
            if file.endswith(".csv"):
                import polars as pl

                schema = pl.scan_csv(file).collect_schema()

                schema = pl.DataFrame(schema=schema).to_arrow().schema
                return Features.from_arrow_schema(schema)
            elif file.endswith(".tsv") or file.endswith(".txt"):
                import polars as pl

                schema = pl.scan_csv(file, separator="\t").collect_schema()

                schema = pl.DataFrame(schema=schema).to_arrow().schema
                return Features.from_arrow_schema(schema)

    def _download_single(
        self, url_or_filename: str, download_config: DownloadConfig, get_schema=False
    ) -> str:
        out = super()._download_single(url_or_filename, download_config)
        if get_schema:
            return out, self._gather_schema(out)
        return out
