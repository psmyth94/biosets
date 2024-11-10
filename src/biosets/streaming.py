"""
Module adapted from `datasets.streaming`. Modifies the functions to inlude biosets as
exceptions.
"""

import inspect
from typing import TYPE_CHECKING

from datasets.download.download_config import DownloadConfig
from datasets.streaming import extend_module_for_streaming
from datasets.utils.py_utils import get_imports, lock_importable_file

if TYPE_CHECKING:
    from datasets.builder import DatasetBuilder


def extend_dataset_builder_for_streaming(builder: "DatasetBuilder"):
    """Extend the dataset builder module and the modules imported by it to support streaming.

    Args:
        builder (:class:`DatasetBuilder`): Dataset builder instance.
    """
    # this extends the open and os.path.join functions for data streaming
    download_config = DownloadConfig(
        storage_options=builder.storage_options, token=builder.token
    )
    extend_module_for_streaming(builder.__module__, download_config=download_config)
    # if needed, we also have to extend additional internal imports (like wmt14 -> wmt_utils)
    if not builder.__module__.startswith(
        "datasets."
    ) and not builder.__module__.startswith(
        "biosets"
    ):  # check that it's not a packaged builder like csv
        importable_file = inspect.getfile(builder.__class__)
        with lock_importable_file(importable_file):
            for imports in get_imports(importable_file):
                if imports[0] == "internal":
                    internal_import_name = imports[1]
                    internal_module_name = ".".join(
                        builder.__module__.split(".")[:-1] + [internal_import_name]
                    )
                    extend_module_for_streaming(
                        internal_module_name, download_config=download_config
                    )

    # builders can inherit from other builders that might use streaming functionality
    # (for example, ImageFolder and AudioFolder inherit from FolderBuilder which implements examples generation)
    # but these parents builders are not patched automatically as they are not instantiated, so we patch them here
    from datasets.builder import DatasetBuilder

    parent_builder_modules = [
        cls.__module__
        for cls in type(builder).__mro__[
            1:
        ]  # make sure it's not the same module we've already patched
        if issubclass(cls, DatasetBuilder)
        and cls.__module__ != DatasetBuilder.__module__
    ]  # check it's not a standard builder from datasets.builder
    for module in parent_builder_modules:
        extend_module_for_streaming(module, download_config=download_config)
