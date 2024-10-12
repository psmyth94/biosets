import importlib

from biosets.utils import logging

from ..patcher import (
    Patcher,
    PatcherConfig,
    get_hashed_patches,
)

logger = logging.get_logger(__name__)


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


ENTITY_PATHS = [
    "biosets.packaged_modules._PACKAGED_DATASETS_MODULES",
    "biosets.packaged_modules._MODULE_SUPPORTS_METADATA",
    "biosets.packaged_modules._MODULE_TO_EXTENSIONS",
    "biosets.arrow_dataset.Dataset",
    "biosets.data_files.METADATA_PATTERNS",
]

MODULE_PATH = [
    "biosets.packaged_modules",
    "biosets.streaming",
]


class DatasetsPatcherConfig(PatcherConfig, metaclass=SingletonMeta):
    """
    Configuration class for patching datasets in the biosets package.

    Attributes:
        patches (list): List of patches to be applied.
        root (module): Root module where the patches will be applied.
        patch_targets (list): List of modules where the patches will be applied.
    """

    def __init__(self):
        self.patches = get_hashed_patches(
            entity_paths=ENTITY_PATHS, module_paths=MODULE_PATH
        )
        self.root = importlib.import_module("datasets")
        self.patch_targets = [
            importlib.import_module("datasets"),
            importlib.import_module("biosets"),
        ]
        super().__init__(
            patches=self.patches, root=self.root, patch_targets=self.patch_targets
        )


class DatasetsPatcher(Patcher, metaclass=SingletonMeta):
    """
    A class for patching datasets.

    This class provides functionality for patching datasets in the GenOmicsML library.

    Attributes:
        config (DatasetsPatcherConfig): An instance of the DatasetsPatcherConfig class.

    """

    def __init__(self):
        super().__init__(config=DatasetsPatcherConfig())
