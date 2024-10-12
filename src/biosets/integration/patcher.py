import ast
import importlib
import importlib.util
import inspect
import json
import os
import posixpath
import shutil
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias, Union

import filelock

import biosets.config

from ..utils import gorilla, logging
from ..utils.file_utils import is_remote_url
from ..utils.fingerprint import Hasher, is_caching_enabled
from ..utils.gorilla import (
    SameSourceAndDestinationError,
    _get_members,
    _module_iterator,
)

logger = logging.get_logger(__name__)

_EXCLUDED_MEMBERS = [
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "__annotations__",
]


def dynamic_import(attr_path: str):
    """
    Dynamically imports an attribute (class, function, variable) from a given path.

    :param attr_path: The full path to the attribute, e.g., 'biosets.packaged_modules._PACKAGED_DATASETS_MODULES'
    :return: The attribute object.
    """
    module_path, attr_name = attr_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def parse_module_for_patching(
    module: Union[ModuleType, str],
    hash_module: bool = True,
    extra=[],
    filter: Optional[Callable] = None,
) -> Dict[str, Tuple[Any, str]]:
    if isinstance(module, str):
        module = importlib.import_module(module)

    members = _get_members(
        module, filter=create_patch_filter(module) if filter is None else filter
    )
    _patches = {}
    for member_name, member in members:
        if hash_module:
            _patches[member_name] = tuple(
                [member, module.__name__, Hasher.hash(member)] + extra
            )
        else:
            _patches[member_name] = tuple([member, module.__name__] + extra)
    return _patches


def create_patch_filter(module: ModuleType):
    excluded_imports = set()
    for node in ast.parse(inspect.getsource(module)).body:
        if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
            excluded_imports.update(
                [
                    name.name if name.asname is None else name.asname
                    for name in node.names
                ]
            )

    def filter(name: str, obj: Any):
        if (
            name in _EXCLUDED_MEMBERS
            or name in excluded_imports
            or inspect.ismodule(obj)
        ):
            return False
        return True

    return filter


def get_hashed_patches(entity_paths=[], module_paths=[]):
    """
    Defines patches with their module paths and dynamically loads and hashes their values.

    :return: A list of patches with their values hashed.
    """

    patches = []
    for path in entity_paths:
        value = dynamic_import(path)
        module_name = path.rsplit(".", 1)[0]
        hashed_value = Hasher.hash(value)
        patches.append(
            (
                path.rsplit(".", maxsplit=1)[-1],
                (value, module_name, hashed_value),
            )
        )

    for path in module_paths:
        patches.extend(parse_module_for_patching(path).items())

    return patches


def create_lock_path(root, rel_path):
    lock_path = posixpath.join(
        root, Path(rel_path).as_posix().replace("/", "_") + ".lock"
    )
    return lock_path


class PatcherConfig:
    """Base class for the wrapper config. Handles patching and caching.

    Args:
        patches (Dict[str, Any]): Dictionary of patches to apply. The keys are the names of the members to patch and the values are the patches.
        package (ModuleType): The target package that will be within the `patch_targets`. All members within patches will
        patch_targets (Union[List[ModuleType], ModuleType]): The packages to patch.
        cache_dir (Optional[Union[Path, str]], optional): The cache directory. Defaults to None.
    """

    patches: Dict[str, Any] = None
    root: ModuleType = None
    patch_targets: Union[List[ModuleType], ModuleType] = None
    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    cache_dir: Optional[Union[Path, str]] = None

    def __init__(
        self,
        patches: Dict[str, Any] = None,
        root: ModuleType = None,
        patch_targets: Union[List[ModuleType], ModuleType] = None,
        cache_dir: Optional[Union[Path, str]] = None,
        **kwargs,
    ):
        self._cache_enabled = is_caching_enabled()

        self.patches = self.patches if self.patches is not None else patches
        self.root = self.root if self.root is not None else root
        self.patch_targets = (
            self.patch_targets if self.patch_targets is not None else patch_targets
        )
        self.cache_dir = self.cache_dir if self.cache_dir is not None else cache_dir

        if isinstance(self.patches, dict):
            self.patches = list(self.patches.items())

        if self.patch_targets is None:
            self.patch_targets = [self.root]
        elif not isinstance(self.patch_targets, list):
            self.patch_targets = [self.patch_targets]

        self.cache_files = None

        if self.cache_dir is None:
            self._cache_dir_root = biosets.config.BIOSETS_PATCHES_CACHE.as_posix()
        else:
            if isinstance(self.cache_dir, str) or isinstance(self.cache_dir, Path):
                self._cache_dir_root = self.cache_dir
            elif isinstance(self.cache_dir, dict):
                self._cache_dir_root = self.cache_dir.get(
                    "root", biosets.config.BIOSETS_PATCHES_CACHE
                )
                # specifies the cache files for each patch target.
                for patch_target in self.patch_targets:
                    if patch_target.__name__ in self.cache_dir:
                        for name, path in self.cache_dir[patch_target.__name__].items():
                            if isinstance(path, Path):
                                path = path.as_posix()
                            if os.path.exists(path):
                                if self.cache_files is None:
                                    self.cache_files = defaultdict(dict)
                                self.cache_files[patch_target.__name__][name] = path

        self._cache_dir_root = (
            self._cache_dir_root
            if is_remote_url(self._cache_dir_root)
            else os.path.expanduser(self._cache_dir_root)
        )

        self._relative_ref_pkg_cache_dir = self._build_relative_cache_dir(self.root)
        self._relative_patch_targets_cache_dirs = {
            p.__name__: self._build_relative_cache_dir(p)
            for p in self.patch_targets
            if hasattr(p, "__name__")
        }

        self.patches = self._sort_patches(self._prepare_patches(self.patches))
        self._output_dir = self._build_output_dir()
        # NOTE: this is causing issues with the cache, so we will disable it for now
        # self.cache_files = self._get_cache_files()
        self.cache_files = None

        self._attemps = kwargs.get("attemps", 1)

    def _generate_modules(self):
        return {
            package.__name__: (
                module for module in _module_iterator(package, recursive=True)
            )
            for package in self.patch_targets
            if hasattr(package, "__name__")
        }

    def _build_output_dir(self):
        _output_dir = Hasher().hash(self.patches)
        os.makedirs(os.path.join(self._cache_dir_root, _output_dir), exist_ok=True)
        return _output_dir

    def _build_relative_cache_dir(self, package):
        """Return the data directory for the current version."""
        version = package.__version__ if hasattr(package, "__version__") else "0.0.0"
        package_name = package.__name__ if hasattr(package, "__name__") else "unknown"
        relative_patches_dir = posixpath.join(package_name, version)
        return relative_patches_dir

    def _prepare_patches(self, patches: Dict[str, Any]):
        if patches is None:
            raise ValueError("patches cannot be None.")
        _patches = []
        if not is_remote_url(self._cache_dir_root):
            source_cache_dir = posixpath.join(
                self._cache_dir_root, self._relative_ref_pkg_cache_dir
            )
            os.makedirs(source_cache_dir, exist_ok=True)
            for name, patch in patches:
                if isinstance(patch, Tuple):
                    if len(patch) == 2:
                        patch_, source_ = patch
                        hash_ = Hasher.hash(patch_)
                        if isinstance(source_, ModuleType):
                            source_ = source_.__name__
                        # check if source exists
                        if not isinstance(source_, str) or not importlib.util.find_spec(
                            source_
                        ):
                            raise ValueError(
                                f"Invalid source: {source_}, source must be a module."
                            )
                        _patches.append((name, (patch_, source_, hash_, None)))
                    elif len(patch) == 3:
                        patch_, source_, hash_ = patch
                        if isinstance(source_, ModuleType):
                            source_ = source_.__name__
                        if not isinstance(source_, str) or not importlib.util.find_spec(
                            source_
                        ):
                            raise ValueError(
                                f"Invalid source: {source_}, source must be a module."
                            )
                        if hash_ is None:
                            hash_ = Hasher.hash(patch)
                        _patches.append((name, (patch_, source_, hash_, None)))
                    elif len(patch) == 4:
                        patch_, source_, hash_, destination_ = patch
                        if isinstance(source_, ModuleType):
                            source_ = source_.__name__
                        if not isinstance(source_, str) or not importlib.util.find_spec(
                            source_
                        ):
                            raise ValueError(
                                f"Invalid source: {source_}, source must be a module."
                            )
                        if hash_ is None:
                            hash_ = Hasher.hash(patch)
                        if isinstance(destination_, str):
                            destination_ = importlib.import_module(destination_)
                        if not isinstance(destination_, ModuleType):
                            raise ValueError(
                                f"Invalid destination: {destination_}, destination must be a module."
                            )
                        _patches.append((name, (patch_, source_, hash_, destination_)))
                else:
                    hash_ = Hasher.hash(patch)
                    origin = inspect.getmodule(patch)
                    source_ = origin.__name__ if hasattr(origin, "__name__") else None
                    _patches.append((name, (patch_, source_, hash_, None)))
        return _patches

    def _get_cache_files(self):
        cache_files = None
        if self._cache_enabled and not os.path.exists(
            posixpath.join(
                self._cache_dir_root, self._output_dir, biosets.config.PATCHES_FILENAME
            )
        ):
            for name, (_, _, hash, _) in self.patches:
                for (
                    patch_target_name,
                    relative_target_cache_dir,
                ) in self._relative_patch_targets_cache_dirs.items():
                    cache_file = posixpath.join(
                        self._cache_dir_root,
                        self._relative_ref_pkg_cache_dir,
                        relative_target_cache_dir,
                        hash,
                    )
                    os.makedirs(cache_file, exist_ok=True)
                    patches_file = f"{cache_file}/{biosets.config.PATCHES_FILENAME}"
                    if os.path.exists(patches_file):
                        if cache_files is None:
                            cache_files = defaultdict(dict)
                        cache_files[patch_target_name][name] = patches_file
                    no_patches_file = f"{cache_file}/{biosets.config.NO_PATCHES_FILENAME}"
                    if os.path.exists(no_patches_file):
                        if cache_files is None:
                            cache_files = defaultdict(dict)
                        cache_files[patch_target_name][name] = no_patches_file
        return cache_files

    @classmethod
    def clear_cache(cls, cache_dir: Optional[Union[Path, str]] = None):
        """Clear a directory in the root cache directory for patches.

        Args:
            cache_dir (Optional[Union[Path, str]], optional): The cache directory to delete.
            If None, the entire cache directory will be deleted. Defaults to None.
        """
        if cache_dir is None:
            cache_dir = (
                cls._cache_dir_root
                if hasattr(cls, "_cache_dir_root")
                else biosets.config.BIOSETS_PATCHES_CACHE.as_posix()
            )
            if hasattr(cls, "_output_dir"):
                cache_dir = posixpath.join(cache_dir, cls._output_dir)

        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    def _clear_cache(self):
        self.cache_files = None
        for _, (_, _, hash, _) in self.patches:
            for (
                relative_target_cache_dir
            ) in self._relative_patch_targets_cache_dirs.values():
                rel_member_dir = posixpath.join(
                    self._relative_ref_pkg_cache_dir, relative_target_cache_dir, hash
                )
                member_dir = posixpath.join(self._cache_dir_root, rel_member_dir)
                # lock the directory
                if os.path.exists(member_dir):
                    for file in os.listdir(member_dir):
                        rel_fp = posixpath.join(rel_member_dir, file)
                        fp = os.path.join(self._cache_dir_root, rel_fp)
                        lock_path = create_lock_path(self._cache_dir_root, rel_fp)
                        with filelock.FileLock(lock_path):
                            os.remove(fp)
                    with filelock.FileLock(
                        create_lock_path(self._cache_dir_root, rel_member_dir)
                    ):
                        os.rmdir(member_dir)
            output_dir = posixpath.join(self._cache_dir_root, self._output_dir)
            fp = posixpath.join(output_dir, biosets.config.PATCHES_FILENAME)
            if os.path.exists(fp):
                with filelock.FileLock(
                    create_lock_path(
                        self._cache_dir_root,
                        posixpath.join(self._output_dir, biosets.config.PATCHES_FILENAME),
                    )
                ):
                    os.remove(fp)
            if os.path.exists(output_dir):
                with filelock.FileLock(
                    create_lock_path(self._cache_dir_root, self._output_dir)
                ):
                    shutil.rmtree(output_dir)

        self._cleanup_empty_dir()

    def _cleanup_empty_dir(self):
        # use os.walk to find empty directories in self._cache_dir_root
        for root, dirs, files in os.walk(self._cache_dir_root, topdown=False):
            if len(dirs) == 0 and len(files) == 0:
                os.rmdir(root)

    def _sort_patches(self, patches: list):
        """Sorts the patches for consistent hashing"""
        unique_keys = [str(p) for p in patches]
        inds = sorted(range(len(unique_keys)), key=lambda index: unique_keys[index])
        return [patches[i] for i in inds]


class Patcher:
    """
    Patcher class for applying patches to target packages/modules.
    """

    config: PatcherConfig = None
    _lock: Path = None

    def __init__(self, config: PatcherConfig = None, **kwargs):
        """
        Initialize the Patcher object.

        Args:
            configs (Union[PatcherConfig, List[PatcherConfig]]): Configuration object or list of configuration objects.
            filter (Optional[Union[str, List[str]]], optional):
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.config = self.config if self.config is not None else config
        self._cache_enabled = is_caching_enabled()
        self.patches = self._sort_patches(self.load_and_prepare_patches())
        self._fingerprint = Hasher.hash(str(self.patches))
        self._validate_cache()

    def _sort_patches(self, patches: list):
        """Sorts the patches for consistent hashing"""
        return self.config._sort_patches(patches)

    @contextmanager
    def patch(self, patches: List[gorilla.Patch]):
        """Context manager for applying additional patches to the target packages/modules."""
        for patch in patches:
            gorilla.apply(patch)

        self._apply_patches()

        yield

        self._revert_patches()

        for patch in patches:
            gorilla.revert(patch)

    def _apply_patches(self):
        if self._lock:
            # means that it is trying to run in nested context of the same instance
            # specify this by setting _cleanup to None
            self._cleanup = None
            return

        thread_id = threading.get_ident()
        file_path = os.path.normpath(
            posixpath.join(self.config._output_dir, f"{self._fingerprint}_{thread_id}")
        )
        self._lock = Path(create_lock_path(self.config._cache_dir_root, file_path))
        self._cleanup = False

        if not self._lock.exists():
            self._cleanup = True
            for patch in self.patches:
                gorilla.apply(patch)
            self._lock.parent.mkdir(parents=True, exist_ok=True)
            self._lock.touch()
        else:
            self._lock = None

    def _revert_patches(self):
        if self._cleanup is None:
            # same instance is already running, we put it back to true
            self._cleanup = True
            return
        if self._cleanup:
            for patch in self.patches:
                gorilla.revert(patch)
            # remove the lock file
            if self._lock and self._lock.exists():
                self._lock.unlink()
        self._lock = None

    def __enter__(self):
        self._apply_patches()

    def __exit__(self, exc_type, exc_value, traceback):
        self._revert_patches()

    def load_and_prepare_patches(self):
        output_file = posixpath.join(
            self.config._output_dir, biosets.config.PATCHES_FILENAME
        )
        fp = posixpath.join(self.config._cache_dir_root, output_file)
        if self._cache_enabled and os.path.exists(fp):
            try:
                return self._load_patches_from_cache(fp)
            except Exception as e:
                # something went wrong with loading the patches from cache, so we will clear the cache and recompile
                self.config._clear_cache()
                logger.warning(f"Failed to load patches from cache: {e}")

        patches: List[gorilla.Patch] = []
        _uncached_patches = defaultdict(dict)
        for name, (patch, source, hash, destination) in self.config.patches:
            if destination is not None:
                try:
                    patches.append(
                        gorilla.Patch(
                            destination,
                            name=name,
                            obj=patch,
                            source=source,
                            settings=self.config.settings,
                        )
                    )
                except SameSourceAndDestinationError:
                    # a patch with the same source and destination was given, raise an error
                    raise SameSourceAndDestinationError(
                        f'Patch with same source and destination: ("{name}", ({name}, "{source}", "{hash}", "{destination}"))'
                    )
            for patch_target in self.config.patch_targets:
                _uncached_patches[patch_target.__name__][name] = (patch, source, hash)

        patches += self.create_patches(_uncached_patches)
        try:
            self._save_patches_to_cache(patches, fp)
        except Exception as e:
            logger.warning(f"Failed to save patches to cache: {e}")

            if os.path.exists(fp):
                with filelock.FileLock(
                    create_lock_path(self.config._cache_dir_root, output_file)
                ):
                    os.remove(fp)
                with filelock.FileLock(
                    create_lock_path(
                        self.config._cache_dir_root, self.config._output_dir
                    )
                ):
                    os.rmdir(
                        os.path.join(
                            self.config._cache_dir_root, self.config._output_dir
                        )
                    )
        return patches

    def create_patches(self, patches: Dict[str, Any] = None):
        results = []
        if len(patches) > 0:
            for patch_target in self.config.patch_targets:
                _patches: List[Tuple[gorilla.Patch, str]] = self._create_patches(
                    patches,
                    patch_target,
                    return_hash=True,
                )
                results += [p for p, _ in _patches]

                # NOTE: this is causing issues with the cache, so we will disable it for now
                # if self._cache_enabled:
                #     self._save_patches_to_patch_cache(_patches, patch_target.__name__)

        return results

    def _create_patches(
        self,
        patches: Dict[str, Any],
        patch_target: ModuleType,
        return_hash=False,
    ) -> Union[List[gorilla.Patch], List[Tuple[gorilla.Patch, str]]]:
        """Create patches for within `~self.config.package`.

        Args:
            patches (`Dict[str, Any]`):
                Dictionary of patches to apply. The keys are the names of the members.
                Values can be either the patch (i.e. the object that will replace the member) or a tuple of
                (patch, source module), (patch, hash), or (patch, source module, hash). If tuple is length 2,
                the second element is inferred based on if it is an instance of `ModuleType` or if str is importable.
                Otherwise, it will be taken as the hash.
            patch_targets (`Union[ModuleType, List[ModuleType]]`):
                The target packages to patch. If recursive is True, all submodules will be patched as well.
            recursive (`bool`, Defaults to True):
                Whether to recursively search for modules to patch.
            return_hash (`bool`, Defaults to False):
                Whether to return the hash of the patch in final output. If True, the output will be a tuple of (patch, hash).

        Returns:
            `List[gorilla.Patch]` or `List[Tuple[gorilla.Patch, str]]`:
                List of patches to apply. If return_hash is True, the output will be a tuple of (patch, hash).
        """

        _patches = []

        try:
            for module in self.config._generate_modules()[patch_target.__name__]:
                for asname, name, value in self._find_patches(
                    patches[patch_target.__name__], module
                ):
                    if isinstance(value, tuple):
                        if len(value) == 3:
                            patch, source, hash = value
                        elif len(value) == 2:
                            patch, source = value
                            if importlib.util.find_spec(source):
                                hash = Hasher.hash(patch) if return_hash else None
                            else:
                                hash = source
                                origin = inspect.getmodule(value)
                                source = (
                                    origin.__name__
                                    if hasattr(origin, "__name__")
                                    else None
                                )
                    else:
                        patch = value
                        origin = inspect.getmodule(value)
                        source = (
                            origin.__name__ if hasattr(origin, "__name__") else None
                        )
                        hash = Hasher.hash(patch) if return_hash else None
                    try:
                        if return_hash:
                            _patches.append(
                                (
                                    gorilla.Patch(
                                        module,
                                        name=asname,
                                        obj=patch,
                                        source=source,
                                        source_name=name,
                                        settings=self.config.settings,
                                    ),
                                    hash,
                                )
                            )
                        else:
                            _patches.append(
                                gorilla.Patch(
                                    module,
                                    name=asname,
                                    obj=patch,
                                    source=source,
                                    source_name=name,
                                    settings=self.config.settings,
                                )
                            )
                    except SameSourceAndDestinationError:
                        pass

        except Exception as e:
            logger.warning(
                f"Failed to create patches: {e} for patch_target: {patch_target.__name__}"
            )
            raise e
        return _patches

    def _get_absolute_module_name(
        self, source_module: Union[str, ModuleType], node: ast.ImportFrom
    ) -> str:
        """
        Reconstructs the absolute module name from a relative import.

        Args:
        - source_module (str): The name of the module that contains the relative import.
        - node (ast.ImportFrom): The AST node representing the relative import.

        Returns:
        - str: The absolute module name.
        """
        if not isinstance(node, ast.ImportFrom):
            raise ValueError("Node must be of type ast.ImportFrom")

        relative_level = node.level
        source_module_parts = source_module.split(".")
        if relative_level > len(source_module_parts):
            raise ValueError("Relative level is too high for the current module")

        source_module_parts = source_module_parts[:-relative_level]
        imported_module_parts = node.module.split(".") if node.module else []

        absolute_module_parts = source_module_parts + imported_module_parts
        return ".".join(absolute_module_parts)

    def _find_patches(
        self, patches: Dict[str, Any], module: ModuleType
    ) -> List[Tuple[str, Any]]:
        """
        Find the patches to apply in the module.

        Args:
            patches (Dict[str, Any]): Dictionary of patches to apply.
            module (ModuleType): Module to search for patches.

        Returns:
            OrderedDict[str, Any]: Ordered dictionary of patches to apply.
        """
        try:
            module_source = inspect.getsource(module)
        except OSError:
            with open(module.__file__, "r") as file:
                module_source = file.read()

        exclude_imports = set()
        asname_map = {}

        package_name = self.config.root.__name__

        for node in ast.parse(module_source).body:
            if isinstance(node, ast.ImportFrom):
                module_name = (
                    node.module
                    if node.level == 0
                    else self._get_absolute_module_name(module.__name__, node)
                )
                if package_name not in module_name:
                    exclude_imports.update([name.name for name in node.names])
                else:
                    for name in node.names:
                        if name.asname is not None:
                            asname_map[name.asname] = name.name

        def is_from_package(member: TypeAlias, module: ModuleType, package_name):
            """Check if the member is from the same package as the module."""
            origin = inspect.getmodule(member)
            return origin is None or origin.__name__.startswith(package_name)

        module_globals = inspect.getmembers(
            module, lambda a: is_from_package(a, module, package_name)
        )

        return [
            (
                (name, name, patches[name])
                if name not in asname_map
                else (name, asname_map[name], patches[asname_map[name]])
            )
            for (name, value) in module_globals
            if (
                name not in exclude_imports
                and name not in _EXCLUDED_MEMBERS
                and not inspect.ismodule(value)
                and (
                    name in patches
                    or (name in asname_map and asname_map[name] in patches)
                )
            )
        ]

    def _save_patches_to_cache(
        self, patches: List[Tuple[gorilla.Patch, str]], file_path: str
    ):
        if not is_remote_url(self.config._cache_dir_root):
            _relative_cache_dir = (
                Path(file_path).relative_to(self.config._cache_dir_root).as_posix()
            )
            with filelock.FileLock(
                create_lock_path(self.config._cache_dir_root, _relative_cache_dir)
            ):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w") as f:
                    json.dump([p.to_dict() for p in patches], f)

    def _save_patches_to_patch_cache(
        self, patches: List[Tuple[gorilla.Patch, str]], patch_target: str
    ):
        """
        Save the patches to the cache.

        Args:
            config (PatcherConfig): Patcher configuration object.
            patches (List[Tuple[gorilla.Patch, str]]): List of patches to save.
            patch_targets_name_or_file_path (str, optional): Name of the patch target or the file path to save the patches to.
        """

        if not is_remote_url(self.config._cache_dir_root):
            gorilla_patches: Dict[str, List[gorilla.Patch]] = defaultdict(list)
            for gorilla_patch, hash in patches:
                gorilla_patches[hash].append(gorilla_patch)

            for _, (_, _, hash, _) in self.config.patches:
                if hash in gorilla_patches:
                    continue
                gorilla_patches[hash] = []

            for hash, gorilla_patch in gorilla_patches.items():
                _relative_cache_dir = posixpath.join(
                    self.config._relative_ref_pkg_cache_dir,
                    self.config._relative_patch_targets_cache_dirs[patch_target],
                    hash,
                )
                if gorilla_patch:
                    file_path = os.path.join(
                        self.config._cache_dir_root,
                        _relative_cache_dir,
                        biosets.config.PATCHES_FILENAME,
                    )
                else:
                    file_path = os.path.join(
                        self.config._cache_dir_root,
                        _relative_cache_dir,
                        biosets.config.NO_PATCHES_FILENAME,
                    )
                    # lock the directory
                    with filelock.FileLock(
                        create_lock_path(
                            self.config._cache_dir_root, _relative_cache_dir
                        )
                    ):
                        try:
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            with open(file_path, "w") as f:
                                json.dump([p.to_dict() for p in gorilla_patch], f)

                        except Exception as e:
                            logger.warning(f"Failed to save patch to cache: {e}")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                os.rmdir(os.path.dirname(file_path))

    def _load_patches_from_cache(self, file_path: str) -> List[gorilla.Patch]:
        """
        Load the patches from the cache.

        Args:
            config (PatcherConfig): Patcher configuration object.
            file_path (str): Path to the cache file.

        Returns:
            List[gorilla.Patch]: List of patches loaded from the cache.
        """
        relative_file_path = (
            Path(file_path).relative_to(self.config._cache_dir_root).as_posix()
        )

        # lock the directory
        with filelock.FileLock(
            create_lock_path(self.config._cache_dir_root, relative_file_path)
        ):
            with open(file_path, "r") as f:
                content = json.load(f)
            if content is None:
                raise ValueError(f"Failed to load patch from cache: {file_path}")
            if isinstance(content, dict):
                return [gorilla.Patch.from_dict(content)]
            elif isinstance(content, list):
                return [gorilla.Patch.from_dict(c) for c in content]

    def _validate_cache(self):
        if not is_remote_url(self.config._cache_dir_root):
            output_dir = posixpath.join(
                self.config._cache_dir_root, self.config._output_dir
            )
            probe_file = os.path.normpath(
                posixpath.join(output_dir, f"{self._fingerprint}.txt")
            )
            # look for a .cache file in the output directory
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    if file.endswith(".txt"):
                        fp = os.path.normpath(os.path.join(root, file))
                        if probe_file != fp:
                            logger.debug(
                                "Cache file is outdated or corrupted, clearing the cache and re-building the patches..."
                            )
                            self.config.clear_cache(self.config._output_dir)
                            self.__init__(self.config)
                        return
            if not os.path.exists(probe_file):
                with filelock.FileLock(
                    create_lock_path(
                        self.config._cache_dir_root, self.config._output_dir
                    )
                ):
                    with open(probe_file, "w") as f:
                        f.write("")
