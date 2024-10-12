import importlib
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import biosets.config
import pytest
from biosets.integration.patcher import (
    Patcher,
    PatcherConfig,
    get_hashed_patches,
)

SOURCE_MODULE_CODE = """
SOURCE_CONSTANT = 42
CONSTANT_WITH_SAME_NAME = 42

def source_function():
    return "new function"

class SourceClass:
    def method(self):
        return "new method"

class ClassWithSameName:
    def method(self):
        return "new method"

def function_with_same_name():
    return "new function"
"""

TARGET_MODULE_CODE = """
TARGET_CONSTANT = 24
CONSTANT_WITH_SAME_NAME = 24
def target_function():
    return "original function"

class TargetClass:
    def method(self):
        return "original method"

class ClassWithSameName:
    def method(self):
        return "original method"

def function_with_same_name():
    return "original function"
"""


pytestmark = pytest.mark.unit


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Define MockPatcherConfig and MockPatcher outside of setUp
class MockPatcherConfig(PatcherConfig, metaclass=SingletonMeta):
    def __init__(self, target_module, source_module):
        self.root = importlib.import_module(target_module)
        self.patch_targets = [self.root]

        self.module_paths = [source_module]

        self.patches = get_hashed_patches(module_paths=self.module_paths)
        super().__init__(
            patches=self.patches,
            root=self.root,
            patch_targets=self.patch_targets,
        )

    def get_mock_patches(self, entity_paths):
        patches = {}
        for path in entity_paths:
            module_name, attr_name = path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            obj = getattr(module, attr_name)
            patches[attr_name] = (obj, module_name)
        return patches


class MockPatcher(Patcher, metaclass=SingletonMeta):
    def __init__(self, target_module, source_module):
        config = MockPatcherConfig(target_module, source_module)
        super().__init__(config=config)


class TestPatcher(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for source and target modules
        self.source_module_dir = tempfile.mkdtemp()
        self.target_module_dir = tempfile.mkdtemp()
        self.cache_dir = biosets.config.BIOSETS_PATCHES_CACHE

        # Names of the modules
        self.source_module_name = "source_module"

        # Write the source module code to a file
        self.source_module_path = os.path.join(
            self.source_module_dir, f"{self.source_module_name}.py"
        )
        with open(self.source_module_path, "w") as f:
            f.write(SOURCE_MODULE_CODE)

        self.target_module_name = "target_module"
        # Write the target module code to a file
        self.target_module_path = os.path.join(
            self.target_module_dir, f"{self.target_module_name}.py"
        )
        with open(self.target_module_path, "w") as f:
            f.write(TARGET_MODULE_CODE)

        # Add both module directories to sys.path
        sys.path.insert(0, self.target_module_dir)
        sys.path.insert(0, self.source_module_dir)

        # Import the source and target modules
        self.source_module = importlib.import_module(self.source_module_name)
        self.target_module = importlib.import_module(self.target_module_name)

        # Initialize MockPatcher
        self.mock_patcher = MockPatcher(
            target_module=self.target_module_name,
            source_module=self.source_module_name,
        )

    def tearDown(self):
        # Clean up the temporary directories and sys.path
        if os.path.exists(self.source_module_dir):
            shutil.rmtree(self.source_module_dir)
        if os.path.exists(self.target_module_dir):
            shutil.rmtree(self.target_module_dir)
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        sys.path.remove(self.source_module_dir)
        sys.path.remove(self.target_module_dir)

        # Remove the source and target modules from sys.modules
        if self.source_module_name in sys.modules:
            del sys.modules[self.source_module_name]
        if self.target_module_name in sys.modules:
            del sys.modules[self.target_module_name]

        # Clear singleton instances
        SingletonMeta._instances.clear()

    def test_mock_patcher_apply_patches(self):
        # Apply patches using MockPatcher and test if the functions are patched
        with self.mock_patcher:
            self.assertEqual(
                self.target_module.function_with_same_name(), "new function"
            )
            class_with_same_name = self.target_module.ClassWithSameName()
            self.assertEqual(class_with_same_name.method(), "new method")
            constant_with_same_name = self.target_module.CONSTANT_WITH_SAME_NAME
            self.assertEqual(constant_with_same_name, 42)

        # After exiting the context manager, the original functions should be restored
        self.assertEqual(
            self.target_module.function_with_same_name(), "original function"
        )
        class_with_same_name = self.target_module.ClassWithSameName()
        self.assertEqual(class_with_same_name.method(), "original method")
        constant_with_same_name = self.target_module.CONSTANT_WITH_SAME_NAME
        self.assertEqual(constant_with_same_name, 24)

    def test_mock_patcher_singleton(self):
        # Ensure that MockPatcher is a singleton
        another_patcher = MockPatcher(
            target_module=self.target_module,
            source_module=self.source_module,
        )
        self.assertIs(self.mock_patcher, another_patcher)

    def test_mock_patcher_config_singleton(self):
        # Ensure that MockPatcherConfig is a singleton
        config1 = self.mock_patcher.config
        config2 = self.mock_patcher.config
        self.assertIs(config1, config2)

    def test_mock_patcher_revert_patches(self):
        # Apply patches and revert them manually
        self.mock_patcher._apply_patches()
        self.assertEqual(self.target_module.function_with_same_name(), "new function")

        self.mock_patcher._revert_patches()
        self.assertEqual(
            self.target_module.function_with_same_name(), "original function"
        )

    def test_mock_patcher_clear_cache(self):
        # Test clearing the cache
        self.mock_patcher.config.clear_cache()
        # ensure that the cache directory is empty
        self.assertFalse(self.cache_dir.exists())

    def test_mock_patcher_exception_handling(self):
        # Test that exceptions within the context manager do not leave patches applied
        try:
            with self.mock_patcher:
                self.assertEqual(
                    self.target_module.function_with_same_name(), "new function"
                )
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Ensure that after the exception, the original functions are restored
        self.assertEqual(
            self.target_module.function_with_same_name(), "original function"
        )

    def test_mock_patcher_thread_safety(self):
        # Test that the MockPatcher is thread-safe
        def thread_target():
            with self.mock_patcher:
                self.assertEqual(
                    self.target_module.function_with_same_name(), "new function"
                )
                self.assertEqual(
                    self.target_module.ClassWithSameName().method(), "new method"
                )

        thread = threading.Thread(target=thread_target)
        thread.start()
        thread.join()

        # Ensure that after the thread has finished, the original functions are restored
        self.assertEqual(
            self.target_module.function_with_same_name(), "original function"
        )
        self.assertEqual(
            self.target_module.ClassWithSameName().method(), "original method"
        )

    def test_mock_patcher_context_manager_nesting(self):
        # Test nesting the MockPatcher context manager
        with self.mock_patcher:
            self.assertEqual(
                self.target_module.function_with_same_name(), "new function"
            )
            with self.mock_patcher:
                self.assertEqual(
                    self.target_module.function_with_same_name(), "new function"
                )
            self.assertEqual(
                self.target_module.function_with_same_name(), "new function"
            )
        self.assertEqual(
            self.target_module.function_with_same_name(), "original function"
        )

    def test_mock_patcher_invalid_patch(self):
        # Test handling of an invalid patch (e.g., invalid source module)
        invalid_entity_paths = ["nonexistent_module.nonexistent_function"]

        class InvalidMockPatcherConfig(PatcherConfig, metaclass=SingletonMeta):
            def __init__(self):
                self.patches = self.get_mock_patches(invalid_entity_paths)
                self.root = self.target_module
                self.patch_targets = [self.target_module]
                super().__init__(
                    patches=self.patches,
                    root=self.root,
                    patch_targets=self.patch_targets,
                )

            def get_mock_patches(self, entity_paths):
                patches = {}
                for path in entity_paths:
                    try:
                        module_name, attr_name = path.rsplit(".", 1)
                        module = importlib.import_module(module_name)
                        obj = getattr(module, attr_name)
                        patches[attr_name] = (obj, module_name)
                    except (ImportError, AttributeError):
                        raise ValueError(f"Invalid entity path: {path}")
                return patches

        with self.assertRaises(ValueError):
            InvalidMockPatcherConfig()

    def test_mock_patcher_with_no_cache(self):
        # Test the MockPatcher when caching is disabled
        with patch("biosets.integration.patcher.is_caching_enabled", return_value=False):
            mock_patcher_no_cache = MockPatcher(
                target_module=self.target_module,
                source_module=self.source_module,
            )

            with mock_patcher_no_cache:
                self.assertEqual(
                    self.target_module.function_with_same_name(), "new function"
                )

            self.assertEqual(
                self.target_module.function_with_same_name(), "original function"
            )
