import importlib
import os
import tempfile
from pathlib import Path
from unittest import TestCase

import datasets
import pyarrow as pa
import pytest
from datasets import DownloadConfig, Features, IterableDataset, Value
from datasets.arrow_writer import ArrowWriter
from datasets.config import METADATA_CONFIGS_FIELD
from datasets.load import PackagedDatasetModuleFactory

from biosets import load_dataset
from biosets.integration import DatasetsPatcher


@pytest.fixture
def data_dir(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    with open(data_dir / "train.txt", "w") as f:
        f.write("foo\n" * 10)
    with open(data_dir / "test.txt", "w") as f:
        f.write("bar\n" * 10)
    return str(data_dir)


@pytest.fixture
def data_dir_with_arrow(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    output_train = os.path.join(data_dir, "train.arrow")
    with ArrowWriter(path=output_train) as writer:
        writer.write_table(pa.Table.from_pydict({"col_1": ["foo"] * 10}))
        num_examples, num_bytes = writer.finalize()
    assert num_examples == 10
    assert num_bytes > 0
    output_test = os.path.join(data_dir, "test.arrow")
    with ArrowWriter(path=output_test) as writer:
        writer.write_table(pa.Table.from_pydict({"col_1": ["bar"] * 10}))
        num_examples, num_bytes = writer.finalize()
    assert num_examples == 10
    assert num_bytes > 0
    return str(data_dir)


@pytest.fixture
def data_dir_with_metadata(tmp_path):
    data_dir = tmp_path / "data_dir_with_metadata"
    data_dir.mkdir()
    with open(data_dir / "train.jpg", "wb") as f:
        f.write(b"train_image_bytes")
    with open(data_dir / "test.jpg", "wb") as f:
        f.write(b"test_image_bytes")
    with open(data_dir / "metadata.jsonl", "w") as f:
        f.write(
            """\
        {"file_name": "train.jpg", "caption": "Cool tran image"}
        {"file_name": "test.jpg", "caption": "Cool test image"}
        """
        )
    return str(data_dir)


@pytest.fixture
def data_dir_with_single_config_in_metadata(tmp_path):
    data_dir = tmp_path / "data_dir_with_one_default_config_in_metadata"

    cats_data_dir = data_dir / "cats"
    cats_data_dir.mkdir(parents=True)
    dogs_data_dir = data_dir / "dogs"
    dogs_data_dir.mkdir(parents=True)

    with open(cats_data_dir / "cat.jpg", "wb") as f:
        f.write(b"this_is_a_cat_image_bytes")
    with open(dogs_data_dir / "dog.jpg", "wb") as f:
        f.write(b"this_is_a_dog_image_bytes")
    with open(data_dir / "README.md", "w") as f:
        f.write(
            f"""\
---
{METADATA_CONFIGS_FIELD}:
  - config_name: custom
    drop_labels: true
---
        """
        )
    return str(data_dir)


@pytest.fixture
def data_dir_with_config_and_data_files(tmp_path):
    data_dir = tmp_path / "data_dir_with_config_and_data_files"

    cats_data_dir = data_dir / "data" / "cats"
    cats_data_dir.mkdir(parents=True)
    dogs_data_dir = data_dir / "data" / "dogs"
    dogs_data_dir.mkdir(parents=True)

    with open(cats_data_dir / "cat.jpg", "wb") as f:
        f.write(b"this_is_a_cat_image_bytes")
    with open(dogs_data_dir / "dog.jpg", "wb") as f:
        f.write(b"this_is_a_dog_image_bytes")
    with open(data_dir / "README.md", "w") as f:
        f.write(
            f"""\
---
{METADATA_CONFIGS_FIELD}:
  - config_name: custom
    data_files: "data/**/*.jpg"
---
        """
        )
    return str(data_dir)


@pytest.fixture
def data_dir_with_two_config_in_metadata(tmp_path):
    data_dir = tmp_path / "data_dir_with_two_configs_in_metadata"
    cats_data_dir = data_dir / "cats"
    cats_data_dir.mkdir(parents=True)
    dogs_data_dir = data_dir / "dogs"
    dogs_data_dir.mkdir(parents=True)

    with open(cats_data_dir / "cat.jpg", "wb") as f:
        f.write(b"this_is_a_cat_image_bytes")
    with open(dogs_data_dir / "dog.jpg", "wb") as f:
        f.write(b"this_is_a_dog_image_bytes")

    with open(data_dir / "README.md", "w") as f:
        f.write(
            f"""\
---
{METADATA_CONFIGS_FIELD}:
  - config_name: "v1"
    drop_labels: true
    default: true
  - config_name: "v2"
    drop_labels: false
---
        """
        )
    return str(data_dir)


@pytest.fixture
def data_dir_with_data_dir_configs_in_metadata(tmp_path):
    data_dir = tmp_path / "data_dir_with_two_configs_in_metadata"
    cats_data_dir = data_dir / "cats"
    cats_data_dir.mkdir(parents=True)
    dogs_data_dir = data_dir / "dogs"
    dogs_data_dir.mkdir(parents=True)

    with open(cats_data_dir / "cat.jpg", "wb") as f:
        f.write(b"this_is_a_cat_image_bytes")
    with open(dogs_data_dir / "dog.jpg", "wb") as f:
        f.write(b"this_is_a_dog_image_bytes")


@pytest.fixture
def sub_data_dirs(tmp_path):
    data_dir2 = tmp_path / "data_dir2"
    relative_subdir1 = "subdir1"
    sub_data_dir1 = data_dir2 / relative_subdir1
    sub_data_dir1.mkdir(parents=True)
    with open(sub_data_dir1 / "train.txt", "w") as f:
        f.write("foo\n" * 10)
    with open(sub_data_dir1 / "test.txt", "w") as f:
        f.write("bar\n" * 10)

    relative_subdir2 = "subdir2"
    sub_data_dir2 = tmp_path / data_dir2 / relative_subdir2
    sub_data_dir2.mkdir(parents=True)
    with open(sub_data_dir2 / "train.txt", "w") as f:
        f.write("foo\n" * 10)
    with open(sub_data_dir2 / "test.txt", "w") as f:
        f.write("bar\n" * 10)

    return str(data_dir2), relative_subdir1


@pytest.fixture
def complex_data_dir(tmp_path):
    data_dir = tmp_path / "complex_data_dir"
    data_dir.mkdir()
    (data_dir / "data").mkdir()
    with open(data_dir / "data" / "train.txt", "w") as f:
        f.write("foo\n" * 10)
    with open(data_dir / "data" / "test.txt", "w") as f:
        f.write("bar\n" * 10)
    with open(data_dir / "README.md", "w") as f:
        f.write("This is a readme")
    with open(data_dir / ".dummy", "w") as f:
        f.write("this is a dummy file that is not a data file")
    return str(data_dir)


@pytest.mark.parametrize("path_extension", ["csv", "csv.bz2"])
@pytest.mark.parametrize("streaming", [False, True])
def test_load_dataset_streaming_csv(path_extension, streaming, csv_path, bz2_csv_path):
    paths = {"csv": csv_path, "csv.bz2": bz2_csv_path}
    data_files = str(paths[path_extension])
    features = Features(
        {"col_1": Value("string"), "col_2": Value("int32"), "col_3": Value("float32")}
    )
    ds = load_dataset(
        "csv",
        split="train",
        data_files=data_files,
        features=features,
        streaming=streaming,
    )
    assert isinstance(ds, IterableDataset if streaming else datasets.Dataset)
    ds_item = next(iter(ds))
    assert ds_item == {"col_1": "0", "col_2": 0, "col_3": 0.0}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "data_file", ["zip_csv_path", "zip_csv_with_dir_path", "csv_path"]
)
def test_load_dataset_zip_csv(
    data_file, streaming, zip_csv_path, zip_csv_with_dir_path, csv_path
):
    data_file_paths = {
        "zip_csv_path": zip_csv_path,
        "zip_csv_with_dir_path": zip_csv_with_dir_path,
        "csv_path": csv_path,
    }
    data_files = str(data_file_paths[data_file])
    expected_size = 8 if data_file.startswith("zip") else 4
    features = Features(
        {"col_1": Value("string"), "col_2": Value("int32"), "col_3": Value("float32")}
    )
    ds = load_dataset(
        "csv",
        split="train",
        data_files=data_files,
        features=features,
        streaming=streaming,
    )
    if streaming:
        ds_item_counter = 0
        for ds_item in ds:
            if ds_item_counter == 0:
                assert ds_item == {"col_1": "0", "col_2": 0, "col_3": 0.0}
            ds_item_counter += 1
        assert ds_item_counter == expected_size
    else:
        assert ds.shape[0] == expected_size
        ds_item = next(iter(ds))
        assert ds_item == {"col_1": "0", "col_2": 0, "col_3": 0.0}


@pytest.mark.parametrize("streaming", [False, True])
def test_load_dataset_arrow(streaming, data_dir_with_arrow):
    ds = load_dataset(
        "arrow", split="train", data_dir=data_dir_with_arrow, streaming=streaming
    )
    expected_size = 10
    if streaming:
        ds_item_counter = 0
        for ds_item in ds:
            if ds_item_counter == 0:
                assert ds_item == {"col_1": "foo"}
            ds_item_counter += 1
        assert ds_item_counter == 10
    else:
        assert ds.num_rows == 10
        assert ds.shape[0] == expected_size
        ds_item = next(iter(ds))
        assert ds_item == {"col_1": "foo"}


class ModuleFactoryTest(TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(
        self,
        jsonl_path,
        data_dir,
        data_dir_with_metadata,
    ):
        self._jsonl_path = jsonl_path
        self._data_dir = data_dir
        self._data_dir_with_metadata = data_dir_with_metadata

    def setUp(self):
        self.hf_modules_cache = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()
        self.download_config = DownloadConfig(cache_dir=self.cache_dir)
        self.dynamic_modules_path = datasets.load.init_dynamic_modules(
            name="test_datasets_modules_" + os.path.basename(self.hf_modules_cache),
            hf_modules_cache=self.hf_modules_cache,
        )

    def test_PackagedDatasetModuleFactory(self):
        with DatasetsPatcher():
            factory = PackagedDatasetModuleFactory(
                "biodata",
                data_files=self._jsonl_path,
                download_config=self.download_config,
            )
            module_factory_result = factory.get_module()
            assert (
                importlib.import_module(module_factory_result.module_path) is not None
            )

    def test_PackagedDatasetModuleFactory_with_data_dir(self):
        with DatasetsPatcher():
            factory = PackagedDatasetModuleFactory(
                "biodata", data_dir=self._data_dir, download_config=self.download_config
            )
            module_factory_result = factory.get_module()
            assert (
                importlib.import_module(module_factory_result.module_path) is not None
            )
            data_files = module_factory_result.builder_kwargs.get("data_files")
            assert (
                data_files is not None
                and len(data_files["train"]) > 0
                and len(data_files["test"]) > 0
            )
            assert Path(data_files["train"][0]).parent.samefile(self._data_dir)
            assert Path(data_files["test"][0]).parent.samefile(self._data_dir)

    def test_PackagedDatasetModuleFactory_with_data_dir_and_metadata(self):
        with DatasetsPatcher():
            factory = PackagedDatasetModuleFactory(
                "biodata",
                data_dir=self._data_dir_with_metadata,
                download_config=self.download_config,
            )
            module_factory_result = factory.get_module()
            assert (
                importlib.import_module(module_factory_result.module_path) is not None
            )
            data_files = module_factory_result.builder_kwargs.get("data_files")
            assert (
                data_files is not None
                and len(data_files["train"]) > 0
                and len(data_files["test"]) > 0
            )
            assert Path(data_files["train"][0]).parent.samefile(
                self._data_dir_with_metadata
            )
            assert Path(data_files["test"][0]).parent.samefile(
                self._data_dir_with_metadata
            )
            assert any(
                Path(data_file).name == "metadata.jsonl"
                for data_file in data_files["train"]
            )
            assert any(
                Path(data_file).name == "metadata.jsonl"
                for data_file in data_files["test"]
            )
