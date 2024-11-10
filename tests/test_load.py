import textwrap
import os

import pyarrow as pa
import pytest
from datasets import Features, Value
from datasets.arrow_writer import ArrowWriter

from biosets import load_dataset


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
def data_dir_biodata(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    filename = data_dir / "samples_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample1,batch1,a,2,a
                sample2,batch2,b,20,b
                """
            )
        )

    filename = data_dir / "samples_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample3,batch3,c,3,c
                sample4,batch4,d,40,d
                """
            )
        )

    filename = data_dir / "features_1.jsonl"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """\
                {"feature": "header1", "metadata1": "a", "metadata2": 2}
                {"feature": "header2", "metadata1": "b", "metadata2": 20}
                """
            )
        )

    filename = data_dir / "features_2.jsonl"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """\
                {"feature": "header3", "metadata1": "a", "metadata2": 2}
                {"feature": "header4", "metadata1": "b", "metadata2": 20}
                """
            )
        )

    filename = data_dir / "data_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header1,header2,header3,header4
                sample1,1,2,3,4
                sample2,10,20,30,40
                """
            )
        )

    filename = data_dir / "data_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header1,header2,header3,header4
                sample3,3,4,5,6
                sample4,30,40,50,60
                """
            )
        )

    return str(data_dir)


@pytest.fixture
def data_dir_biodata_with_split_names(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    filename = data_dir / "train-samples_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample1,batch1,a,2,a
                sample2,batch2,b,20,b
                """
            )
        )

    filename = data_dir / "test-samples_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample3,batch3,c,3,c
                sample4,batch4,d,40,d
                """
            )
        )

    filename = data_dir / "features.jsonl"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """\
                {"feature": "header1", "metadata1": "a", "metadata2": 2}
                {"feature": "header2", "metadata1": "b", "metadata2": 20}
                """
            )
        )

    filename = data_dir / "train-data_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header1,header2
                sample1,1,2
                sample2,10,20
                """
            )
        )

    filename = data_dir / "test-data_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header1,header2
                sample3,3,4
                sample4,30,40
                """
            )
        )

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
def sample_metadata_file_2(tmp_path):
    filename = tmp_path / "sample_metadata_2.csv"
    data = textwrap.dedent(
        """
        sample,batch,metadata1,metadata2,target
        sample3,batch3,c,3,c
        sample4,batch4,d,40,d
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def data_with_samples(tmp_path):
    filename = tmp_path / "data_with_samples.csv"
    data = textwrap.dedent(
        """
        sample,header1,header2
        sample1,1,2
        sample2,10,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def data_with_samples_2(tmp_path):
    filename = tmp_path / "data_with_samples_2.csv"
    data = textwrap.dedent(
        """
        sample,header1,header2
        sample3,3,4
        sample4,30,40
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


def test_load_dataset_with_dir(data_dir_biodata):
    ds = load_dataset(
        "csv",
        split="train",
        data_dir=data_dir_biodata,
        labels=["a", "b", "c", "d"],
    )
    ds_item = next(iter(ds))
    assert ds_item == {
        "sample": "sample1",
        "batch": "batch1",
        "metadata1": "a",
        "metadata2": 2,
        "target": "a",
        "header1": 1,
        "header2": 2,
        "header3": 3,
        "header4": 4,
        "labels": 0,
    }


def test_load_dataset_with_dir_and_split_names(data_dir_biodata_with_split_names):
    ds = load_dataset(
        "csv",
        data_dir=data_dir_biodata_with_split_names,
        labels=["a", "b", "c", "d"],
    )
    ds_train_item = next(iter(ds["train"]))
    assert ds_train_item == {
        "sample": "sample1",
        "batch": "batch1",
        "metadata1": "a",
        "metadata2": 2,
        "target": "a",
        "header1": 1,
        "header2": 2,
        "labels": 0,
    }

    ds_test_item = next(iter(ds["test"]))
    assert ds_test_item == {
        "sample": "sample3",
        "batch": "batch3",
        "metadata1": "c",
        "metadata2": 3,
        "target": "c",
        "header1": 3,
        "header2": 4,
        "labels": 2,
    }


@pytest.mark.parametrize("path_extension", ["csv", "csv.bz2"])
@pytest.mark.parametrize("streaming", [False])
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
    # assert isinstance(ds, IterableDataset if streaming else datasets.Dataset)
    ds_item = next(iter(ds))
    assert ds_item == {"col_1": "0", "col_2": 0, "col_3": 0.0}


@pytest.mark.parametrize("streaming", [False])
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


@pytest.mark.parametrize("streaming", [False])
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


# class ModuleFactoryTest(TestCase):
#     @pytest.fixture(autouse=True)
#     def inject_fixtures(
#         self,
#         jsonl_path,
#         data_dir,
#         data_dir_with_metadata,
#     ):
#         self._jsonl_path = jsonl_path
#         self._data_dir = data_dir
#         self._data_dir_with_metadata = data_dir_with_metadata
#
#     def setUp(self):
#         self.hf_modules_cache = tempfile.mkdtemp()
#         self.cache_dir = tempfile.mkdtemp()
#         self.download_config = DownloadConfig(cache_dir=self.cache_dir)
#         self.dynamic_modules_path = datasets.load.init_dynamic_modules(
#             name="test_datasets_modules_" + os.path.basename(self.hf_modules_cache),
#             hf_modules_cache=self.hf_modules_cache,
#         )
#
#     def test_PackagedDatasetModuleFactory(self):
#         factory = PackagedDatasetModuleFactory(
#             "csv",
#             data_files=self._jsonl_path,
#             download_config=self.download_config,
#         )
#         module_factory_result = factory.get_module()
#         assert importlib.import_module(module_factory_result.module_path) is not None
#
#     def test_PackagedDatasetModuleFactory_with_data_dir(self):
#         factory = PackagedDatasetModuleFactory(
#             "csv", data_dir=self._data_dir, download_config=self.download_config
#         )
#         module_factory_result = factory.get_module()
#         assert importlib.import_module(module_factory_result.module_path) is not None
#         data_files = module_factory_result.builder_kwargs.get("data_files")
#         assert (
#             data_files is not None
#             and len(data_files["train"]) > 0
#             and len(data_files["test"]) > 0
#         )
#         assert Path(data_files["train"][0]).parent.samefile(self._data_dir)
#         assert Path(data_files["test"][0]).parent.samefile(self._data_dir)
#
#     def test_PackagedDatasetModuleFactory_with_data_dir_and_metadata(self):
#         factory = PackagedDatasetModuleFactory(
#             "biodata",
#             data_dir=self._data_dir_with_metadata,
#             download_config=self.download_config,
#         )
#         module_factory_result = factory.get_module()
#         assert importlib.import_module(module_factory_result.module_path) is not None
#         data_files = module_factory_result.builder_kwargs.get("data_files")
#         assert (
#             data_files is not None
#             and len(data_files["train"]) > 0
#             and len(data_files["test"]) > 0
#         )
#         assert Path(data_files["train"][0]).parent.samefile(
#             self._data_dir_with_metadata
#         )
#         assert Path(data_files["test"][0]).parent.samefile(self._data_dir_with_metadata)
#         assert any(
#             Path(data_file).name == "metadata.jsonl"
#             for data_file in data_files["train"]
#         )
#         assert any(
#             Path(data_file).name == "metadata.jsonl" for data_file in data_files["test"]
#         )
