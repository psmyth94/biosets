import json
import os
import shutil
import textwrap
import unittest
from collections import defaultdict

import datasets.builder
import pandas as pd
import pyarrow as pa
import pytest
from biosets.features import Abundance
from biosets.load import load_dataset, patch_dataset_load
from biosets.packaged_modules.biodata.biodata import (
    TARGET_COLUMN,
    BioData,
    BioDataConfig,
)
from biosets.packaged_modules.csv.csv import Csv
from biosets.packaged_modules.npz.npz import SparseReader
from biosets.utils import logging
from datasets.data_files import (
    DataFilesDict,
    DataFilesList,
    _get_origin_metadata,
)
from datasets.features import Features, Value
from datasets.packaged_modules.json.json import Json

logger = logging.get_logger(__name__)


@pytest.fixture
def csv_file(tmp_path):
    filename = tmp_path / "file.csv"
    data = textwrap.dedent(
        """
        header1,header2
        1,10
        20,2
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def jsonl_file(tmp_path):
    filename = tmp_path / "file.jsonl"
    data = textwrap.dedent(
        """
        {"sample": "sample1", "header1": 1, "header2": 10}
        {"sample": "sample2", "header1": 20, "header2": 2}
        {"sample": "sample3", "header1": 3, "header2": 30}
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def txt_file(tmp_path):
    filename = tmp_path / "file.txt"
    data = textwrap.dedent(
        """
        sample\theader1\theader2
        sample1\t1\t10
        sample2\t20\t2
        sample3\t3\t30
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def npz_file(tmp_path):
    import scipy.sparse as sp

    filename = tmp_path / "file.npz"

    n_samples = 3
    n_features = 2

    data = sp.csr_matrix(
        sp.random(
            n_samples,
            n_features,
            density=0.5,
            format="csr",
            random_state=42,
        )
    )
    sp.save_npz(filename, data)
    return str(filename)


@pytest.fixture
def sample_metadata_file_combined(tmp_path):
    filename = tmp_path / "sample_metadata_combined.csv"
    data = textwrap.dedent(
        """
        sample,batch,metadata1,metadata2,target
        sample1,batch1,a,1,a
        sample2,batch2,b,20,b
        sample3,batch3,c,3,c
        sample4,batch4,d,40,d
        sample5,batch5,e,5,e
        sample6,batch6,f,60,f
        sample7,batch7,g,7,g
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def feature_metadata_file(tmp_path):
    filename = tmp_path / "feature_metadata.csv"
    data = textwrap.dedent(
        """
        feature,metadata1,metadata2
        header1,a,1
        header2,b,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def multiclass(tmp_path):
    filename = tmp_path / "file_multiclass.csv"
    data = textwrap.dedent(
        f"""
        header1,header2,{TARGET_COLUMN}
        1,10,a
        20,2,b
        3,30,c
        40,4,d
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def data_with_index_missing_sample_column(tmp_path):
    filename = tmp_path / "data_with_index_missing_sample_column.csv"
    data = textwrap.dedent(
        """
        header1,header2
        1,10
        20,2
        3,30
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


# Sharded files
## Two files with unequal number of rows for sample and data


@pytest.fixture
def sample_metadata_file(tmp_path):
    filename = tmp_path / "sample_metadata.csv"
    data = textwrap.dedent(
        """
        sample,batch,metadata1,metadata2,target
        sample1,batch1,a,1,a
        sample2,batch2,b,20,b
        sample3,batch3,c,3,c
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def sample_metadata_file_2(tmp_path):
    filename = tmp_path / "sample_metadata_2.csv"
    data = textwrap.dedent(
        """
        sample,batch,metadata1,metadata2,target
        sample4,batch4,d,40,d
        sample5,batch5,e,5,e
        sample6,batch6,f,60,f
        sample7,batch7,g,7,g
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
        sample1,1,10
        sample2,20,2
        sample3,3,30
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
        sample4,40,4
        sample5,5,50
        sample6,60,6
        sample7,7,70
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


# End Sharded files


@pytest.fixture
def data_dir(
    tmp_path,
    data_with_samples,
    data_with_samples_2,
    sample_metadata_file,
    sample_metadata_file_2,
    feature_metadata_file,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # copy all files to data_dir
    shutil.copy(data_with_samples, data_dir / "data_with_samples.csv")
    shutil.copy(data_with_samples_2, data_dir / "data_with_samples_2.csv")
    shutil.copy(sample_metadata_file, data_dir / "sample_metadata.csv")
    shutil.copy(sample_metadata_file_2, data_dir / "sample_metadata_2.csv")
    shutil.copy(feature_metadata_file, data_dir / "feature_metadata.csv")
    return str(data_dir)


@pytest.fixture
def data_with_samples_combined(tmp_path):
    filename = tmp_path / "data_with_samples_combined.csv"
    data = textwrap.dedent(
        """
        sample,header1,header2
        sample1,1,10
        sample2,20,2
        sample3,3,30
        sample4,40,4
        sample5,5,50
        sample6,60,6
        sample7,7,70
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def feature_metadata_missing_sample_column(tmp_path):
    filename = tmp_path / "feature_metadata_missing_sample_column.csv"
    data = textwrap.dedent(
        """
        feature,metadata1,metadata2
        header1,a,2
        header3,b,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def feature_metadata_matching_sample_column_index(tmp_path):
    filename = tmp_path / "feature_metadata_matching_sample_column_index.csv"
    data = textwrap.dedent(
        """
        feature,metadata1,metadata2
        header1,a,2
        header2,b,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def data_with_metadata(tmp_path):
    filename = tmp_path / "data_with_metadata.csv"
    data = textwrap.dedent(
        """
        sample,metadata1,metadata2,header1,header2,target
        sample1,a,1,1,10,a
        sample2,b,20,20,2,b
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def data_with_unmatched_sample_column(tmp_path):
    filename = tmp_path / "file_unmatched_sample.csv"
    data = textwrap.dedent(
        """
        sample_id,header1,header2
        sample1,1,10
        sample2,20,2
        sample3,3,30
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def feature_metadata_with_missing_header(tmp_path):
    filename = tmp_path / "feature_metadata_missing_header.csv"
    data = textwrap.dedent(
        """
        feature_name,metadata1,metadata2
        header1,1,2
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def feature_metadata_with_missing_feature_column(tmp_path):
    filename = tmp_path / "feature_metadata_with_missing_feature_column.csv"
    data = textwrap.dedent(
        """
        genus,metadata1,metadata2
        header1,1,2
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def unsupported_file(tmp_path):
    filename = tmp_path / "unsupported_file.unsupported"
    data = textwrap.dedent(
        """
        header1,header2
        1,2
        10,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


class TestBioDataConfig(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, csv_file, sample_metadata_file, feature_metadata_file):
        self.csv_file = csv_file
        self.sample_metadata_file = sample_metadata_file
        self.feature_metadata_file = feature_metadata_file

    def create_config(
        self,
        data_files=None,
        sample_metadata_files=None,
        feature_metadata_files=None,
        name="test_config",
        **kwargs,
    ):
        if data_files and isinstance(data_files, (list, str)):
            if isinstance(data_files, str):
                data_files = [data_files]
            origin_metadata = _get_origin_metadata(data_files)
            data_files = DataFilesDict(
                {"train": DataFilesList(data_files, origin_metadata)}
            )
        elif isinstance(data_files, dict):
            data_files_dict = {}
            for split, files in data_files.items():
                if isinstance(files, str):
                    files = [files]
                origin_metadata = _get_origin_metadata(files)
                data_files_dict[split] = DataFilesList(files, origin_metadata)
            data_files = DataFilesDict(data_files_dict)
        return BioDataConfig(
            name=name,
            data_files=data_files,
            sample_metadata_files=sample_metadata_files,
            feature_metadata_files=feature_metadata_files,
            **kwargs,
        )

    def test_post_init_data_files_with_invalid_config_name(self):
        with self.assertRaises(datasets.builder.InvalidConfigName):
            self.create_config(data_files=self.csv_file, name="invalid|name")

    def test_post_init_data_files_with_multiple_splits_and_sample_metadata_as_list(
        self,
    ):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files={"train": self.csv_file, "test": self.csv_file},
                sample_metadata_files=[self.sample_metadata_file],
            )
        self.assertIn(
            "When data_files has multiple splits, sample_metadata_files must be a dict with matching keys.",
            str(context.exception),
        )

    def test_post_init_data_files_with_multiple_splits_and_feature_metadata_as_list(
        self,
    ):
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            feature_metadata_files=[self.feature_metadata_file],
        )
        self.assertIsInstance(config.feature_metadata_files, DataFilesDict)
        self.assertIn("train", config.feature_metadata_files)
        self.assertIn("test", config.feature_metadata_files)

    def test_post_init_sample_metadata_files_with_extra_split(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files={"train": self.csv_file},
                sample_metadata_files={
                    "train": self.sample_metadata_file,
                    "test": self.sample_metadata_file,
                },
            )
        self.assertIn(
            "Sample metadata files contain keys {'test'} which are not present in data_files.",
            str(context.exception),
        )

    def test_post_init_feature_metadata_files_with_extra_split(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files={"train": self.csv_file},
                feature_metadata_files={
                    "train": self.feature_metadata_file,
                    "test": self.feature_metadata_file,
                },
            )
        self.assertIn(
            "Feature metadata files contain keys {'test'} which are not present in data_files.",
            str(context.exception),
        )

    def test_post_init_sample_metadata_files_with_missing_split(self):
        # Should process correctly even if sample_metadata_files dict has fewer splits than data_files
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            sample_metadata_files={"train": self.sample_metadata_file},
        )
        self.assertIn("train", config.sample_metadata_files)
        self.assertNotIn("test", config.sample_metadata_files)

    def test_post_init_feature_metadata_files_with_missing_split(self):
        # Should process correctly even if feature_metadata_files dict has fewer splits than data_files
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            feature_metadata_files={"train": self.feature_metadata_file},
        )
        self.assertIn("train", config.feature_metadata_files)
        self.assertIn("test", config.feature_metadata_files)

    def test_post_init_with_metadata_dir_and_no_sample_metadata_files(self):
        config = self.create_config(
            data_files=self.csv_file,
            sample_metadata_files=None,
            sample_metadata_dir=os.path.dirname(self.sample_metadata_file),
        )
        self.assertIsInstance(config.sample_metadata_files, DataFilesDict)

    def test_post_init_with_feature_metadata_dir_and_no_feature_metadata_files(self):
        config = self.create_config(
            data_files=self.csv_file,
            feature_metadata_files=None,
            feature_metadata_dir=os.path.dirname(self.feature_metadata_file),
        )
        self.assertIsInstance(config.feature_metadata_files, DataFilesDict)

    def test_post_init_with_nonexistent_metadata_dir(self):
        with self.assertRaises(FileNotFoundError):
            self.create_config(
                data_files=self.csv_file,
                sample_metadata_files=None,
                sample_metadata_dir="/nonexistent/path",
            )

    def test_post_init_with_nonexistent_feature_metadata_dir(self):
        with self.assertRaises(FileNotFoundError):
            self.create_config(
                data_files=self.csv_file,
                feature_metadata_files=None,
                feature_metadata_dir="/nonexistent/path",
            )

    # def test_post_init_with_invalid_characters_in_file_paths(self):
    #     invalid_path = "invalid|path/sample_metadata.csv"
    #     with self.assertRaises(datasets.builder.InvalidConfigName):
    #         self.create_config(
    #             data_files=self.csv_file, sample_metadata_files=invalid_path
    #         )

    def test_post_init_with_data_files_as_none(self):
        with self.assertRaises(ValueError):
            self.create_config(
                data_files=None,
                sample_metadata_files=self.sample_metadata_file,
                feature_metadata_files=self.feature_metadata_file,
            )

    def test_post_init_with_all_none(self):
        with self.assertRaises(ValueError):
            self.create_config(
                data_files=None, sample_metadata_files=None, feature_metadata_files=None
            )

    def test_post_init_with_data_files_dict_and_sample_metadata_files_as_dict(self):
        config = self.create_config(
            data_files={"train": [self.csv_file], "test": [self.csv_file]},
            sample_metadata_files={
                "train": [self.sample_metadata_file],
                "test": [self.sample_metadata_file],
            },
        )
        self.assertIn("train", config.sample_metadata_files)
        self.assertIn("test", config.sample_metadata_files)

    def test_post_init_with_data_files_dict_and_feature_metadata_files_as_dict(self):
        config = self.create_config(
            data_files={"train": [self.csv_file], "test": [self.csv_file]},
            feature_metadata_files={
                "train": [self.feature_metadata_file],
                "test": [self.feature_metadata_file],
            },
        )
        self.assertIn("train", config.feature_metadata_files)
        self.assertIn("test", config.feature_metadata_files)

    def test_post_init_with_mismatched_shards_in_sample_metadata_files(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files={"train": [self.csv_file, self.csv_file, self.csv_file]},
                sample_metadata_files={
                    "train": [self.sample_metadata_file, self.sample_metadata_file]
                },
            )
        self.assertIn(
            "The number of sharded sample metadata files must match the number of sharded data files in split 'train'.",
            str(context.exception),
        )

    def test_post_init_with_single_data_file_and_single_sample_metadata_file(self):
        config = self.create_config(
            data_files=self.csv_file, sample_metadata_files=self.sample_metadata_file
        )
        self.assertEqual(len(config.data_files["train"]), 1)
        self.assertEqual(len(config.sample_metadata_files["train"]), 1)

    def test_post_init_with_multiple_splits_and_no_sample_metadata_files(self):
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            sample_metadata_files=None,
        )
        self.assertEqual(config.sample_metadata_files, defaultdict(list))

    def test_post_init_with_multiple_splits_and_no_feature_metadata_files(self):
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            feature_metadata_files=None,
        )
        self.assertEqual(config.feature_metadata_files, defaultdict(list))

    def test_post_init_with_sample_metadata_files_as_dict_and_shards_mismatch(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files={"train": [self.csv_file, self.csv_file]},
                sample_metadata_files={
                    "train": [
                        self.sample_metadata_file,
                        self.sample_metadata_file,
                        self.sample_metadata_file,
                    ]
                },
            )
        self.assertIn(
            "The number of sharded sample metadata files must match the number of sharded data files in split 'train'.",
            str(context.exception),
        )

    def test_post_init_with_feature_metadata_files_as_list_and_multiple_splits(self):
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            feature_metadata_files=[self.feature_metadata_file],
        )
        self.assertIsInstance(config.feature_metadata_files, DataFilesDict)
        self.assertEqual(len(config.feature_metadata_files["train"]), 1)
        self.assertEqual(len(config.feature_metadata_files["test"]), 1)

    def test_post_init_with_feature_metadata_files_as_str_and_multiple_splits(self):
        config = self.create_config(
            data_files={"train": self.csv_file, "test": self.csv_file},
            feature_metadata_files=self.feature_metadata_file,
        )
        self.assertIsInstance(config.feature_metadata_files, DataFilesDict)
        self.assertEqual(len(config.feature_metadata_files["train"]), 1)
        self.assertEqual(len(config.feature_metadata_files["test"]), 1)

    def test_post_init_with_sample_metadata_files_as_str_and_single_split(self):
        config = self.create_config(
            data_files=self.csv_file, sample_metadata_files=self.sample_metadata_file
        )
        self.assertIsInstance(config.sample_metadata_files, DataFilesDict)
        self.assertEqual(len(config.sample_metadata_files["train"]), 1)

    def test_post_init_with_sample_metadata_files_as_list_and_single_split(self):
        config = self.create_config(
            data_files=self.csv_file, sample_metadata_files=[self.sample_metadata_file]
        )
        self.assertIsInstance(config.sample_metadata_files, DataFilesDict)
        self.assertEqual(len(config.sample_metadata_files["train"]), 1)

    def test_post_init_with_sample_metadata_files_as_list_and_mismatched_shards(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files=[self.csv_file, self.csv_file, self.csv_file],
                sample_metadata_files=[
                    self.sample_metadata_file,
                    self.sample_metadata_file,
                ],
            )
        self.assertIn(
            "The number of sharded sample metadata files must match the number of sharded data files in split 'train'.",
            str(context.exception),
        )

    def test_post_init_no_data_files(self):
        with self.assertRaises(ValueError):
            self.create_config()

    def test_post_init_empty_data_files(self):
        with self.assertRaises(ValueError):
            self.create_config(data_files={})

    def test_post_init_invalid_sample_metadata_files(self):
        with self.assertRaises(FileNotFoundError):
            self.create_config(
                data_files=self.csv_file,
                sample_metadata_files="nonexistent/path/sample_metadata.csv",
            )

    def test_post_init_invalid_feature_metadata_files(self):
        with self.assertRaises(FileNotFoundError):
            self.create_config(
                data_files=self.csv_file,
                feature_metadata_files="nonexistent/path/feature_metadata.csv",
            )

    def test_post_init_feature_metadata_as_list(self):
        config = self.create_config(
            data_files=[self.csv_file, self.csv_file],
            feature_metadata_files=[
                self.feature_metadata_file,
                self.feature_metadata_file,
            ],
        )
        self.assertIsInstance(config.feature_metadata_files, DataFilesDict)

    def test_post_init_with_multi_sample_metadata_and_one_data_file(self):
        self.create_config(
            data_files=[self.csv_file],
            sample_metadata_files=[
                self.sample_metadata_file,
                self.sample_metadata_file,
            ],
        )

    def test_post_init_with_multi_data_and_one_sample_metadata_data_file(self):
        self.create_config(
            data_files=[self.csv_file, self.csv_file],
            sample_metadata_files=self.sample_metadata_file,
        )

    def test_post_init_with_nonexistent_key_in_sample_metadata_files(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files=[self.csv_file, self.csv_file],
                sample_metadata_files={
                    "nonexistent_key": [
                        self.sample_metadata_file,
                        self.sample_metadata_file,
                    ],
                },
            )
        self.assertIn(
            "Sample metadata files contain keys {'nonexistent_key'} which are not present in "
            "data_files.",
            str(context.exception),
        )

    def test_post_init_with_nonexistent_key_in_feature_metadata_files(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files=[self.csv_file, self.csv_file],
                feature_metadata_files={
                    "nonexistent_key": [
                        self.feature_metadata_file,
                        self.feature_metadata_file,
                    ],
                },
            )
        self.assertIn(
            "Feature metadata files contain keys {'nonexistent_key'} which are not present in "
            "data_files.",
            str(context.exception),
        )

    def test_post_init_with_unequal_sample_metadata_and_data_files(self):
        with self.assertRaises(ValueError) as context:
            self.create_config(
                data_files=[self.csv_file, self.csv_file],
                sample_metadata_files=[
                    self.sample_metadata_file,
                    self.sample_metadata_file,
                    self.sample_metadata_file,
                ],
            )
        self.assertIn(
            "The number of sharded sample metadata files must match the number "
            "of sharded data files in split 'train'.",
            str(context.exception),
        )

    def test_post_init_with_missing_key_in_sample_metadata_files(self):
        self.create_config(
            data_files={
                "train": [self.csv_file, self.csv_file],
                "test": [self.csv_file, self.csv_file],
            },
            sample_metadata_files={
                "train": [self.sample_metadata_file, self.sample_metadata_file],
            },
        )

    def test_post_init_with_matching_keys_in_sample_metadata_files(self):
        self.create_config(
            data_files={
                "train": [self.csv_file, self.csv_file],
                "test": [self.csv_file, self.csv_file],
            },
            sample_metadata_files={
                "train": [self.sample_metadata_file, self.sample_metadata_file],
                "test": [self.sample_metadata_file, self.sample_metadata_file],
            },
        )

    def test_post_init_with_missing_key_in_feature_metadata_files(self):
        self.create_config(
            data_files={
                "train": [self.csv_file, self.csv_file],
                "test": [self.csv_file, self.csv_file],
            },
            feature_metadata_files={
                "train": [self.feature_metadata_file, self.feature_metadata_file],
            },
        )

    def test_post_init_with_matching_keys_in_feature_metadata_files(self):
        self.create_config(
            data_files={
                "train": [self.csv_file, self.csv_file],
                "test": [self.csv_file, self.csv_file],
            },
            feature_metadata_files={
                "train": [self.feature_metadata_file, self.feature_metadata_file],
                "test": [self.feature_metadata_file, self.feature_metadata_file],
            },
        )

    def test_get_builder_kwargs_none_files(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            builder_kwargs, config_path, module_path = config._get_builder_kwargs(
                None, config.builder_kwargs
            )
            self.assertEqual(builder_kwargs, {})

    def test_get_builder_kwargs_empty_files(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {}
            builder_kwargs, config_path, module_path = config._get_builder_kwargs(
                files, config.builder_kwargs
            )
            self.assertEqual(builder_kwargs, {})

    def test_get_builder_kwargs_mixed_extensions(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {"train": [self.csv_file, "data/train.unsupported"]}
            builder_kwargs, config_path, module_path = config._get_builder_kwargs(
                files, config.builder_kwargs
            )
            self.assertIsInstance(builder_kwargs, dict)
            self.assertIn("data_files", builder_kwargs)
            self.assertIn("path", builder_kwargs)

    def test_get_builder_kwargs_valid(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {"train": [self.csv_file]}
            builder_kwargs, config_path, module_path = config._get_builder_kwargs(
                files, config.builder_kwargs
            )
            self.assertIsInstance(builder_kwargs, dict)
            self.assertIn("data_files", builder_kwargs)

    def test_get_builder_kwargs_invalid_extension(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {"train": ["data/train.unsupported"]}
            builder_kwargs, config_path, module_path = config._get_builder_kwargs(
                files, config.builder_kwargs
            )
            self.assertEqual(builder_kwargs, {})

    def test_post_init_with_unequal_sample_metadata_and_data_files_and_batch_size(self):
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            self.create_config(
                data_files={
                    "train": [self.csv_file, self.csv_file],
                },
                sample_metadata_files={
                    "train": [self.sample_metadata_file],
                },
                builder_kwargs={"batch_size": 3},
                sample_metadata_builder_kwargs={"batch_size": 2},
            )

            self.assertIn(
                "Loading sample metadata with batch_size=2 "
                "is not recommended when the number of data files is different from "
                "the number of metadata files. Consider setting\n"
                "sample_metadata_builder_kwargs={'batch_size': None}\n"
                "to load the metadata files without batching.",
                log.output[0],
            )

    def test_post_init_with_unequal_batch_size(self):
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            self.create_config(
                data_files={
                    "train": [self.csv_file, self.csv_file],
                },
                sample_metadata_files={
                    "train": [self.sample_metadata_file, self.sample_metadata_file],
                },
                builder_kwargs={"batch_size": 3},
                sample_metadata_builder_kwargs={"batch_size": 2},
            )

            self.assertIn(
                "The batch size when reading sample metadata "
                "(batch_size=2) is different from the batch size when reading data "
                "(batch_size=3). This may lead to unexpected behavior.",
                log.output[0],
            )


class TestBioData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(
        self,
        csv_file,
        jsonl_file,
        txt_file,
        npz_file,
        sample_metadata_file,
        sample_metadata_file_2,
        sample_metadata_file_combined,
        feature_metadata_file,
        multiclass,
        data_with_index_missing_sample_column,
        data_with_samples,
        data_with_samples_2,
        data_with_samples_combined,
        feature_metadata_missing_sample_column,
        feature_metadata_matching_sample_column_index,
        data_with_metadata,
        data_with_unmatched_sample_column,
        feature_metadata_with_missing_header,
        feature_metadata_with_missing_feature_column,
    ):
        self.csv_file = csv_file
        self.jsonl_file = jsonl_file
        self.txt_file = txt_file
        self.npz_file = npz_file
        self.sample_metadata_file = sample_metadata_file
        self.sample_metadata_file_2 = sample_metadata_file_2
        self.sample_metadata_file_combined = sample_metadata_file_combined
        self.feature_metadata_file = feature_metadata_file
        self.multiclass = multiclass
        self.data_with_index_missing_sample_column = (
            data_with_index_missing_sample_column
        )
        self.data_with_samples = data_with_samples
        self.data_with_samples_2 = data_with_samples_2
        self.data_with_samples_combined = data_with_samples_combined
        self.feature_metadata_missing_header = feature_metadata_missing_sample_column
        self.feature_metadata_matching_sample_column_index = (
            feature_metadata_matching_sample_column_index
        )
        self.data_with_metadata = data_with_metadata
        self.data_with_unmatched_sample_column = data_with_unmatched_sample_column
        self.feature_metadata_with_missing_header = feature_metadata_with_missing_header
        self.feature_metadata_with_missing_feature_column = (
            feature_metadata_with_missing_feature_column
        )

    def setUp(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.BioDataConfig.__post_init__"
        ):
            self.data = BioData()
            self.data.config = BioDataConfig(name="test_config")
        logging.logging.Logger.warning_once = logging.logging.Logger.warning

    def test_init(self):
        self.assertIsInstance(self.data, BioData)

    def test_init_invalid_args(self):
        with self.assertRaises(TypeError):
            BioData(unexpected_arg="unexpected")

    def test_info_custom_features(self):
        custom_features = Features({"custom_feature": Value("int64")})
        self.data.config.features = custom_features
        dataset_info = self.data._info()
        self.assertEqual(dataset_info.features, custom_features)

    def test_info_no_features(self):
        self.data.config.features = None
        dataset_info = self.data._info()
        self.assertIsNone(dataset_info.features)

    def test_set_columns_valid(self):
        data_features = ["sample", "batch", "target", "feature1", "feature2"]
        sample_metadata = pd.DataFrame({"sample": [1, 2, 3]})
        feature_metadata = pa.Table.from_pandas(
            pd.DataFrame({"features": ["feature1", "feature2"], "info": [1, 2]})
        )
        self.data._set_columns(data_features, sample_metadata, feature_metadata)
        self.assertEqual(self.data.config.sample_column, "sample")
        self.assertEqual(self.data.config.batch_column, "batch")
        self.assertEqual(self.data.config.target_column, "target")
        self.assertEqual(self.data.config.feature_column, "features")

    def test_set_columns_no_feature_metadata(self):
        data_features = ["feature1", "feature2"]
        self.data._set_columns(data_features, feature_metadata=None)
        self.assertIsNone(self.data.config.feature_column)

    def test_convert_feature_metadata_to_dict_valid(self):
        feature_metadata = pa.Table.from_pandas(
            pd.DataFrame({"features": ["A", "B", "C"], "info": [1, 2, 3]})
        )
        self.data.config.feature_column = "features"
        metadata_dict = self.data._convert_feature_metadata_to_dict(feature_metadata)
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict["A"]["info"], 1)

    def test_convert_feature_metadata_to_dict_none(self):
        feature_metadata = None
        with self.assertRaises(ValueError):
            self.data._convert_feature_metadata_to_dict(feature_metadata)

    def test_convert_feature_metadata_to_dict_empty(self):
        feature_metadata = pa.Table.from_pandas(pd.DataFrame())
        with self.assertRaises(ValueError):
            self.data._convert_feature_metadata_to_dict(feature_metadata)

    def test_convert_feature_metadata_missing_feature_column(self):
        feature_metadata = pa.Table.from_pandas(pd.DataFrame({"metadata": [1, 2, 3]}))
        self.data.config.feature_column = "feature"
        with self.assertRaises(KeyError):
            self.data._convert_feature_metadata_to_dict(feature_metadata)

    def test_split_generators_valid(self):
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.csv_file], _get_origin_metadata([self.csv_file])
                )
            }
        )
        self.data.config = BioDataConfig(data_files=data_files)
        dl_manager = unittest.mock.MagicMock()
        with patch_dataset_load():
            split_generators = self.data._split_generators(dl_manager)
        self.assertEqual(len(split_generators), 0)

    def test_split_generators_no_data(self):
        self.data.config.data_files = None
        with self.assertRaises(ValueError):
            self.data._split_generators(None)

    def test_generate_tables_csv(self):
        origin_metadata = _get_origin_metadata([self.csv_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(
                reader, [[self.csv_file]], split_name="train"
            )
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the samples column in data table. Available "
                "columns in data table: ['header1', 'header2']",
                log.output[0],
            )
            self.assertIn(
                "Could not find the batches column in data table. Available "
                "columns in data table: ['header1', 'header2']",
                log.output[1],
            )

        self.assertEqual(pa_table.num_columns, 2)
        self.assertEqual(pa_table.num_rows, 2)
        self.assertEqual(pa_table.column_names, ["header1", "header2"])
        self.assertEqual(pa_table.column("header1").to_pylist(), [1, 20])
        self.assertEqual(pa_table.column("header2").to_pylist(), [10, 2])

    def test_generate_tables_jsonl(self):
        origin_metadata = _get_origin_metadata([self.jsonl_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.jsonl_file], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Json()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()
        file = self.jsonl_file
        generator = biodata._generate_tables(
            reader,
            [[file]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file]],
            },
            feature_metadata_generator=feature_metadata_reader,
            feature_metadata_generator_kwargs={
                "files": [[self.feature_metadata_file]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertEqual(pa_table.num_columns, 8)
        self.assertEqual(pa_table.num_rows, 3)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c"],
        )
        self.assertEqual(pa_table.column("metadata2").to_pylist(), [1, 20, 3])
        self.assertEqual(pa_table.column("header1").to_pylist(), [1, 20, 3])
        self.assertEqual(pa_table.column("header2").to_pylist(), [10, 2, 30])

    def test_generate_tables_txt(self):
        origin_metadata = _get_origin_metadata([self.txt_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.txt_file], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv(sep="\t")
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()
        file = self.txt_file
        generator = biodata._generate_tables(
            reader,
            [[file]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file]],
            },
            feature_metadata_generator=feature_metadata_reader,
            feature_metadata_generator_kwargs={
                "files": [[self.feature_metadata_file]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertEqual(pa_table.num_columns, 8)
        self.assertEqual(pa_table.num_rows, 3)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c"],
        )
        self.assertEqual(pa_table.column("metadata2").to_pylist(), [1, 20, 3])
        self.assertEqual(pa_table.column("header1").to_pylist(), [1, 20, 3])
        self.assertEqual(pa_table.column("header2").to_pylist(), [10, 2, 30])

    def test_generate_tables_npz(self):
        origin_metadata = _get_origin_metadata([self.npz_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.npz_file], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = SparseReader()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()
        file = self.npz_file
        generator = biodata._generate_tables(
            reader,
            [[file]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file]],
            },
            feature_metadata_generator=feature_metadata_reader,
            feature_metadata_generator_kwargs={
                "files": [[self.feature_metadata_file]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertEqual(pa_table.num_columns, 8)
        self.assertEqual(pa_table.num_rows, 3)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c"],
        )
        self.assertEqual(pa_table.column("metadata2").to_pylist(), [1, 20, 3])
        self.assertEqual(
            pa_table.column("header1").to_pylist(),
            [0.5986584841970366, 0.15601864044243652, 0.0],
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [0.0, 0.0, 0.15599452033620265]
        )

    def test_generate_tables_multiclass_labels(self):
        origin_metadata = _get_origin_metadata([self.multiclass])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.multiclass], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(
            reader, [[self.multiclass]], split_name="train"
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertEqual(pa_table.num_columns, 4)
        self.assertEqual(pa_table.num_rows, 4)
        self.assertEqual(
            pa_table.column_names,
            ["header1", "header2", TARGET_COLUMN, TARGET_COLUMN + "_"],
        )
        self.assertEqual(
            pa_table.column(TARGET_COLUMN).to_pylist(), ["a", "b", "c", "d"]
        )

    def test_generate_tables_missing_sample_column(self):
        origin_metadata = _get_origin_metadata(
            [self.data_with_index_missing_sample_column]
        )
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.data_with_index_missing_sample_column], origin_metadata
                )
            }
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()

        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(
                reader,
                [[self.data_with_index_missing_sample_column]],
                sample_metadata_generator=sample_metadata_reader,
                sample_metadata_generator_kwargs={
                    "files": [[self.sample_metadata_file]],
                },
                feature_metadata_generator=feature_metadata_reader,
                feature_metadata_generator_kwargs={
                    "files": [[self.feature_metadata_file]],
                },
                split_name="train",
            )
            pa.concat_tables([table for _, table in generator])

            default_column_name = biodata.SAMPLE_COLUMN
            possible_col = "sample"
            which_table = "metadata"
            other_table = "data"
            msg = (
                f"A match for the {default_column_name} column was found in "
                f"{which_table} table: '{possible_col}'\n"
                f"But it was not found in {other_table} table.\n"
                "Please add or rename the column in the appropriate table.\n"
                f"Using the {default_column_name} column detected from the "
                f"{which_table} table."
            )
            self.assertIn(msg, log.output[0])

    def test_generate_tables_matching_sample_column_name(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        biodata.config.sample_column = "sample"
        reader = Csv()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()

        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file]],
            },
            feature_metadata_generator=feature_metadata_reader,
            feature_metadata_generator_kwargs={
                "files": [[self.feature_metadata_file]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(
            pa_table.column("sample").to_pylist(), ["sample1", "sample2", "sample3"]
        )

    def test_generate_tables_feature_metadata_missing_header(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_missing_header,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(
                reader,
                [[self.data_with_samples]],
                sample_metadata_generator=sample_metadata_reader,
                sample_metadata_generator_kwargs={
                    "files": [[self.sample_metadata_file]],
                },
                feature_metadata_generator=feature_metadata_reader,
                feature_metadata_generator_kwargs={
                    "files": [[self.feature_metadata_missing_header]],
                },
                split_name="train",
            )
            pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the following columns in the data table: {'header3'}",
                log.output[0],
            )

    def test_generate_tables_feature_metadata_matching_sample_column_name(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        feature_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples]],
            feature_metadata_generator=feature_metadata_reader,
            feature_metadata_generator_kwargs={
                "files": [[self.feature_metadata_file]],
            },
            split_name="train",
        )
        pa.concat_tables([table for _, table in generator])

        self.assertIn("header1", biodata.info.features)
        self.assertIn("header2", biodata.info.features)
        self.assertIn("metadata1", biodata.info.features["header1"].metadata)
        self.assertIn("metadata2", biodata.info.features["header1"].metadata)

    def test_generate_tables_with_all_data_in_one_file(self):
        origin_metadata = _get_origin_metadata([self.data_with_metadata])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_metadata], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_metadata]],
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("metadata1", pa_table.column_names)
        self.assertIn("metadata2", pa_table.column_names)
        self.assertEqual(pa_table.column("metadata1").to_pylist(), ["a", "b"])
        self.assertEqual(pa_table.column("metadata2").to_pylist(), [1, 20])

    def test_generate_tables_unmatched_sample_column(self):
        origin_metadata = _get_origin_metadata([self.data_with_unmatched_sample_column])
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.data_with_unmatched_sample_column], origin_metadata
                )
            }
        )
        biodata = BioData(
            data_files=data_files, sample_metadata_files=self.sample_metadata_file
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(
                reader,
                [[self.data_with_unmatched_sample_column]],
                sample_metadata_generator=sample_metadata_reader,
                sample_metadata_generator_kwargs={
                    "files": [[self.sample_metadata_file]],
                },
                split_name="train",
            )
            pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Two possible matches found for the samples column:\n"
                "1. 'sample' in metadata table\n"
                "2. 'sample_id' in data table\n"
                "Please rename the columns or provide the `sample_column` argument to "
                "avoid ambiguity.\n"
                "Using the samples column detected from the metadata table.",
                log.output[0],
            )

    def test_generate_tables_feature_metadata_with_missing_header(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_with_missing_header,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file]],
            },
            feature_metadata_generator=feature_metadata_reader,
            feature_metadata_generator_kwargs={
                "files": [[self.feature_metadata_with_missing_feature_column]],
            },
            split_name="train",
        )
        pa.concat_tables([table for _, table in generator])

    def test_generate_tables_feature_metadata_with_missing_feature_column(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_with_missing_feature_column,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        feature_metadata_reader = Csv()
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(
                reader,
                [[self.data_with_samples]],
                sample_metadata_generator=sample_metadata_reader,
                sample_metadata_generator_kwargs={
                    "files": [[self.sample_metadata_file]],
                },
                feature_metadata_generator=feature_metadata_reader,
                feature_metadata_generator_kwargs={
                    "files": [[self.feature_metadata_with_missing_feature_column]],
                },
                split_name="train",
            )
            pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the features column in metadata table", log.output[0]
            )

    def test_generate_tables_with_sharded_sample_metadata_and_data_files(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.data_with_samples, self.data_with_samples_2], origin_metadata
                )
            }
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=[
                self.sample_metadata_file,
                self.sample_metadata_file_2,
            ],
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples, self.data_with_samples_2]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file, self.sample_metadata_file_2]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(pa_table.num_rows, 7)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "sample5",
                "sample6",
                "sample7",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6", "batch7"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c", "d", "e", "f", "g"],
        )
        self.assertEqual(
            pa_table.column("metadata2").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header1").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [10, 2, 30, 4, 50, 6, 70]
        )

    def test_generate_tables_with_sharded_sample_metadata_only(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples_combined], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=[
                self.sample_metadata_file,
                self.sample_metadata_file_2,
            ],
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples_combined]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file, self.sample_metadata_file_2]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(pa_table.num_rows, 7)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "sample5",
                "sample6",
                "sample7",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6", "batch7"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c", "d", "e", "f", "g"],
        )
        self.assertEqual(
            pa_table.column("header1").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [10, 2, 30, 4, 50, 6, 70]
        )

    def test_generate_tables_with_sharded_data_files_only(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.data_with_samples, self.data_with_samples_2], origin_metadata
                )
            }
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file_combined,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples, self.data_with_samples_2]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file_combined]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(pa_table.num_rows, 7)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "sample5",
                "sample6",
                "sample7",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6", "batch7"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c", "d", "e", "f", "g"],
        )
        self.assertEqual(
            pa_table.column("header1").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [10, 2, 30, 4, 50, 6, 70]
        )

    def test_generate_tables_with_sharded_files_and_batch_size(self):
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.data_with_samples, self.data_with_samples_2], origin_metadata
                )
            }
        )

        sample_metadata_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.sample_metadata_file, self.sample_metadata_file_2],
                    origin_metadata,
                )
            }
        )
        biodata = BioData(
            data_files=data_files, sample_metadata_files=sample_metadata_files
        )

        biodata.INPUT_FEATURE = Abundance
        reader = Csv(batch_size=2)
        sample_metadata_reader = Csv(batch_size=2)
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples, self.data_with_samples_2]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file, self.sample_metadata_file_2]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(pa_table.num_rows, 7)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "sample5",
                "sample6",
                "sample7",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6", "batch7"],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c", "d", "e", "f", "g"],
        )
        self.assertEqual(
            pa_table.column("header1").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [10, 2, 30, 4, 50, 6, 70]
        )

    def test_generate_tables_with_missing_samples_in_sample_metadata(self):
        """
        When samples are missing in the sample metadata file, which are found in the
        data, the final table should ADD the columns from the sample metadata file
        as None values.
        """
        origin_metadata = _get_origin_metadata([self.data_with_samples])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples_combined], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples_combined]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertEqual(pa_table.num_rows, 7)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            [
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "sample5",
                "sample6",
                "sample7",
            ],
        )
        self.assertEqual(
            pa_table.column("batch").to_pylist(),
            ["batch1", "batch2", "batch3", None, None, None, None],
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c", None, None, None, None],
        )
        self.assertEqual(
            pa_table.column("header1").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [10, 2, 30, 4, 50, 6, 70]
        )
        self.assertEqual(
            pa_table.column("metadata1").to_pylist(),
            ["a", "b", "c", None, None, None, None],
        )
        self.assertEqual(
            pa_table.column("metadata2").to_pylist(), [1, 20, 3, None, None, None, None]
        )
        self.assertEqual(
            pa_table.column("header1").to_pylist(), [1, 20, 3, 40, 5, 60, 7]
        )
        self.assertEqual(
            pa_table.column("header2").to_pylist(), [10, 2, 30, 4, 50, 6, 70]
        )

    def test_generate_tables_with_missing_samples_in_data_files(self):
        """
        When samples are missing in the data file, which are found in the sample
        metadata, the final table should IGNORE the samples that are not present in
        the data file.
        """
        origin_metadata = _get_origin_metadata([self.data_with_samples_2])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.data_with_samples_2], origin_metadata)}
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file_combined,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        sample_metadata_reader = Csv()
        generator = biodata._generate_tables(
            reader,
            [[self.data_with_samples_2]],
            sample_metadata_generator=sample_metadata_reader,
            sample_metadata_generator_kwargs={
                "files": [[self.sample_metadata_file_combined]],
            },
            split_name="train",
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(pa_table.num_rows, 4)
        self.assertEqual(
            pa_table.column("sample").to_pylist(),
            ["sample4", "sample5", "sample6", "sample7"],
        )
        self.assertEqual(pa_table.column("metadata1").to_pylist(), ["d", "e", "f", "g"])
        self.assertEqual(pa_table.column("metadata2").to_pylist(), [40, 5, 60, 7])
        self.assertEqual(pa_table.column("header1").to_pylist(), [40, 5, 60, 7])
        self.assertEqual(pa_table.column("header2").to_pylist(), [4, 50, 6, 70])

    def test_abundance_data_loading_binarized(self):
        origin_metadata = _get_origin_metadata([self.multiclass])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.multiclass], origin_metadata)}
        )

        biodata = BioData(
            data_files=data_files,
            positive_labels=["a", "b"],
            negative_labels=["c", "d"],
            target_column=TARGET_COLUMN,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(
            reader, [[self.multiclass]], split_name="train"
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        assert pa_table.num_columns == 4
        assert pa_table.num_rows == 4
        assert pa_table.column_names == [
            "header1",
            "header2",
            TARGET_COLUMN,
            TARGET_COLUMN + "_",
        ]
        assert pa_table.column("header1").to_pylist() == [1, 20, 3, 40]
        assert pa_table.column("header2").to_pylist() == [10, 2, 30, 4]
        assert pa_table.column(TARGET_COLUMN + "_").to_pylist() == [1, 1, 0, 0]
        metadata = pa_table.schema.metadata[b"huggingface"].decode()
        metadata = json.loads(metadata)
        assert (
            metadata["info"]["features"][TARGET_COLUMN + "_"]["_type"]
            == "BinClassLabel"
        )
        assert metadata["info"]["features"][TARGET_COLUMN + "_"]["positive_labels"] == [
            "a",
            "b",
        ]
        assert metadata["info"]["features"][TARGET_COLUMN + "_"]["negative_labels"] == [
            "c",
            "d",
        ]
        assert metadata["info"]["features"][TARGET_COLUMN + "_"]["names"] == [
            "negative",
            "positive",
        ]

    def test_abundance_data_loading_binarized_with_missing_labels(self):
        origin_metadata = _get_origin_metadata([self.multiclass])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.multiclass], origin_metadata)}
        )

        biodata = BioData(
            data_files=data_files,
            positive_labels=["a", "b"],
            negative_labels=["c"],
            target_column=TARGET_COLUMN,
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(
            reader, [[self.multiclass]], split_name="train"
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        assert pa_table.num_columns == 4
        assert pa_table.num_rows == 4
        assert pa_table.column_names == [
            "header1",
            "header2",
            TARGET_COLUMN,
            TARGET_COLUMN + "_",
        ]
        assert pa_table.column("header1").to_pylist() == [1, 20, 3, 40]
        assert pa_table.column("header2").to_pylist() == [10, 2, 30, 4]
        assert pa_table.column(TARGET_COLUMN + "_").to_pylist() == [1, 1, 0, -1]
        metadata = pa_table.schema.metadata[b"huggingface"].decode()
        metadata = json.loads(metadata)
        assert (
            metadata["info"]["features"][TARGET_COLUMN + "_"]["_type"]
            == "BinClassLabel"
        )
        assert metadata["info"]["features"][TARGET_COLUMN + "_"]["positive_labels"] == [
            "a",
            "b",
        ]
        assert metadata["info"]["features"][TARGET_COLUMN + "_"]["negative_labels"] == [
            "c"
        ]
        assert metadata["info"]["features"][TARGET_COLUMN + "_"]["names"] == [
            "negative",
            "positive",
        ]

    def test_read_metadata_invalid_paths(self):
        metadata_files = ["nonexistent/path/metadata.csv"]
        with self.assertRaises(Exception):
            self.data._read_metadata(metadata_files)

    def test_create_features_valid(self):
        schema = pa.schema([("sample", pa.int64()), ("target", pa.float64())])
        features = self.data._create_features(schema, column_names=["sample", "target"])
        self.assertIsInstance(features, Features)
        self.assertIn("sample", features)

    def test_create_features_missing_schema(self):
        with self.assertRaises(ValueError) as context:
            self.data._create_features(
                {"invalid": {"dtype": "string", "id": None, "_type": "Value"}},
                column_names=["sample", "target"],
            )
            self.assertIn(
                "Could not find the column 'invalid' in the data table.",
                str(context.exception),
            )

    def test_create_features_schema_none(self):
        schema = None
        with self.assertRaises(ValueError):
            self.data._create_features(schema, column_names=["sample", "target"])

    def test_create_features_empty_schema(self):
        schema = {}
        with self.assertRaises(ValueError):
            self.data._create_features(schema, column_names=["sample", "target"])

    def test_create_features_invalid_feature_metadata(self):
        schema = pa.schema([("sample", pa.int64()), ("target", pa.float64())])
        feature_metadata = {"non_existant": "metadata"}
        # should still be fine, just without metadata for that feature
        self.data._create_features(
            schema,
            feature_metadata=feature_metadata,
            column_names=["sample", "target"],
        )

    def test_create_features_binary_schema_types(self):
        schema = pa.schema([("sample", pa.binary()), ("target", pa.large_binary())])
        self.data._create_features(schema, column_names=["sample", "target"])

    def test_biodata_load_dataset(self):
        load_dataset(
            "otu",
            data_files=self.data_with_samples,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
            target_column="metadata1",
        )["train"]
