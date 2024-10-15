import json
import textwrap
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from biosets.features import Abundance
from biosets.load import load_dataset
from biosets.packaged_modules.biodata.biodata import BioData, BioDataConfig
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
        1,2
        10,20
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
        {"header1": 1, "header2": 2}
        {"header1": 10, "header2": 20}
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
        header1	header2
        1	2
        10	20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def npz_file(tmp_path):
    import scipy.sparse as sp

    filename = tmp_path / "file.npz"

    n_samples = 2
    n_features = 5

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
def sample_metadata_file(tmp_path):
    filename = tmp_path / "sample_metadata.csv"
    data = textwrap.dedent(
        """
        sample,batch,metadata1,metadata2,target
        sample1,batch1,a,2,a
        sample2,batch2,b,20,b
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
        header1,a,2
        header2,b,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_multiclass(tmp_path):
    filename = tmp_path / "file_multiclass.csv"
    data = textwrap.dedent(
        """
        header1,header2,labels
        1,2,a
        10,20,b
        2,3,c
        30,40,d
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_with_index_missing_sample_column(tmp_path):
    filename = tmp_path / "file_with_index_missing_sample_column.csv"
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


@pytest.fixture
def csv_file_with_index_matching_sample_column_name(tmp_path):
    filename = tmp_path / "file_with_index_matching_sample_column_name.csv"
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
def csv_file_with_index_matching_sample_column_index(tmp_path):
    filename = tmp_path / "file_with_index_matching_sample_column_index.csv"
    data = textwrap.dedent(
        """
        sample_id,header1,header2
        sample1,1,2
        sample2,10,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_feature_metadata_missing_sample_column(tmp_path):
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
def csv_file_feature_metadata_matching_sample_column_index(tmp_path):
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
def csv_file_with_metadata(tmp_path):
    filename = tmp_path / "file_with_metadata.csv"
    data = textwrap.dedent(
        """
        sample,metadata1,metadata2,header1,header2
        sample1,1,2,1,2
        sample2,10,20,10,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_with_unmatched_sample_column(tmp_path):
    filename = tmp_path / "file_unmatched_sample.csv"
    data = textwrap.dedent(
        """
        sample_id,header1,header2
        sample1,1,2
        sample2,10,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_feature_metadata_with_missing_header(tmp_path):
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
        self, data_files=None, sample_metadata_files=None, feature_metadata_files=None
    ):
        if data_files:
            if isinstance(data_files, str):
                data_files = [data_files]
            origin_metadata = _get_origin_metadata(data_files)
            data_files = DataFilesDict(
                {"train": DataFilesList(data_files, origin_metadata)}
            )
        return BioDataConfig(
            name="test_config",
            data_files=data_files,
            sample_metadata_files=sample_metadata_files,
            feature_metadata_files=feature_metadata_files,
        )

    def test_post_init_no_data_files(self):
        with self.assertRaises(ValueError):
            config = self.create_config()

    def test_post_init_empty_data_files(self):
        with self.assertRaises(ValueError):
            config = self.create_config(data_files={})

    def test_post_init_invalid_sample_metadata_files(self):
        with self.assertRaises(FileNotFoundError):
            config = self.create_config(
                data_files=self.csv_file,
                sample_metadata_files="nonexistent/path/sample_metadata.csv",
            )

    def test_post_init_invalid_feature_metadata_files(self):
        with self.assertRaises(FileNotFoundError):
            config = self.create_config(
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
        self.assertIsInstance(config.feature_metadata_files, DataFilesList)

    def test_get_builder_kwargs_none_files(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            builder_kwargs = config._get_builder_kwargs(None)
            self.assertEqual(builder_kwargs, {})

    def test_get_builder_kwargs_empty_files(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {}
            builder_kwargs = config._get_builder_kwargs(files)
            self.assertEqual(builder_kwargs, {})

    def test_get_builder_kwargs_mixed_extensions(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {"train": [self.csv_file, "data/train.unsupported"]}
            builder_kwargs = config._get_builder_kwargs(files)
            self.assertIsInstance(builder_kwargs, dict)
            self.assertIn("data_files", builder_kwargs)
            self.assertIn("path", builder_kwargs)

    def test_get_builder_kwargs_conflicting_builder_kwargs(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            config.builder_kwargs = {"separator": ",", "sep": ";"}
            files = {"train": [self.csv_file]}
            builder_kwargs = config._get_builder_kwargs(files)
            self.assertEqual(builder_kwargs.get("separator"), ",")
            self.assertEqual(builder_kwargs.get("hf_kwargs", {}).get("sep"), ";")

    def test_get_builder_kwargs_valid(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {"train": [self.csv_file]}
            builder_kwargs = config._get_builder_kwargs(files)
            self.assertIsInstance(builder_kwargs, dict)
            self.assertIn("data_files", builder_kwargs)

    def test_get_builder_kwargs_invalid_extension(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
        ):
            config = BioDataConfig(name="test_config")
            files = {"train": ["data/train.unsupported"]}
            builder_kwargs = config._get_builder_kwargs(files)
            self.assertEqual(builder_kwargs, {})


class TestBioData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(
        self,
        csv_file,
        jsonl_file,
        txt_file,
        npz_file,
        sample_metadata_file,
        feature_metadata_file,
        csv_file_multiclass,
        csv_file_with_index_missing_sample_column,
        csv_file_with_index_matching_sample_column_name,
        csv_file_with_index_matching_sample_column_index,
        csv_file_feature_metadata_missing_sample_column,
        csv_file_feature_metadata_matching_sample_column_index,
        csv_file_with_metadata,
        csv_file_with_unmatched_sample_column,
        csv_file_feature_metadata_with_missing_header,
    ):
        self.csv_file = csv_file
        self.jsonl_file = jsonl_file
        self.txt_file = txt_file
        self.npz_file = npz_file
        self.sample_metadata_file = sample_metadata_file
        self.feature_metadata_file = feature_metadata_file
        self.csv_file_multiclass = csv_file_multiclass
        self.csv_file_with_index_missing_sample_column = (
            csv_file_with_index_missing_sample_column
        )
        self.csv_file_with_index_matching_sample_column_name = (
            csv_file_with_index_matching_sample_column_name
        )
        self.csv_file_with_index_matching_sample_column_index = (
            csv_file_with_index_matching_sample_column_index
        )
        self.csv_file_feature_metadata_missing_header = (
            csv_file_feature_metadata_missing_sample_column
        )
        self.csv_file_feature_metadata_matching_sample_column_index = (
            csv_file_feature_metadata_matching_sample_column_index
        )
        self.csv_file_with_metadata = csv_file_with_metadata
        self.csv_file_with_unmatched_sample_column = (
            csv_file_with_unmatched_sample_column
        )
        self.csv_file_feature_metadata_with_missing_header = (
            csv_file_feature_metadata_with_missing_header
        )

    def setUp(self):
        with unittest.mock.patch(
            "biosets.packaged_modules.biodata.biodata.BioDataConfig.__post_init__"
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
            generator = biodata._generate_tables(reader, [[self.csv_file]])
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the samples column in metadata table\nAvailable "
                "columns in data table: ['header1', 'header2']",
                log.output[0],
            )
            self.assertIn(
                "Could not find the batches column in metadata table\nAvailable "
                "columns in data table: ['header1', 'header2']",
                log.output[1],
            )

        self.assertEqual(pa_table.num_columns, 2)
        self.assertEqual(pa_table.num_rows, 2)
        self.assertEqual(pa_table.column_names, ["header1", "header2"])
        self.assertEqual(pa_table.column("header1").to_pylist(), [1, 10])
        self.assertEqual(pa_table.column("header2").to_pylist(), [2, 20])

    def test_generate_tables_jsonl(self):
        origin_metadata = _get_origin_metadata([self.jsonl_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.jsonl_file], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = Json()
        file = self.jsonl_file
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(reader, [[file]])
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the samples column in metadata table\nAvailable "
                "columns in data table: ['header1', 'header2']",
                log.output[0],
            )
            self.assertIn(
                "Could not find the batches column in metadata table\nAvailable "
                "columns in data table: ['header1', 'header2']",
                log.output[1],
            )

        self.assertEqual(pa_table.num_columns, 2)
        self.assertEqual(pa_table.num_rows, 2)
        self.assertEqual(pa_table.column_names, ["header1", "header2"])
        self.assertEqual(pa_table.column("header1").to_pylist(), [1, 10])
        self.assertEqual(pa_table.column("header2").to_pylist(), [2, 20])

    def test_generate_tables_txt(self):
        origin_metadata = _get_origin_metadata([self.txt_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.txt_file], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = Csv(separator="\t")
        file = self.csv_file
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(reader, [[file]])
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the samples column in metadata table\nAvailable "
                "columns in data table: ['header1', 'header2']",
                log.output[0],
            )
            self.assertIn(
                "Could not find the batches column in metadata table\nAvailable "
                "columns in data table: ['header1', 'header2']",
                log.output[1],
            )

        self.assertEqual(pa_table.num_columns, 2)
        self.assertEqual(pa_table.num_rows, 2)
        self.assertEqual(pa_table.column_names, ["header1", "header2"])
        self.assertEqual(pa_table.column("header1").to_pylist(), [1, 10])
        self.assertEqual(pa_table.column("header2").to_pylist(), [2, 20])

    def test_generate_tables_npz(self):
        origin_metadata = _get_origin_metadata([self.npz_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.npz_file], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = SparseReader()
        file = self.npz_file
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(reader, [[file]])
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "Could not find the samples column in metadata table\nAvailable "
                "columns in data table: ['0', '1', '2', '3', '4']",
                log.output[0],
            )
            self.assertIn(
                "Could not find the batches column in metadata table\nAvailable "
                "columns in data table: ['0', '1', '2', '3', '4']",
                log.output[1],
            )

        self.assertEqual(pa_table.num_columns, 5)
        self.assertEqual(pa_table.num_rows, 2)

    def test_generate_tables_multiclass_labels(self):
        origin_metadata = _get_origin_metadata([self.csv_file_multiclass])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file_multiclass], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(reader, [[self.csv_file_multiclass]])
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertEqual(pa_table.num_columns, 4)
        self.assertEqual(pa_table.num_rows, 4)
        self.assertEqual(
            pa_table.column_names, ["header1", "header2", "labels", "labels_"]
        )
        self.assertEqual(pa_table.column("labels").to_pylist(), ["a", "b", "c", "d"])

    def test_generate_tables_missing_sample_column(self):
        origin_metadata = _get_origin_metadata(
            [self.csv_file_with_index_missing_sample_column]
        )
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.csv_file_with_index_missing_sample_column], origin_metadata
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

        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            generator = biodata._generate_tables(
                reader, [[self.csv_file_with_index_missing_sample_column]]
            )
            pa.concat_tables([table for _, table in generator])

            default_column_name = biodata.SAMPLE_COLUMN
            possible_col = "sample"
            which_table = "metadata"
            other_table = "data"
            msg = (
                f"A match for the {default_column_name} column was found in "
                f"{which_table} table: {possible_col}\n"
                f"But it was not found in {other_table} table.\n"
                "Please add or rename the column in the appropriate table.\n"
                f"Using the {default_column_name} column detected from the "
                f"{which_table} table."
            )
            self.assertIn(msg, log.output[0])

    def test_generate_tables_matching_sample_column_name(self):
        origin_metadata = _get_origin_metadata(
            [self.csv_file_with_index_matching_sample_column_name]
        )
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.csv_file_with_index_matching_sample_column_name],
                    origin_metadata,
                )
            }
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
        )
        biodata.INPUT_FEATURE = Abundance
        biodata.config.sample_column = "sample"
        reader = Csv()
        generator = biodata._generate_tables(
            reader, [[self.csv_file_with_index_matching_sample_column_name]]
        )
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("sample", pa_table.column_names)
        self.assertEqual(pa_table.column("sample").to_pylist(), ["sample1", "sample2"])

    def test_generate_tables_feature_metadata_missing_header(self):
        origin_metadata = _get_origin_metadata(
            [self.csv_file_with_index_matching_sample_column_name]
        )
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.csv_file_with_index_matching_sample_column_name],
                    origin_metadata,
                )
            }
        )
        biodata = BioData(
            data_files=data_files,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.csv_file_feature_metadata_missing_header,
        )
        biodata.INPUT_FEATURE = Abundance
        biodata.config.feature_metadata_files = [
            self.csv_file_feature_metadata_missing_header
        ]
        reader = Csv()
        with self.assertLogs(
            "biosets.packaged_modules.biodata.biodata", level="WARNING"
        ) as log:
            biodata._generate_tables(
                reader, [[self.csv_file_with_index_matching_sample_column_name]]
            )
            self.assertIn(
                "Could not find the following columns in the data table: {'header3'}",
                log.output[0],
            )

    def test_generate_tables_feature_metadata_matching_sample_column_name(self):
        origin_metadata = _get_origin_metadata([self.csv_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        biodata.config.feature_metadata_files = [self.feature_metadata_file]
        biodata.config.feature_column = "feature"
        reader = Csv()
        generator = biodata._generate_tables(reader, [[self.csv_file]])
        pa.concat_tables([table for _, table in generator])

        self.assertIn("header1", biodata.info.features)
        self.assertIn("header2", biodata.info.features)
        self.assertIn("metadata1", biodata.info.features["header1"].metadata)
        self.assertIn("metadata2", biodata.info.features["header1"].metadata)

    def test_generate_tables_with_all_data_in_one_file(self):
        origin_metadata = _get_origin_metadata([self.csv_file_with_metadata])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file_with_metadata], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        biodata.config.sample_metadata_files = [
            self.sample_metadata_file,
        ]
        biodata.config.sample_column = "sample"
        reader = Csv()
        generator = biodata._generate_tables(reader, [[self.csv_file_with_metadata]])
        pa_table = pa.concat_tables([table for _, table in generator])

        self.assertIn("metadata1", pa_table.column_names)
        self.assertIn("metadata2", pa_table.column_names)
        self.assertEqual(pa_table.column("metadata1").to_pylist(), ["a", "b"])
        self.assertEqual(pa_table.column("metadata2").to_pylist(), [2, 20])

    def test_generate_tables_unmatched_sample_column(self):
        origin_metadata = _get_origin_metadata(
            [self.csv_file_with_unmatched_sample_column]
        )
        data_files = DataFilesDict(
            {
                "train": DataFilesList(
                    [self.csv_file_with_unmatched_sample_column], origin_metadata
                )
            }
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        biodata.config.sample_metadata_files = [self.sample_metadata_file]
        biodata.config.sample_column = "sample_id"
        reader = Csv()
        with self.assertLogs(level="WARNING") as log:
            generator = biodata._generate_tables(
                reader, [[self.csv_file_with_unmatched_sample_column]]
            )
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn(
                "was found in the data table but not in the metadata table",
                log.output[0],
            )

    def test_generate_tables_feature_metadata_with_missing_header(self):
        origin_metadata = _get_origin_metadata([self.csv_file])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file], origin_metadata)}
        )
        biodata = BioData(data_files=data_files)
        biodata.INPUT_FEATURE = Abundance
        biodata.config.feature_metadata_files = [
            self.csv_file_feature_metadata_with_missing_header
        ]
        reader = Csv()
        with self.assertLogs(level="WARNING") as log:
            generator = biodata._generate_tables(reader, [[self.csv_file]])
            pa_table = pa.concat_tables([table for _, table in generator])
            self.assertIn("Could not find the feature column", log.output[0])

    def test_abundance_data_loading_binarized(self):
        origin_metadata = _get_origin_metadata([self.csv_file_multiclass])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file_multiclass], origin_metadata)}
        )

        biodata = BioData(
            data_files=data_files,
            positive_labels=["a", "b"],
            negative_labels=["c", "d"],
            target_column="labels",
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(reader, [[self.csv_file_multiclass]])
        pa_table = pa.concat_tables([table for _, table in generator])

        assert pa_table.num_columns == 4
        assert pa_table.num_rows == 4
        assert pa_table.column_names == ["header1", "header2", "labels", "labels_"]
        assert pa_table.column("header1").to_pylist() == [1, 10, 2, 30]
        assert pa_table.column("header2").to_pylist() == [2, 20, 3, 40]
        assert pa_table.column("labels_").to_pylist() == [1, 1, 0, 0]
        metadata = pa_table.schema.metadata[b"huggingface"].decode()
        metadata = json.loads(metadata)
        assert metadata["info"]["features"]["labels_"]["_type"] == "BinClassLabel"
        assert metadata["info"]["features"]["labels_"]["positive_labels"] == ["a", "b"]
        assert metadata["info"]["features"]["labels_"]["negative_labels"] == ["c", "d"]
        assert metadata["info"]["features"]["labels_"]["names"] == [
            "negative",
            "positive",
        ]

    def test_abundance_data_loading_binarized_with_missing_labels(self):
        origin_metadata = _get_origin_metadata([self.csv_file_multiclass])
        data_files = DataFilesDict(
            {"train": DataFilesList([self.csv_file_multiclass], origin_metadata)}
        )

        biodata = BioData(
            data_files=data_files,
            positive_labels=["a", "b"],
            negative_labels=["c"],
            target_column="labels",
        )
        biodata.INPUT_FEATURE = Abundance
        reader = Csv()
        generator = biodata._generate_tables(reader, [[self.csv_file_multiclass]])
        pa_table = pa.concat_tables([table for _, table in generator])

        assert pa_table.num_columns == 4
        assert pa_table.num_rows == 4
        assert pa_table.column_names == ["header1", "header2", "labels", "labels_"]
        assert pa_table.column("header1").to_pylist() == [1, 10, 2, 30]
        assert pa_table.column("header2").to_pylist() == [2, 20, 3, 40]
        assert pa_table.column("labels_").to_pylist() == [1, 1, 0, -1]
        metadata = pa_table.schema.metadata[b"huggingface"].decode()
        metadata = json.loads(metadata)
        assert metadata["info"]["features"]["labels_"]["_type"] == "BinClassLabel"
        assert metadata["info"]["features"]["labels_"]["positive_labels"] == ["a", "b"]
        assert metadata["info"]["features"]["labels_"]["negative_labels"] == ["c"]
        assert metadata["info"]["features"]["labels_"]["names"] == [
            "negative",
            "positive",
        ]

    def test_read_metadata_valid(self):
        metadata_files = [self.sample_metadata_file]
        metadata = self.data._read_metadata(
            metadata_files, use_polars=False, to_arrow=True
        )
        self.assertIsInstance(metadata, pa.Table)

    def test_read_metadata_invalid(self):
        metadata_files = [str(Path("data/unsupported_metadata.unsupported"))]
        with self.assertRaises(Exception):
            self.data._read_metadata(metadata_files)

    def test_read_metadata_none_metadata_files(self):
        metadata_files = None
        with self.assertRaises(TypeError):
            self.data._read_metadata(metadata_files)

    def test_read_metadata_empty_metadata_files(self):
        metadata_files = []
        metadata = self.data._read_metadata(metadata_files)
        self.assertIsNone(metadata)

    def test_read_metadata_invalid_paths(self):
        metadata_files = ["nonexistent/path/metadata.csv"]
        with self.assertRaises(Exception):
            self.data._read_metadata(metadata_files)

    def test_create_features_valid(self):
        schema = pa.schema([("sample", pa.int64()), ("target", pa.float64())])
        features = self.data._create_features(schema)
        self.assertIsInstance(features, Features)
        self.assertIn("sample", features)

    def test_create_features_missing_schema(self):
        with self.assertRaises(ValueError):
            self.data._create_features({"invalid": {}})

    def test_create_features_schema_none(self):
        schema = None
        with self.assertRaises(ValueError):
            self.data._create_features(schema)

    def test_create_features_empty_schema(self):
        schema = {}
        with self.assertRaises(ValueError):
            self.data._create_features(schema)

    def test_create_features_invalid_feature_metadata(self):
        schema = pa.schema([("sample", pa.int64()), ("target", pa.float64())])
        feature_metadata = "invalid_metadata"
        with self.assertRaises(TypeError):
            self.data._create_features(schema, feature_metadata)

    def test_create_features_unsupported_schema_types(self):
        schema = pa.schema([("sample", pa.binary()), ("target", pa.large_binary())])
        with self.assertRaises(ValueError):
            self.data._create_features(schema)

    def test_biodata_load_dataset(self):
        load_dataset(
            "otu",
            data_files=self.csv_file,
            sample_metadata_files=self.sample_metadata_file,
            feature_metadata_files=self.feature_metadata_file,
            target_column="metadata1",
        )["train"]


# def test_biodata_load_dataset_with_sparse_reader(
#     npz_file, csv_file_sample_metadata, csv_file_feature_metadata
# ):
#     load_dataset(
#         "snp",
#         data_files=npz_file,
#         sample_metadata_files=csv_file_sample_metadata,
#         feature_metadata_files=csv_file_feature_metadata,
#         target_column="metadata1",
#     )["train"]
