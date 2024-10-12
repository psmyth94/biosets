import json
import textwrap

import pyarrow as pa
import pytest
from datasets.data_files import DataFilesDict, DataFilesList, _get_origin_metadata

from biosets.features import Abundance
from biosets.load import load_dataset
from biosets.packaged_modules.biodata.biodata import BioData
from biosets.packaged_modules.csv.csv import Csv


@pytest.fixture
def csv_file(tmp_path):
    filename = tmp_path / "file.csv"
    data = textwrap.dedent(
        """\
        header1,header2
        1,2
        10,20
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_multiclass(tmp_path):
    filename = tmp_path / "file.csv"
    data = textwrap.dedent(
        """\
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


@pytest.fixture(
    params=[
        "missing_sample_column",
        "matching_sample_column_name",
        "matching_sample_column_index",
    ]
)
def csv_file_with_index(tmp_path, request):
    filename = tmp_path / f"file_with_index_{request.param}.csv"
    if request.param == "missing_sample_column":
        data = textwrap.dedent(
            """\
            header1,header2
            1,2
            10,20
            """
        )
    elif request.param == "matching_sample_column_name":
        data = textwrap.dedent(
            """\
            sample,header1,header2
            sample1,1,2
            sample2,10,20
            """
        )
    else:
        data = textwrap.dedent(
            """\
            sample_id,header1,header2
            sample1,1,2
            sample2,10,20
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


@pytest.fixture(
    params=[
        "missing_sample_column",
        "matching_sample_column_name",
        "matching_sample_column_index",
    ]
)
def csv_file_feature_metadata(tmp_path, request):
    filename = tmp_path / f"feature_metadata_{request.param}.csv"
    if request.param == "non_matching_feature_values":
        data = textwrap.dedent(
            """\
            feature,metadata1,metadata2
            header1,a,2
            header3,b,20
            """
        )
    else:
        data = textwrap.dedent(
            """\
            feature,metadata1,metadata2
            header1,a,2
            header2,b,20
            """
        )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture(params=["matching_sample_values"])
def csv_file_sample_metadata(tmp_path, request):
    filename = tmp_path / f"sample_metadata_{request.param}.csv"
    if request.param == "non_matching_sample_values":
        data = textwrap.dedent(
            """\
            sample,metadata1,metadata2
            sample1,a,2
            sample3,b,20
            """
        )
    else:
        data = textwrap.dedent(
            """\
            sample,metadata1,metadata2
            sample1,a,2
            sample2,b,20
            """
        )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


@pytest.fixture
def csv_file_with_metadata(tmp_path):
    filename = tmp_path / "file_with_metadata.csv"
    data = textwrap.dedent(
        """\
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
    filename = tmp_path / "file.csv"
    data = textwrap.dedent(
        """\
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
    filename = tmp_path / "feature_metadata.csv"
    data = textwrap.dedent(
        """\
        feature_name,metadata1,metadata2
        header1,1,2
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


def test_biodata_csv_file(csv_file):
    origin_metadata = _get_origin_metadata([csv_file])
    data_files = DataFilesDict({"train": DataFilesList([csv_file], origin_metadata)})
    biodata = BioData(data_files=data_files)
    biodata.INPUT_FEATURE = Abundance
    csv_generator = Csv()
    generator = biodata._generate_tables(csv_generator, [[csv_file]])
    pa_table = pa.concat_tables([table for _, table in generator])

    assert pa_table.num_columns == 2
    assert pa_table.num_rows == 2
    assert pa_table.column_names == ["header1", "header2"]
    assert pa_table.column("header1").to_pylist() == [1, 10]
    assert pa_table.column("header2").to_pylist() == [2, 20]


def test_biodata_csv_file_with_metadata(
    csv_file_with_index, csv_file_sample_metadata, csv_file_feature_metadata
):
    origin_metadata = _get_origin_metadata([csv_file_with_index])
    data_files = DataFilesDict(
        {"train": DataFilesList([csv_file_with_index], origin_metadata)}
    )
    biodata = BioData(
        data_files=data_files,
        sample_metadata_files=[csv_file_sample_metadata],
        feature_metadata_files=[csv_file_feature_metadata],
    )
    biodata.INPUT_FEATURE = Abundance
    csv_generator = Csv()
    generator = biodata._generate_tables(csv_generator, [[csv_file_with_index]])
    pa_table = pa.concat_tables([table for _, table in generator])
    metadata = json.loads(pa_table.schema.metadata[b"huggingface"].decode())
    features = metadata["info"]["features"]
    assert pa_table.num_rows == 2
    assert features["header1"]["_type"] == "Abundance"


def test_abundance_data_loading_binarized(csv_file_multiclass):
    origin_metadata = _get_origin_metadata([csv_file_multiclass])
    data_files = DataFilesDict(
        {"train": DataFilesList([csv_file_multiclass], origin_metadata)}
    )

    biodata = BioData(
        data_files=data_files,
        positive_labels=["a", "b"],
        negative_labels=["c", "d"],
        target_column="labels",
    )
    biodata.INPUT_FEATURE = Abundance
    csv_generator = Csv()
    generator = biodata._generate_tables(csv_generator, [[csv_file_multiclass]])
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
    assert metadata["info"]["features"]["labels_"]["names"] == ["negative", "positive"]


def test_abundance_data_loading_binarized_with_missing_labels(csv_file_multiclass):
    origin_metadata = _get_origin_metadata([csv_file_multiclass])
    data_files = DataFilesDict(
        {"train": DataFilesList([csv_file_multiclass], origin_metadata)}
    )

    biodata = BioData(
        data_files=data_files,
        positive_labels=["a", "b"],
        negative_labels=["c"],
        target_column="labels",
    )
    biodata.INPUT_FEATURE = Abundance
    csv_generator = Csv()
    generator = biodata._generate_tables(csv_generator, [[csv_file_multiclass]])
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
    assert metadata["info"]["features"]["labels_"]["names"] == ["negative", "positive"]


def test_biodata_load_dataset(
    csv_file_with_index, csv_file_sample_metadata, csv_file_feature_metadata
):
    load_dataset(
        "otu",
        data_files=csv_file_with_index,
        sample_metadata_files=csv_file_sample_metadata,
        feature_metadata_files=csv_file_feature_metadata,
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
