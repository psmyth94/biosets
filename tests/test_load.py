import os
import textwrap

import pyarrow as pa
import pytest
from biosets import load_dataset
from datasets import Features, Value
from datasets.arrow_writer import ArrowWriter
from datasets.exceptions import DatasetGenerationError

from biosets.packaged_modules.biodata.biodata import TARGET_COLUMN


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
def data_with_metadata(tmp_path):
    filename = tmp_path / "data_with_metadata.csv"
    data = textwrap.dedent(
        """
        sample,metadata1,metadata2,header1,header2,target
        sample1,a,1,1,10,a
        sample2,b,20,20,2,b
        sample3,c,3,3,30,c
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
        header3,c,3
        """
    )
    with open(filename, "w") as f:
        f.write(data)
    return str(filename)


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


def _create_unevenly_distributed_samples(data_dir):
    filename = data_dir / "samples_train_1.csv"
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

    filename = data_dir / "samples_train_2.csv"
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

    filename = data_dir / "samples_test_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample5,batch5,e,5,e
                sample6,batch6,f,60,f
                """
            )
        )

    filename = data_dir / "samples_test_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample7,batch7,g,7,g
                sample8,batch8,h,80,h
                """
            )
        )

    filename = data_dir / "samples_test_3.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,batch,metadata1,metadata2,target
                sample9,batch9,i,9,i
                sample10,batch10,j,100,j
                """
            )
        )


@pytest.fixture
def data_dir_biodata_with_mismatched_shards_arrow(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()

    _create_unevenly_distributed_samples(data_dir)

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

    filename = data_dir / "features_3.jsonl"

    output_train = os.path.join(data_dir, "data_train_1.arrow")
    with ArrowWriter(path=output_train) as writer:
        writer.write_table(
            pa.Table.from_pydict({"header1": [1, 10], "header2": [2, 20]})
        )
        num_examples, num_bytes = writer.finalize()
    assert num_examples == 2
    assert num_bytes > 0
    output_test = os.path.join(data_dir, "data_train_2.arrow")
    with ArrowWriter(path=output_test) as writer:
        writer.write_table(
            pa.Table.from_pydict({"header3": [3, 30], "header4": [4, 40]})
        )
        num_examples, num_bytes = writer.finalize()
    assert num_examples == 2
    assert num_bytes > 0
    output_test = os.path.join(data_dir, "data_test_1.arrow")
    with ArrowWriter(path=output_test) as writer:
        writer.write_table(
            pa.Table.from_pydict({"header1": [5, 50], "header2": [6, 60]})
        )
        num_examples, num_bytes = writer.finalize()

    assert num_examples == 2
    assert num_bytes > 0
    output_test = os.path.join(data_dir, "data_test_2.arrow")
    with ArrowWriter(path=output_test) as writer:
        writer.write_table(
            pa.Table.from_pydict({"header3": [7, 70], "header4": [8, 80]})
        )
        num_examples, num_bytes = writer.finalize()
    assert num_examples == 2
    assert num_bytes > 0
    output_test = os.path.join(data_dir, "data_test_3.arrow")
    with ArrowWriter(path=output_test) as writer:
        writer.write_table(
            pa.Table.from_pydict({"header5": [9, 90], "header6": [10, 100]})
        )
        num_examples, num_bytes = writer.finalize()
    assert num_examples == 2
    assert num_bytes > 0

    return str(data_dir)


@pytest.fixture
def data_dir_biodata_with_mismatched_shards_parquet(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    _create_unevenly_distributed_samples(data_dir)

    # filename = data_dir / "features_1.jsonl"
    # with open(filename, "w") as f:
    #     f.write(
    #         textwrap.dedent(
    #             """\
    #             {"feature": "header1", "metadata1": "a", "metadata2": 2}
    #             {"feature": "header2", "metadata1": "b", "metadata2": 20}
    #             """
    #         )
    #     )
    #
    # filename = data_dir / "features_2.jsonl"
    # with open(filename, "w") as f:
    #     f.write(
    #         textwrap.dedent(
    #             """\
    #             {"feature": "header3", "metadata1": "a", "metadata2": 2}
    #             {"feature": "header4", "metadata1": "b", "metadata2": 20}
    #             """
    #         )
    #     )

    output_train = os.path.join(data_dir, "data_train_1.parquet")
    table = pa.Table.from_pydict({"header1": [1, 10], "header2": [2, 20]})
    pa.parquet.write_table(table, output_train)
    output_test = os.path.join(data_dir, "data_train_2.parquet")
    table = pa.Table.from_pydict({"header3": [3, 30], "header4": [4, 40]})
    pa.parquet.write_table(table, output_test)
    output_test = os.path.join(data_dir, "data_test_1.parquet")
    table = pa.Table.from_pydict({"header1": [5, 50], "header2": [6, 60]})
    pa.parquet.write_table(table, output_test)
    output_test = os.path.join(data_dir, "data_test_2.parquet")
    table = pa.Table.from_pydict({"header3": [7, 70], "header4": [8, 80]})
    pa.parquet.write_table(table, output_test)
    output_test = os.path.join(data_dir, "data_test_3.parquet")
    table = pa.Table.from_pydict({"header5": [9, 90], "header6": [10, 100]})
    pa.parquet.write_table(table, output_test)

    return str(data_dir)


@pytest.fixture
def data_dir_biodata_with_mismatched_shards_csv(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    _create_unevenly_distributed_samples(data_dir)

    # filename = data_dir / "features_1.jsonl"
    # with open(filename, "w") as f:
    #     f.write(
    #         textwrap.dedent(
    #             """\
    #             {"feature": "header1", "metadata1": "a", "metadata2": 2}
    #             {"feature": "header2", "metadata1": "b", "metadata2": 20}
    #             """
    #         )
    #     )
    #
    # filename = data_dir / "features_2.jsonl"
    # with open(filename, "w") as f:
    #     f.write(
    #         textwrap.dedent(
    #             """\
    #             {"feature": "header3", "metadata1": "a", "metadata2": 2}
    #             {"feature": "header4", "metadata1": "b", "metadata2": 20}
    #             """
    #         )
    #     )

    filename = data_dir / "data_train_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header1,header2
                sample1,3,4
                sample2,30,40
                """
            )
        )

    filename = data_dir / "data_train_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header3,header4
                sample3,5,6
                sample4,50,60
                """
            )
        )

    filename = data_dir / "data_test_1.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header1,header2
                sample5,7,8
                sample6,70,80
                """
            )
        )

    filename = data_dir / "data_test_2.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header3,header4
                sample7,9,10
                sample8,90,100
                """
            )
        )

    filename = data_dir / "data_test_3.csv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample,header5,header6
                sample9,11,12
                sample10,110,120
                """
            )
        )

    return str(data_dir)


@pytest.fixture
def data_dir_biodata_with_mismatched_shards_tsv(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    _create_unevenly_distributed_samples(data_dir)

    # filename = data_dir / "features_1.jsonl"
    # with open(filename, "w") as f:
    #     f.write(
    #         textwrap.dedent(
    #             """\
    #             {"feature": "header1", "metadata1": "a", "metadata2": 2}
    #             {"feature": "header2", "metadata1": "b", "metadata2": 20}
    #             """
    #         )
    #     )
    #
    # filename = data_dir / "features_2.jsonl"
    # with open(filename, "w") as f:
    #     f.write(
    #         textwrap.dedent(
    #             """\
    #             {"feature": "header3", "metadata1": "a", "metadata2": 2}
    #             {"feature": "header4", "metadata1": "b", "metadata2": 20}
    #             """
    #         )
    #     )

    filename = data_dir / "data_train_1.tsv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample\theader1\theader2
                sample1\t3\t4
                sample2\t30\t40
                """
            )
        )

    filename = data_dir / "data_train_2.tsv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample\theader3\theader4
                sample3\t5\t6
                sample4\t50\t60
                """
            )
        )

    filename = data_dir / "data_test_1.tsv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample\theader1\theader2
                sample5\t7\t8
                sample6\t70\t80
                """
            )
        )

    filename = data_dir / "data_test_2.tsv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample\theader3\theader4
                sample7\t9\t10
                sample8\t90\t100
                """
            )
        )

    filename = data_dir / "data_test_3.tsv"
    with open(filename, "w") as f:
        f.write(
            textwrap.dedent(
                """
                sample\theader5\theader6
                sample9\t11\t12
                sample10\t110\t120
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
        TARGET_COLUMN: 0,
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
        TARGET_COLUMN: 0,
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
        TARGET_COLUMN: 2,
    }


@pytest.mark.parametrize("with_labels", [True, False])
@pytest.mark.parametrize("encode_labels", [True, False])
def test_load_dataset_with_dir_mismatched_shards_arrow(
    data_dir_biodata_with_mismatched_shards_arrow,
    with_labels,
    encode_labels,
):
    if with_labels:
        ds = load_dataset(
            "arrow",
            data_dir=data_dir_biodata_with_mismatched_shards_arrow,
            labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            encode_labels=encode_labels,
            add_missing_columns=True,
        )
    else:
        try:
            ds = load_dataset(
                "arrow",
                data_dir=data_dir_biodata_with_mismatched_shards_arrow,
                add_missing_columns=True,
                encode_labels=encode_labels,
            )
        except DatasetGenerationError as e:
            # ensure that the error before DatasetGenerationError is a ValueError
            assert str(e.__cause__) == (
                "Labels must be provided if multiple sample metadata files "
                "are provided. Either set `labels`, `positive_labels` "
                "and/or `negative_labels` in `load_dataset`."
            )
            return

    ds_train_item = next(iter(ds["train"]))
    assert ds["train"].shape[0] == 4
    expected_train_item = {
        "sample": "sample1",
        "batch": "batch1",
        "metadata1": "a",
        "metadata2": 2,
        "target": "a",
        "header1": 1,
        "header2": 2,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
    }
    if encode_labels:
        expected_train_item[TARGET_COLUMN] = 0
    assert ds_train_item == expected_train_item

    ds_test_item = next(iter(ds["test"]))
    expected_test_item = {
        "sample": "sample5",
        "batch": "batch5",
        "metadata1": "e",
        "metadata2": 5,
        "target": "e",
        "header1": 5,
        "header2": 6,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
    }
    if encode_labels:
        expected_test_item[TARGET_COLUMN] = 4
    assert ds["test"].shape[0] == 6
    assert ds_test_item == expected_test_item


def test_load_dataset_with_dir_mismatched_shards_parquet(
    data_dir_biodata_with_mismatched_shards_parquet,
):
    ds = load_dataset(
        "parquet",
        data_dir=data_dir_biodata_with_mismatched_shards_parquet,
        labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        add_missing_columns=True,
    )

    ds_train_item = next(iter(ds["train"]))
    assert ds["train"].shape[0] == 4
    expected_train_item = {
        "sample": "sample1",
        "batch": "batch1",
        "metadata1": "a",
        "metadata2": 2,
        "target": "a",
        "header1": 1,
        "header2": 2,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
        TARGET_COLUMN: 0,
    }
    assert ds_train_item == expected_train_item

    ds_test_item = next(iter(ds["test"]))
    expected_test_item = {
        "sample": "sample5",
        "batch": "batch5",
        "metadata1": "e",
        "metadata2": 5,
        "target": "e",
        "header1": 5,
        "header2": 6,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
        TARGET_COLUMN: 4,
    }
    assert ds["test"].shape[0] == 6
    assert ds_test_item == expected_test_item


def test_load_dataset_with_dir_mismatched_shards_csv(
    data_dir_biodata_with_mismatched_shards_csv,
):
    ds = load_dataset(
        "csv",
        data_dir=data_dir_biodata_with_mismatched_shards_csv,
        labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        add_missing_columns=True,
    )

    ds_train_item = next(iter(ds["train"]))
    assert ds["train"].shape[0] == 4
    expected_train_item = {
        "sample": "sample1",
        "batch": "batch1",
        "metadata1": "a",
        "metadata2": 2,
        "target": "a",
        "header1": 3,
        "header2": 4,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
        TARGET_COLUMN: 0,
    }
    assert ds_train_item == expected_train_item

    ds_test_item = next(iter(ds["test"]))
    expected_test_item = {
        "sample": "sample5",
        "batch": "batch5",
        "metadata1": "e",
        "metadata2": 5,
        "target": "e",
        "header1": 7,
        "header2": 8,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
        TARGET_COLUMN: 4,
    }
    assert ds["test"].shape[0] == 6
    assert ds_test_item == expected_test_item


def test_load_dataset_with_dir_mismatched_shards_tsv(
    data_dir_biodata_with_mismatched_shards_tsv,
):
    ds = load_dataset(
        "csv",
        sep="\t",
        data_dir=data_dir_biodata_with_mismatched_shards_tsv,
        labels=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        add_missing_columns=True,
    )

    ds_train_item = next(iter(ds["train"]))
    assert ds["train"].shape[0] == 4
    expected_train_item = {
        "sample": "sample1",
        "batch": "batch1",
        "metadata1": "a",
        "metadata2": 2,
        "target": "a",
        "header1": 3,
        "header2": 4,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
        TARGET_COLUMN: 0,
    }
    assert ds_train_item == expected_train_item

    ds_test_item = next(iter(ds["test"]))
    expected_test_item = {
        "sample": "sample5",
        "batch": "batch5",
        "metadata1": "e",
        "metadata2": 5,
        "target": "e",
        "header1": 7,
        "header2": 8,
        "header3": None,
        "header4": None,
        "header5": None,
        "header6": None,
        TARGET_COLUMN: 4,
    }
    assert ds["test"].shape[0] == 6
    assert ds_test_item == expected_test_item


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


def test_biodata_load_dataset_with_sparse_reader(
    npz_file, sample_metadata_file, feature_metadata_file
):
    data = load_dataset(
        "snp",
        data_files=npz_file,
        sample_metadata_files=sample_metadata_file,
        feature_metadata_files=feature_metadata_file,
        target_column="target",
    )["train"]
    pd_data = data.to_pandas()
    assert pd_data["sample"].tolist() == ["sample1", "sample2", "sample3"]
    assert pd_data["target"].tolist() == ["a", "b", "c"]
    assert set(pd_data[TARGET_COLUMN].tolist()) == set([0, 1, 2])


def test_biodata_load_dataset_with_multiple_files_and_without_labels(
    data_with_metadata, sample_metadata_file, feature_metadata_file
):
    try:
        load_dataset(
            "snp",
            data_files=[data_with_metadata, data_with_metadata],
            sample_metadata_files=[
                sample_metadata_file,
                sample_metadata_file,
            ],
            feature_metadata_files=feature_metadata_file,
            target_column="target",
        )["train"]
    except DatasetGenerationError as e:
        assert (
            "Labels must be provided if multiple sample metadata files "
            "are provided. Either set `labels`, `positive_labels` "
            "and/or `negative_labels` in `load_dataset`." in str(e.__cause__)
        )

    try:
        load_dataset(
            "snp",
            data_files=[data_with_metadata, data_with_metadata],
            feature_metadata_files=feature_metadata_file,
            target_column="target",
        )["train"]

    except DatasetGenerationError as e:
        assert (
            "Labels must be provided if multiple data files "
            "are provided and the target column is found in the "
            "data table. Either set `labels`, `positive_labels` "
            "and/or `negative_labels` in `load_dataset`." in str(e.__cause__)
        )


def test_biodata_load_dataset_with_multiple_files_and_with_labels(
    data_with_metadata, feature_metadata_file
):
    data = load_dataset(
        "snp",
        data_files=[data_with_metadata, data_with_metadata],
        feature_metadata_files=feature_metadata_file,
        labels=["a", "b"],
        target_column="target",
    )["train"]
    pd_data = data.to_pandas()
    assert len(pd_data) == 6
    assert pd_data["sample"].tolist() == [
        "sample1",
        "sample2",
        "sample3",
        "sample1",
        "sample2",
        "sample3",
    ]
    assert pd_data["target"].tolist() == ["a", "b", "c", "a", "b", "c"]
    assert set(pd_data[TARGET_COLUMN].tolist()) == set([0, 1, -1, 0, 1, -1])


def test_biodata_load_dataset_with_multiple_files_and_positive_labels(
    data_with_metadata, feature_metadata_file
):
    data = load_dataset(
        "snp",
        data_files=[data_with_metadata, data_with_metadata],
        feature_metadata_files=feature_metadata_file,
        positive_labels=["a", "b"],
        target_column="target",
    )["train"]
    pd_data = data.to_pandas()
    assert len(pd_data) == 6
    assert pd_data["sample"].tolist() == [
        "sample1",
        "sample2",
        "sample3",
        "sample1",
        "sample2",
        "sample3",
    ]
    assert pd_data["target"].tolist() == ["a", "b", "c", "a", "b", "c"]
    assert set(pd_data[TARGET_COLUMN].tolist()) == set([1, 1, 0, 1, 1, 0])


def test_biodata_load_dataset_with_multiple_files_and_negative_labels(
    data_with_metadata, feature_metadata_file
):
    data = load_dataset(
        "snp",
        data_files=[data_with_metadata, data_with_metadata],
        feature_metadata_files=feature_metadata_file,
        negative_labels=["a", "b"],
        target_column="target",
    )["train"]
    pd_data = data.to_pandas()
    assert pd_data["sample"].tolist() == [
        "sample1",
        "sample2",
        "sample3",
        "sample1",
        "sample2",
        "sample3",
    ]
    assert pd_data["target"].tolist() == ["a", "b", "c", "a", "b", "c"]
    assert set(pd_data[TARGET_COLUMN].tolist()) == set([0, 0, 1, 0, 0, 1])


def test_biodata_load_dataset_with_multiple_sample_files_and_labels(
    npz_file, sample_metadata_file, sample_metadata_file_2, feature_metadata_file
):
    data = load_dataset(
        "snp",
        data_files=[npz_file, npz_file],
        sample_metadata_files=[sample_metadata_file, sample_metadata_file],
        feature_metadata_files=feature_metadata_file,
        labels=["a", "b", "c"],
        target_column="target",
    )["train"]
    pd_data = data.to_pandas()
    assert pd_data["sample"].tolist() == [
        "sample1",
        "sample2",
        "sample3",
        "sample1",
        "sample2",
        "sample3",
    ]
    assert pd_data["target"].tolist() == ["a", "b", "c", "a", "b", "c"]
    assert set(pd_data[TARGET_COLUMN].tolist()) == set([0, 1, 2, 0, 1, 2])


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
