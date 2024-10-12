import pyarrow as pa
import pytest

from biosets.packaged_modules.npz.npz import (
    SparseReader,
)


@pytest.fixture
def npz_file(tmp_path):
    import scipy.sparse as sp

    filename = tmp_path / "file.npz"

    n_samples = 10
    n_features = 3

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


def test_sparse_reader(npz_file):
    import scipy.sparse as sp

    n_samples = 10
    n_features = 3

    npz = SparseReader()
    npz.config.batch_size = 2
    generator = npz._generate_tables([[npz_file]])
    pa_table = pa.concat_tables([table for _, table in generator])
    generated_content = pa_table.to_pandas().values.tolist()
    test_data = (
        sp.csr_matrix(
            sp.random(
                n_samples,
                n_features,
                density=0.5,
                format="csr",
                random_state=42,
            )
        )
        .toarray()
        .tolist()
    )
    assert generated_content == test_data
