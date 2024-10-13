from pathlib import Path

import pytest
from biosets.utils.logging import silence

import datasets.config

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './mock_packages')))

pytest_plugins = ["tests.fixtures.files", "tests.fixtures.fsspec"]


def pytest_collection_modifyitems(config, items):
    # Mark tests as "unit" by default if not marked as "integration" (or already marked as "unit")
    for item in items:
        if any(marker in item.keywords for marker in ["integration", "unit"]):
            continue
        item.add_marker(pytest.mark.unit)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "torchaudio_latest: mark test to run with torchaudio>=0.12"
    )


@pytest.fixture(autouse=True)
def set_test_cache_config(tmp_path_factory, monkeypatch):
    test_cache_home = tmp_path_factory.getbasetemp() / "cache"
    test_patches_cache = test_cache_home / "patches"
    test_datasets_cache = test_cache_home / "datasets"
    test_modules_cache = test_cache_home / "modules"
    monkeypatch.setattr("biosets.config.BIOSETS_CACHE_HOME", Path(test_cache_home))
    monkeypatch.setattr(
        "biosets.config.BIOSETS_PATCHES_CACHE", Path(test_patches_cache)
    )
    monkeypatch.setattr("datasets.config.HF_DATASETS_CACHE", str(test_datasets_cache))
    monkeypatch.setattr(
        "biosets.config.BIOSETS_DATASETS_CACHE",
        str(test_datasets_cache),
    )
    monkeypatch.setattr(
        "biosets.config.BIOSETS_DATASETS_CACHE", Path(test_datasets_cache)
    )
    monkeypatch.setattr("datasets.config.HF_MODULES_CACHE", str(test_modules_cache))
    test_downloaded_datasets_path = test_datasets_cache / "downloads"
    monkeypatch.setattr(
        "datasets.config.DOWNLOADED_DATASETS_PATH", str(test_downloaded_datasets_path)
    )
    monkeypatch.setattr(
        "biosets.config.DOWNLOADED_BIOSETS_PATH",
        str(test_downloaded_datasets_path),
    )
    test_extracted_datasets_path = test_datasets_cache / "downloads" / "extracted"
    monkeypatch.setattr(
        "datasets.config.EXTRACTED_DATASETS_PATH", str(test_extracted_datasets_path)
    )
    monkeypatch.setattr(
        "biosets.config.EXTRACTED_BIOSETS_PATH",
        str(test_extracted_datasets_path),
    )


# @pytest.fixture(autouse=True, scope="session")
# def disable_tqdm_output():
#     disable_progress_bar()


# @pytest.fixture(autouse=True, scope="session")
# def set_info_verbosity():
#     set_verbosity_info()


@pytest.fixture(autouse=True, scope="session")
def silence_ouput():
    silence()


@pytest.fixture(autouse=True)
def set_update_download_counts_to_false(monkeypatch):
    # don't take tests into account when counting downloads
    monkeypatch.setattr("datasets.config.HF_UPDATE_DOWNLOAD_COUNTS", False)


@pytest.fixture
def set_sqlalchemy_silence_uber_warning(monkeypatch):
    # Required to suppress RemovedIn20Warning when feature(s) are not compatible with SQLAlchemy 2.0
    # To be removed once SQLAlchemy 2.0 supported
    monkeypatch.setattr("sqlalchemy.util.deprecations.SILENCE_UBER_WARNING", True)


@pytest.fixture(autouse=True, scope="session")
def zero_time_out_for_remote_code():
    datasets.config.TIME_OUT_REMOTE_CODE = 0
