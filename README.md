<p align="center">
    $${\Huge{\textbf{\textsf{\color{#2E8B57}Bio\color{#4682B4}sets}}}}$$
    <br/>
    <br/>
</p> 
<p align="center">
    <a href="https://github.com/psmyth94/biosets/actions/workflows/ci_cd_pipeline.yml?query=branch%3Amain"><img alt="Build" src="https://github.com/psmyth94/biosets/actions/workflows/ci_cd_pipeline.yml/badge.svg?branch=main"></a>
    <a href="https://github.com/psmyth94/biosets/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/psmyth94/biosets.svg?color=blue"></a>
    <a href="https://github.com/psmyth94/biosets/tree/main/docs"><img alt="Documentation" src="https://img.shields.io/website/http/github/psmyth94/biosets/tree/main/docs.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/psmyth94/biosets/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/psmyth94/biosets.svg"></a>
    <a href="CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg"></a>
</p>

**Biosets** is a specialized library for bioinformatics data, providing the following main features:

- **Easy data loading and metadata integration**: Load and preprocess bioinformatics datasets involving samples, batches, features, and associated metadata seamlessly.
- **Automated data handling for bioinformatics tasks**: Features automatic column inference, custom feature classes with metadata, and support for classification/regression tasks.

<h3 align="center">
    <!-- <a href="YOUR_COURSE_URL"><img src="YOUR_COURSE_BANNER_URL"></a> -->
</h3>

Biosets is designed to enable the bioinformatics community to efficiently handle data involving multiple dimensions such as samples, features, and metadata.

## Features

Biosets offers powerful tools tailored for bioinformatics data:

- **Bioinformatics Specialization**: Streamlines data management specific to bioinformatics, such as handling samples, features, batches, and associated metadata.
- **Automatic Column Detection**: Infers sample, batch, input features, and target columns, simplifying downstream preprocessing.
- **Custom Data Classes**: Leverages specialized data classes (`ValueWithMetadata`, `Sample`, `Batch`, `RegressionTarget`, etc.) to manage metadata-rich bioinformatics data.
- **Polars Integration**: Optional Polars integration enables high-performance data manipulation, ideal for large datasets.
- **Flexible Task Support**: Native support for binary classification, multiclass classification, multiclass-to-binary classification, and regression, adapting to diverse bioinformatics tasks.
- **Integration with Hugging Face Datasets**: `load_dataset` function supports loading various bioinformatics formats like CSV, JSON, NPZ, and more, including metadata integration.
- **Arrow File Caching**: Uses Apache Arrow for efficient on-disk caching, enabling fast access to large datasets without memory limitations.

Biosets helps bioinformatics researchers focus on analysis rather than data handling, with seamless compatibility with Hugging Face Datasets.

## Installation

### With pip

You can install **Biosets** from PyPI:

```bash
pip install biosets
```

### With conda

Install **Biosets** via conda:

```bash
conda install -c conda-forge biosets
```

For more installation details, see the [installation documentation](YOUR_DOCS_INSTALLATION_URL).

## Usage

**Biosets** provides a straightforward API for handling bioinformatics datasets with integrated metadata management. Here's a quick example:

```python
from biosets import load_biodata

bio_data = load_dataset(
    data_files="data_with_samples.csv",
    sample_metadata_files="sample_metadata.csv",
    feature_metadata_files="feature_metadata.csv",
    target_column="metadata1",
    experiment_type="metagenomics",
    batch_column="batch",
    sample_column="sample",
    metadata_columns=["metadata1", "metadata2"],
    drop_samples=False
)["train"]
```

For further details, check the [advance usage documentation](./docs/DATA_LOADING.md).

## Main Differences Between Biosets and Hugging Face Datasets

- **Bioinformatics Focus**: While Hugging Face Datasets is a general-purpose library, Biosets is tailored for the bioinformatics domain.
- **Seamless Metadata Integration**: Biosets is built for datasets with metadata dependencies, like sample and feature metadata.
- **Automatic Column Detection**: Reduces preprocessing time with automatic inference of sample, batch, feature, and label columns.
- **Specialized Data Classes**: Biosets introduces custom classes (e.g., `Sample`, `Batch`, `ValueWithMetadata`) to enable richer data representation.

## Disclaimers

Biosets may run Python code from custom `datasets` scripts to handle specific data formats. For security, users should:

- Inspect dataset scripts prior to execution.
- Use pinned versions for any repository dependencies.

If you manage a dataset and wish to update or remove it, please open a discussion or pull request on the Community tab of hugging face's datasets page.

## BibTeX

If you'd like to cite **Biosets**, please use the following:

```bibtex
@misc{smyth2024biosets,
    title = {Biosets: A Bioinformatics Library for Data and Metadata management},
    author = {Patrick Smyth},
    year = {2024},
    url = {https://github.com/psmyth94/biosets},
    note = {A library designed to support bioinformatics data with custom features, metadata integration, and compatibility with Hugging Face Datasets.}
}
```

