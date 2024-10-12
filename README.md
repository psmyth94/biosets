# BioSets: Dataset Creation for Biological Research

Please note that this project is in the early stages of development. The documentation
and features are subject to change.

## Overview

BioSets is a library built on top of the `datasets` library for loading, manipulating,
and processing biological datasets for machine learning purposes. It supports genomics,
transcriptomics, proteomics, metabolomics, and other types of biological data.

This repository contains tools and documentation for creating biological datasets using
BioSets. The library loads biological data from local files, creates custom datasets,
and handles large volumes of biological information. BioSets is intended for
researchers and data scientists in bioinformatics, systems biology, and biotechnology.

## Features

🧬 **Loading sample metadata and feature metadata**: BioSets loads both sample
metadata and feature metadata.

🧬 **Support for various biological data types**: Includes predefined classes for
genomic variants, gene expression data, clinical trial data, and OTU tables.

🧬 **Automatic Sample/Batch Detection**: Automatically detects sample and batch
information from the loaded data to handle batch effects and confounding factors.

🧬 **Custom dataset creation**: Create custom datasets with specific features,
metadata, and labels.

🧬 **Integration with datasets library**: BioSets builds on the `datasets` library's
functionality. Note that if `path` is not a value found in `biosets.list_experiments()`,
it acts like Huggingface's `datasets` library.

## Getting Started

To use the BioSets library, clone the repository and install the necessary
dependencies. After setting up your environment, create your dataset by following the
steps below.

### Installation

Install BioSets using pip:

```bash
pip install biosets
```

### Creating a Biological Dataset

To create a dataset for biological research using BioSets, follow these steps:

1. **Organize Your Data**: Prepare your biological data in a structured format that
BioSets can process (e.g., directory of relevant files).

2. **Load Your Data with Metadata**: Use `load_dataset()` to load your data along with
sample metadata and feature metadata:

   ```python
   from biosets import load_dataset

   dataset = load_dataset(
       "snp",
       data_files="/path/to/snp_data.csv",
       sample_metadata_files="/path/to/sample_metadata.csv",
       feature_metadata_files="/path/to/feature_metadata.csv",
   )
   ```

3. **Utilize Metadata for Analysis**: The loaded dataset allows you to access and use
metadata in downstream analyses. For example, you can handle abundance data differently
based on its type:

   ```python
   from biosets.features import Abundance
   for k, v in dataset.features.items():
       if isinstance(v, Abundance):
           print(f"Processing abundance feature: {k}")
   ```

### Dataset Examples

#### Loading Specific Experiments

Use specific experiment types for loading data, such as `otu`, `maldi`, `rna`, or `snp`
to ensure the appropriate configuration is applied:

🧬 **OTU Data**

  ```python
  dataset = load_dataset("otu", data_files="/path/to/otu_data.csv")
  ```

🧬 **RNA Data**

  ```python
  dataset = load_dataset("rna", data_files="/path/to/rna_data.csv")
  ```

🧬 **SNP Data**

  ```python
  dataset = load_dataset("snp", data_files="/path/to/snp_data.csv")
  ```

### Next Steps

After creating your biological dataset, you can use BioSets for feature extraction, model
training, or data visualization.

For more advanced usage, refer to the [dataset loading
documentation](src/biosets/DATASET_LOADING.md). For building custom datasets, refer to
the [custom dataset creation documentation](src/biosets/CUSTOM_DATASETS.md).

For any additional information, refer to the [datasets library
documentation](https://huggingface.co/docs/datasets/).

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features,
open an issue or submit a pull request. For major changes, open an issue first to
discuss it.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
