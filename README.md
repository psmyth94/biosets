# BioSets: Dataset Creation for Biological Research

🧬 BioSets is a specialized library built on top of the `datasets` library, designed to
facilitate the loading, manipulation, and processing of biological datasets for machine
learning purposes. It supports various types of biological data, including omics
datasets such as genomics, transcriptomics, proteomics, and metabolomics, as well as
other types of tabular biological data. This library is intended to provide users
with an efficient way to work with biological data in their machine learning pipelines.

## Overview

This repository contains tools and documentation for creating biological datasets using
BioSets. The library provides capabilities for loading biological data from local
files, creating custom datasets, and handling large volumes of biological information
with ease. BioSets is particularly useful for researchers and data scientists working
in fields such as bioinformatics, systems biology, and biotechnology.

BioSets is geared towards accelerating the loading and processing of high-dimensional
data, which many machine learning libraries lack. This is achieved through efficient
handling of both sample metadata and feature metadata, enabling users to build modular
and high-performance data processing pipelines.

## Features

🧬 **Loading sample metadata and feature metadata**: BioSets provides the unique
capability to load both sample metadata and feature metadata, facilitating modular
downstream analysis pipelines. This ensures that users can easily manage and access
detailed information about each sample and feature, improving the interpretability and
flexibility of their datasets.

🧬 **Support for various biological data types**: BioSets includes predefined classes
for different biological data types, such as genomic variants, gene expression data,
clinical trial data, and OTU tables.

🧬 **Automatic Sample/Batch Detection**: BioSets can automatically detect sample and
batch information from the loaded data, making it easier to handle batch effects and
other confounding factors in downstream analyses.

🧬 **Custom dataset creation**: Create tailored datasets with custom features, metadata,
and labels.

🧬 **Integration with datasets library**: BioSets builds on the functionality provided
by the `datasets` library. For general-purpose dataset operations, users can refer to
the `datasets` library documentation. If you do not use any of the
`biosets.list_experiments()`, then it will simply act like Huggingface's `datasets`
library.

## Getting Started

To use the BioSets library, you'll need to clone the repository and install the
necessary dependencies. After setting up your environment, you can create your own
dataset by following the steps below.

### Installation

You can install BioSets using pip:

```bash
pip install biosets
```

### Creating a Biological Dataset

To create a dataset for biological research using BioSets, follow these steps:

1. **Organize Your Data**: Prepare your biological data in a structured format that
BioSets can process (e.g., directory of relevant files).

2. **Load Your Data with Metadata**: Use `load_dataset()` to load your data along with
sample metadata and feature metadata. This modular approach allows for more detailed
downstream analyses:

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
metadata easily in downstream analyses. For example, you can handle abundance data
differently based on its type:

   ```python
   from biosets.features import Abundance
   for k, v in dataset.features.items():
       if isinstance(v, Abundance):
           print(f"Processing abundance feature: {k}")
   ```

   This feature is particularly useful for modular pipeline development, where certain
   analyses or transformations are applied only to specific types of data, such as
   abundance measurements.

### Dataset Examples

#### Loading Specific Experiments

With BioSets, users are encouraged to use specific experiment types for loading data,
such as `otu`, `maldi`, `rna`, or `snp` to ensure the appropriate configuration is
applied:

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

After creating your biological dataset, you can leverage BioSets for downstream tasks
such as feature extraction, model training, or data visualization.

For more advanced usage for loading and processing biological datasets, refer to the
[dataset loading documentation](src/biosets/DATASET_LOADING.md). For building custom
datasets, refer to the [custom dataset creation documentation](src/biosets/CUSTOM_DATASETS.md).

For any additional information not covered in the BioSets documentation,
please refer to the [datasets library documentation](https://huggingface.co/docs/datasets/).

## Contributing

We welcome contributions to the BioSets project! If you have suggestions for
improvements or new features, feel free to open an issue or submit a pull request. For
major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
