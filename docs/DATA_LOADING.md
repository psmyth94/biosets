# BioSets Data Loading Guide

## Overview

BioSets is designed to facilitate the loading and manipulation of omics datasets for machine learning purposes. This guide will show you how to load omics data using BioSets from various sources including local files, in-memory data, and more.

## Table of Contents

- [Local Loading Script](#local-loading-script)
- [Local and Remote Files](#local-and-remote-files)
  - [CSV](#csv)
  - [JSON](#json)
  - [Parquet](#parquet)
  - [Arrow](#arrow)
  - [SQL](#sql)
- [Multiprocessing](#multiprocessing)
- [In-memory Data](#in-memory-data)
  - [Python Dictionary](#python-dictionary)
  - [Python List of Dictionaries](#python-list-of-dictionaries)
  - [Python Generator](#python-generator)
  - [Pandas DataFrame](#pandas-dataframe)
- [Slice Splits](#slice-splits)
- [Percent Slicing and Rounding](#percent-slicing-and-rounding)
- [Troubleshooting](#troubleshooting)
  - [Specify Features](#specify-features)

## Local Loading Script

You may have a BioSets loading script locally on your computer. In this case, load the omics dataset by passing one of the following paths to `load_dataset()`:

- The local path to the loading script file.
- The local path to the directory containing the loading script file (only if the script file has the same name as the directory).

Pass `trust_remote_code=True` to allow BioSets to execute the loading script:

```python
dataset = load_dataset("path/to/local/loading_script/loading_script.py", split="train", trust_remote_code=True)
dataset = load_dataset("path/to/local/loading_script", split="train", trust_remote_code=True)  # equivalent because the file has the same name as the directory
```

## Local and Remote Files

Omics datasets can be loaded from local files stored on your computer and from remote files. The data is most likely stored as a CSV, JSON, TXT, or Parquet file. The `load_dataset()` function can load each of these file types.

### CSV

BioSets can read an omics dataset made up of one or several CSV files (in this case, pass your CSV files as a list):

```python
from biosets import load_dataset

dataset = load_dataset("csv", data_files="my_file.csv")
```

### JSON

JSON files are loaded directly with `load_dataset()` as shown below:

```python
from biosets import load_dataset

dataset = load_dataset("json", data_files="my_file.json")
```

JSON files have diverse formats, but the most efficient format is to have multiple JSON objects; each line represents an individual row of data. For example:

```json
{"gene": "TP53", "expression": 2.5, "sample": "A"}
{"gene": "BRCA1", "expression": 5.2, "sample": "B"}
```

Another JSON format you may encounter is a nested field, in which case you’ll need to specify the `field` argument as shown in the following:

```json
{"version": "0.1.0",
 "data": [{"gene": "TP53", "expression": 2.5, "sample": "A"},
          {"gene": "BRCA1", "expression": 5.2, "sample": "B"}]
}
```

```python
from biosets import load_dataset

dataset = load_dataset("json", data_files="my_file.json", field="data")
```

To load remote JSON files via HTTP, pass the URLs instead:

```python
base_url = "https://example.com/omics_data/"

dataset = load_dataset("json", data_files={"train": base_url + "train.json", "validation": base_url + "validation.json"}, field="data")
```

### Parquet

Parquet files are stored in a columnar format, unlike row-based files like CSV. Large datasets may be stored in a Parquet file because it is more efficient and faster at returning your query.

To load a Parquet file:

```python
from biosets import load_dataset

dataset = load_dataset("parquet", data_files={'train': 'train.parquet', 'test': 'test.parquet'})
```

To load remote Parquet files via HTTP, pass the URLs instead:

```python
base_url = "https://example.com/omics_data/"

data_files = {"train": base_url + "train.parquet"}

omics_data = load_dataset("parquet", data_files=data_files, split="train")
```

### Arrow

Arrow files are stored in an in-memory columnar format, unlike row-based formats like CSV and uncompressed formats like Parquet.

To load an Arrow file:

```python
from biosets import load_dataset

dataset = load_dataset("arrow", data_files={'train': 'train.arrow', 'test': 'test.arrow'})
```

To load remote Arrow files via HTTP, pass the URLs instead:

```python
base_url = "https://example.com/omics_data/"

data_files = {"train": base_url + "train.arrow"}

omics_data = load_dataset("arrow", data_files=data_files, split="train")
```

Arrow is the file format used by BioSets under the hood, therefore you can load a local Arrow file using `Dataset.from_file()` directly:

```python
from biosets import Dataset

dataset = Dataset.from_file("data.arrow")
```

Unlike `load_dataset()`, `Dataset.from_file()` memory maps the Arrow file without preparing the dataset in the cache, saving you disk space. The cache directory to store intermediate processing results will be the Arrow file directory in that case.

For now, only the Arrow streaming format is supported. The Arrow IPC file format (also known as Feather V2) is not supported.

### SQL

Read database contents with `from_sql()` by specifying the URI to connect to your database. You can read both table names and queries:

```python
from biosets import Dataset
# load entire table
dataset = Dataset.from_sql("omics_data_table", con="sqlite:///omics_data.db")
# load from query
dataset = Dataset.from_sql("SELECT gene, expression FROM table WHERE condition='cancer'", con="sqlite:///omics_data.db")
```

## Multiprocessing

When an omics dataset is made of several files (that we call “shards”), it is possible to significantly speed up the dataset downloading and preparation step.

You can choose how many processes you’d like to use to prepare a dataset in parallel using `num_proc`. In this case, each process is given a subset of shards to prepare:

```python
from biosets import load_dataset

omics_data = load_dataset("large_omics_dataset", num_proc=8)
```

## In-memory Data

BioSets will also allow you to create a `Dataset` directly from in-memory data structures like Python dictionaries and Pandas DataFrames.

### Python Dictionary

Load Python dictionaries with `from_dict()`:

```python
from biosets import Dataset

my_dict = {"gene": ["TP53", "BRCA1", "EGFR"], "expression": [2.5, 5.2, 3.8]}

dataset = Dataset.from_dict(my_dict)
```

### Python List of Dictionaries

Load a list of Python dictionaries with `from_list()`:

```python
from biosets import Dataset

my_list = [{"gene": "TP53", "expression": 2.5}, {"gene": "BRCA1", "expression": 5.2}, {"gene": "EGFR", "expression": 3.8}]

dataset = Dataset.from_list(my_list)
```

### Python Generator

Create a dataset from a Python generator with `from_generator()`:

```python
from biosets import Dataset

def my_gen():
    for gene, expression in zip(["TP53", "BRCA1", "EGFR"], [2.5, 5.2, 3.8]):
        yield {"gene": gene, "expression": expression}

dataset = Dataset.from_generator(my_gen)
```

This approach supports loading data larger than available memory.

You can also define a sharded dataset by passing lists to `gen_kwargs`:

```python
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"line": line}

shards = [f"data{i}.txt" for i in range(32)]

ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})

ds = ds.shuffle(seed=42, buffer_size=10_000)  # shuffles the shards order + uses a shuffle buffer

from torch.utils.data import DataLoader

dataloader = DataLoader(ds.with_format("torch"), num_workers=4)  # give each worker a subset of 32/4=8 shards
```

### Pandas DataFrame

Load Pandas DataFrames with `from_pandas()`:

```python
from biosets import Dataset
import pandas as pd

df = pd.DataFrame({"gene": ["TP53", "BRCA1", "EGFR"], "expression": [2.5, 5.2, 3.8]})
dataset = Dataset.from_pandas(df)
```

## Slice Splits

You can also choose only to load specific slices of a split. There are two options for slicing a split: using strings or the `ReadInstruction` API. Strings are more compact and readable for simple cases, while `ReadInstruction` is easier to use with variable slicing parameters.

Concatenate a train and test split by:

```python
train_test_ds = biosets.load_dataset("biodata", split="train+test")
```

Select specific rows of the train split:

```python
train_10_20_ds = biosets.load_dataset("biodata", split="train[10:20]")
```

Or select a percentage of a split with:

```python
train_10pct_ds = biosets.load_dataset("biodata", split="train[:10%]")
```

Select a combination of percentages from each split:

```python
train_10_80pct_ds = biosets.load_dataset("biodata", split="train[:10%]+train[-80%:]")
```

Finally, you can even create cross-validated splits. The example below creates 10-fold cross-validated splits. Each validation dataset is a 10% chunk, and the training dataset makes up the remaining complementary 90% chunk:

```python
val_ds = biosets.load_dataset("biodata", split=[f"train[{k}%:{k+10}%]" for k in range(0, 100, 10)])
train_ds = biosets.load_dataset("biodata", split=[f"train[:{k}%]+train[{k+10}%:]" for k in range(0, 100, 10)])
```

## Percent Slicing and Rounding

The default behavior is to round the boundaries to the nearest integer for datasets where the requested slice boundaries do not divide evenly by 100. As shown below, some slices may contain more examples than others. For instance, if the following train split includes 999 records, then:

```python
# 19 records, from 500 (included) to 519 (excluded).
train_50_52_ds = biosets.load_dataset("biodata", split="train[50%:52%]")
# 20 records, from 519 (included) to 539 (excluded).
train_52_54_ds = biosets.load_dataset("biodata", split="train[52%:54%]")
```

If you want equal-sized splits, use `pct1_dropremainder` rounding instead. This treats the specified percentage boundaries as multiples of 1%.

```python
# 18 records, from 450 (included) to 468 (excluded).
train_50_52pct1_ds = biosets.load_dataset("biodata", split=datasets.ReadInstruction("train", from_=50, to=52, unit="%", rounding="pct1_dropremainder"))
# 18 records, from 468 (included) to 486 (excluded).
train_52_54pct1_ds = biosets.load_dataset("biodata", split=datasets.ReadInstruction("train",from_=52, to=54, unit="%", rounding="pct1_dropremainder"))
# Or equivalently:
train_50_52pct1_ds = biosets.load_dataset("biodata", split="train[50%:52%](pct1_dropremainder)")
train_52_54pct1_ds = biosets.load_dataset("biodata", split="train[52%:54%](pct1_dropremainder)")
```

`pct1_dropremainder` rounding may truncate the last examples in a dataset if the number of examples in your dataset doesn’t divide evenly by 100.

## Troubleshooting

Sometimes, you may get unexpected results when you load an omics dataset. One common issue you may encounter is specifying features of a dataset.

## Specify Features

When you create a dataset from local files, the features are automatically inferred by
Apache Arrow. However, the dataset’s features may not always align with your
expectations, or you may want to define the features yourself.

You can drive the inference towards specific experiment types by specifying the
`experiment_type` parameter in `load_dataset()`:

```python
dataset = load_dataset('csv', data_files=file_dict, experiment_type="rna-seq")
```

Which will create a dataset with the following features:

```python
dataset['train'].features
{'gene': Metadata(dtype='string', id=None),
 'expression': Expression(dtype='float', id=None),
 'label': ClassLabel(num_classes=3, names=['normal', 'cancerous', 'benign'], names_file=None, id=None)}
```

Say you have several columns that can be a target label, or you do not have a column called
`label` `labels`, `target`, or `targets`, you can specify the `target_column` parameter
in `load_dataset()`. This will create a `ClassLabel`, `BinClassLabel`, or `RegressionTarget`
depending on the datatype:

```python
dataset = load_dataset('csv', data_files=file_dict, target_column='target')
```

You can also binarize class labels by passing a list in `positive_labels` and/or
`negative_label`, as well as naming these two via `positive_label_name` and
`negative_label_name`:

```python
dataset = load_dataset('csv', data_files=file_dict, target_column="label", positive_labels=["cancerous"], negative_labels=["normal"], positive_label_name="Cancer", negative_label_name="Healthy")
```

You can separate information about sample metadata and feature metadata into separate
files and pass them using `sample_metadata_files` and `feature_metadata_files` parameters:

```python
dataset = load_dataset('csv', data_files=file_dict, sample_metadata_files="sample_metadata.csv", feature_metadata_files="feature_metadata.csv")
```

You can map the rows from sample metadata to the rows in the main dataset by specifying
the `sample_column` parameter:

```python
dataset = load_dataset('csv', data_files=file_dict, sample_metadata_files="sample_metadata.csv", feature_metadata_files="feature_metadata.csv", sample_column="sample_id")
```

For more advanced customizations, you can define custom features. Start by defining your
own labels with the Features class:

```python
from biosets.datasets.features import Metadata, Value, ClassLabel, Features
class_names = ["normal", "cancerous", "benign"]
omics_features = Features({
    'gene': Metadata('string'),
    'expression': Expression(dtype='float'),
    'label': ClassLabel(names=class_names)
})

```
Next, specify the features parameter in load_dataset() with the features you just created:

```python
dataset = load_dataset('csv', data_files=file_dict, delimiter=';', column_names=['gene', 'expression', 'label'], features=omics_features)
```

## Custom Features

You can also create a custom class that inherits Value and use `register_feature` in
`biosets.datasets.features` for custom features, like having a feature class with special
decoding or additional attributes like metadata. Here’s an example:

```python
from biosets.datasets.features import Value, register_feature
from dataclasses import dataclass, field

@dataclass
class CustomFeature(Value):
    metadata: dict = field(default_factory=dict)
    dtype: str = field(default="float32")
    _type: str = field(default="Expression", init=False, repr=False)

    def decode_example(self, value):
        # Custom decoding logic
        return value

# Register the custom feature
register_feature(CustomFeature)

# Use the custom feature in a dataset
custom_features = Features({'gene': CustomFeature(dtype='string', metadata={'source': 'genbank'}), 'expression': Value('float')})
dataset = load_dataset('csv', data_files=file_dict, features=custom_features)
```

With this setup, you can tailor the features to meet the specific requirements of
your omics data, ensuring that BioSets processes your data correctly and efficiently.

