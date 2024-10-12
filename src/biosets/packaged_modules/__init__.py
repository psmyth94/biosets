import inspect

from datasets.packaged_modules import (
    _MODULE_SUPPORTS_METADATA as __MODULE_SUPPORTS_METADATA,
)
from datasets.packaged_modules import (
    _MODULE_TO_EXTENSIONS as __MODULE_TO_EXTENSIONS,
)
from datasets.packaged_modules import (
    _PACKAGED_DATASETS_MODULES as __PACKAGED_DATASETS_MODULES,
)
from datasets.packaged_modules import (
    _hash_python_lines,
)

## biosets ##
from .biodata import biodata
from .csv import csv
from .genomics import genomics, snp
from .metagenomics import metagenomics, otu
from .npz import npz
from .proteomics import maldi, proteomics

_PACKAGED_DATASETS_MODULES = __PACKAGED_DATASETS_MODULES.copy()
_MODULE_SUPPORTS_METADATA = __MODULE_SUPPORTS_METADATA.copy()
_MODULE_TO_EXTENSIONS = __MODULE_TO_EXTENSIONS.copy()

# get importable module names and hash for caching
_PACKAGED_DATASETS_MODULES["csv"] = (
    csv.__name__,
    _hash_python_lines(inspect.getsource(csv).splitlines()),
)
# get importable module names and hash for caching
_PACKAGED_DATASETS_MODULES["npz"] = (
    npz.__name__,
    _hash_python_lines(inspect.getsource(npz).splitlines()),
)
_PACKAGED_DATASETS_MODULES["biodata"] = (
    biodata.__name__,
    _hash_python_lines(inspect.getsource(biodata).splitlines()),
)
# OMIC SPECIFIC MODULES
_PACKAGED_DATASETS_MODULES["metagenomics"] = (
    metagenomics.__name__,
    _hash_python_lines(inspect.getsource(metagenomics).splitlines()),
)
_PACKAGED_DATASETS_MODULES["genomics"] = (
    genomics.__name__,
    _hash_python_lines(inspect.getsource(genomics).splitlines()),
)
_PACKAGED_DATASETS_MODULES["proteomics"] = (
    proteomics.__name__,
    _hash_python_lines(inspect.getsource(proteomics).splitlines()),
)
# DATASET SPECIFIC MODULES
_PACKAGED_DATASETS_MODULES["otu"] = (
    otu.__name__,
    _hash_python_lines(inspect.getsource(metagenomics).splitlines()),
)

_PACKAGED_DATASETS_MODULES["snp"] = (
    snp.__name__,
    _hash_python_lines(inspect.getsource(genomics).splitlines()),
)

_PACKAGED_DATASETS_MODULES["maldi"] = (
    maldi.__name__,
    _hash_python_lines(inspect.getsource(maldi).splitlines()),
)


_MODULE_SUPPORTS_METADATA.add("biodata")
_MODULE_SUPPORTS_METADATA.add("metagenomics")
_MODULE_SUPPORTS_METADATA.add("otu")
_MODULE_SUPPORTS_METADATA.add("genomics")
_MODULE_SUPPORTS_METADATA.add("snp")
_MODULE_SUPPORTS_METADATA.add("maldi")


# This is for rolling up the dataset type if there is no specific implementation for it
DATASET_NAME_TO_OMIC_TYPE = {
    "biodata": None,
    "metagenomics": "metagenomics",
    "genomics": "genomics",
    "proteomics": "proteomics",
    "metabolomics": "metabolomics",
    "otu": "metagenomics",
    "snp": "genomics",
    "maldi": "proteomics",
    "kmer": "genomics",
    "rna-seq": "transcriptomics",
    "ms1": "proteomics",
    "ms2": "proteomics",
}

DATASET_NAME_ALIAS = {
    "maldi-tof": "maldi",
    "ms/ms": "ms2",
    "ms": "ms1",
    "rna": "rna-seq",
    "asv": "otu",  # we are not treating ASV as a separate dataset type currently
}
