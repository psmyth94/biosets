from dataclasses import dataclass, field

from .metadata import ValueWithMetadata

# NOTE: make sure that all these feature types are registered
# in /src/biosets/datasets/features/__init__.py !!!
## MULTIOMIC ##


@dataclass
class Expression(ValueWithMetadata):
    dtype: str = field(default="float32")
    _type: str = field(default="Expression", init=False, repr=False)


## METAGENOMICS ##


@dataclass
class Abundance(ValueWithMetadata):
    dtype: str = field(default="int64")
    _type: str = field(default="Abundance", init=False, repr=False)


## GENOMICS ##


@dataclass
class GenomicVariant(ValueWithMetadata):
    dtype: str = field(default="int8")
    _type: str = field(default="GenomicVariant", init=False, repr=False)


@dataclass
class ReadCount(ValueWithMetadata):
    dtype: str = field(default="int64")
    _type: str = field(default="ReadCount", init=False, repr=False)


@dataclass
class KmerCount(ValueWithMetadata):
    dtype: str = field(default="int64")
    _type: str = field(default="KmerCount", init=False, repr=False)


## PROTEOMICS ##


@dataclass
class PeakIntensity(ValueWithMetadata):
    dtype: str = field(default="float32")
    _type: str = field(default="PeakIntensity", init=False, repr=False)
