__all__ = [
    "Batch",
    "Metadata",
    "Sample",
    "Abundance",
    "PeakIntensity",
    "ReadCount",
    "KmerCount",
    "GenomicVariant",
    "RegressionTarget",
    "ValueWithMetadata",
    "BinClassLabel",
]

import warnings

from datasets.features.features import ClassLabel, register_feature

from .metadata import Batch, Metadata, Sample, ValueWithMetadata
from .omics import (
    Abundance,
    Expression,
    GenomicVariant,
    KmerCount,
    PeakIntensity,
    ReadCount,
)
from .targets import BinClassLabel, RegressionTarget

NUMERIC_FEATURES = (Abundance, ReadCount, KmerCount, Expression, PeakIntensity)
CATEGORICAL_FEATURES = (
    GenomicVariant,
    Batch,
    Sample,
    ClassLabel,
    BinClassLabel,
)
METADATA_FEATURE_TYPES = (
    Sample,
    Batch,
    ClassLabel,
    BinClassLabel,
    RegressionTarget,
    Metadata,
)

METADATA_FEATURE_TYPES_NOT_TARGET = (
    Sample,
    Batch,
    Metadata,
)

TARGET_FEATURE_TYPES = (
    RegressionTarget,
    ClassLabel,
    BinClassLabel,
)

FEATURES_WITH_METADATA = (
    ValueWithMetadata,
    Abundance,
    PeakIntensity,
    GenomicVariant,
    Expression,
    ReadCount,
    KmerCount,
)

# register all features
warnings.filterwarnings("ignore", category=UserWarning)
register_feature(Sample, "Sample")
register_feature(ReadCount, "ReadCount")
register_feature(KmerCount, "KmerCount")
register_feature(Abundance, "Abundance")
register_feature(PeakIntensity, "PeakIntensity")
register_feature(Batch, "Batch")
register_feature(RegressionTarget, "RegressionTarget")
register_feature(Metadata, "Metadata")
register_feature(GenomicVariant, "GenomicVariant")
register_feature(Expression, "Expression")
register_feature(ValueWithMetadata, "ValueWithMetadata")
register_feature(BinClassLabel, "BinClassLabel")
warnings.filterwarnings("default", category=UserWarning)
