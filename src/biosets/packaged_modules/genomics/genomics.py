# NOTE:
from ..biodata.biodata import BioData
from ...features import GenomicVariant


class Genomics(BioData):
    INPUT_FEATURE = GenomicVariant
