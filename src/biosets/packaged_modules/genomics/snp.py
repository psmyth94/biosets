from ...features import GenomicVariant
from ..biodata.biodata import BioData


class SNP(BioData):
    INPUT_FEATURE = GenomicVariant
