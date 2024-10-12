# NOTE:
from ..biodata.biodata import BioData
from ...features import Abundance


class Metagenomics(BioData):
    INPUT_FEATURE = Abundance  # must accept metadata argument
