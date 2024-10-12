from ...features import Abundance
from ..biodata.biodata import BioData


class OTU(BioData):
    INPUT_FEATURE = Abundance  # must accept metadata argument
