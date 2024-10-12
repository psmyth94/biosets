from ...features import PeakIntensity
from ..biodata.biodata import BioData


class Maldi(BioData):
    INPUT_FEATURE = PeakIntensity
