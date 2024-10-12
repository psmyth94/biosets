from biosets.features.omics import PeakIntensity

from ..biodata.biodata import BioData


class Proteomics(BioData):
    INPUT_FEATURE = PeakIntensity
