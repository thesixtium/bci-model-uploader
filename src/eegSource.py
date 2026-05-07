import os

from bci_essentials.io.xdf_sources import XdfEegSource, XdfMarkerSource

class EegSource:

    def __init__(self):
        filename = os.path.join("../test/data", "p300_example.xdf")
        self.eegSource = XdfEegSource(filename)
        self.markerSource = XdfMarkerSource(filename)
        self.thereIsNewData = False

    def getSource(self):
        return self.eegSource

    def getMarker(self):
        return self.markerSource

    def isThereNewData(self):
        return self.thereIsNewData