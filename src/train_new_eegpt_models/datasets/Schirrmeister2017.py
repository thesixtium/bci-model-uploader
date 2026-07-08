from src.train_new_eegpt_models.datasets.dataset import Dataset
from moabb.datasets import BNCI2014_004

class DatasetSchirrmeister2017( Dataset ):
    def __init__( self ):
        super().__init__(
            "BNCI2014_004",
            BNCI2014_004(),
            4,
            0.5,
            100,
            0,      # tmin
            4,    # tmax
            500,
            ["F4", "C4", "P4", "P3", "C3", "F3"]
        )
