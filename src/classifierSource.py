from bci_essentials.classification.erp_rg_classifier import ErpRgClassifier
from bci_essentials.paradigm.p300_paradigm import P300Paradigm

class ClassifierSource:

    def __init__( self ):
        self.classifier = ErpRgClassifier()
        self.paradigm = P300Paradigm()

        self.model = None
        self.newModel = None
        self.modelName = None

        self.thereIsNewData = False
        self.needToUpdate = False


    def setup( self ):
        pass


    def getClassifier( self ):
        return self.classifier


    def getParadigm( self ):
        return self.paradigm


    def isThereNewData( self ):
        return self.thereIsNewData


    def update( self ):
        if self.needToUpdate:
            self.needToUpdate = False
            self.thereIsNewData = True
            self.classifier.clf = self.newModel
            self.model = self.newModel
            self.classifier.set_p300_clf_settings(
                n_splits=5,
                lico_expansion_factor=4,
                oversample_ratio=0,
                undersample_ratio=0,
                random_seed=35,
                remove_flats=True,
            )
