from src.classifierRunner import ClassifierRunner
from src.dummyClassifierSource import DummyClassifierSource
from src.eegSource import EegSource

cs = DummyClassifierSource()
es = EegSource()
cr = ClassifierRunner( cs, es, None )
cr.run()