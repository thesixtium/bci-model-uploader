from src.classifierSource import ClassifierSource


class DummyClassifierSource(ClassifierSource):
    def __init__(self):
        super().__init__()

        self.thereIsNewData = True