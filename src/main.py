from src.run_models.bciModelUploader import BciModelUploader
from src.run_models.dummyMiApplication1 import DummyMiApplication1
from src.run_models.dummyMiApplication2 import DummyMiApplication2
from src.run_models.nfcReader import NfcReader
from src.run_models.applicationDictionary import ApplicationDictionary


application_dict = ApplicationDictionary(
    {
        0: DummyMiApplication1(
            name="MyApp",
            model=None,
            params={"classifications": {0: "left_hand", 1: "right_hand"}},
        ),
        1: DummyMiApplication2(
            name="MyApp",
            model=None,
            params={"classifications": {0: "left_hand", 1: "right_hand"}},
        )
    }
)

nfc_reader = NfcReader()

classifier = None

bci_model_uploader = BciModelUploader( application_dict, nfc_reader, classifier )
bci_model_uploader.run()