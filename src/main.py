import os
print(os.listdir())

from src.run_models.bciModelUploader import BciModelUploader
from src.run_models.dummyMiApplication1 import DummyMiApplication1
from src.run_models.dummyMiApplication2 import DummyMiApplication2
from src.run_models.testingMiApplication import TestingMiApplication
from src.run_models.nfcReader import NfcReader
from src.run_models.applicationDictionary import ApplicationDictionary
from src.run_models.model import Model
from src.run_models.classifier import Classifier


application_dict = ApplicationDictionary(
    {
        0: TestingMiApplication(
            name="MyApp",
            model=Model(
            "Schirrmeister2017.ckpt",
            ["F4", "C4", "P4", "P3", "C3", "F3"],
            {0: 'feet', 1: 'left_hand', 2: 'rest', 3: 'right_hand'},
            0.01,
            1,
            20
            ),
            params={"classifications": {0: 'feet', 1: 'left_hand', 2: 'rest', 3: 'right_hand'}},
        ),
        #1: DummyMiApplication2(
        #    name="MyApp",
        #    model=Model(
        #        "DSI7_Dummy.ckpt",
        #        ["F4", "C4", "P4", "P3", "C3", "F3"],
        #        {0: "left_hand", 1: "right_hand"},
        #        0.01,
        #        1,
        #        20
        #    ),
        #    params={"classifications": {0: "left_hand", 1: "right_hand"}},
        #)
    }
)

print("nfc_reader")
nfc_reader = NfcReader()

print("classifier")
classifier = Classifier()

print("bci_model_uploader")
bci_model_uploader = BciModelUploader( application_dict, nfc_reader, classifier )

print("run")
bci_model_uploader.run()