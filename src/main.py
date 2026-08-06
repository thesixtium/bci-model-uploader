import os

from sympy.benchmarks.bench_meijerint import sigma1

print(os.listdir())

from src.run_models.bciModelUploader import BciModelUploader
from src.run_models.testingMiApplication import TestingMiApplication
from src.run_models.testingSsvepApplication import TestingSsvepApplication
from src.run_models.nfcReader import NfcReader
from src.run_models.applicationDictionary import ApplicationDictionary
from src.run_models.model import Model
from src.run_models.classifier import Classifier


application_dict = ApplicationDictionary(
    {
        0: TestingMiApplication(
            name="glowstickS1",
            model=Model(
            "Schirrmeister2017.ckpt",
            ["F4", "C4", "P4", "P3", "C3", "F3"],
            {1: 'left_hand', 2: 'rest', 3: 'right_hand'},
            0.01,
            1,
            20
            ),
            params={"classifications": {1: 'left_hand', 2: 'rest', 3: 'right_hand'}},
        ),
        1: TestingSsvepApplication(
            name="ssvepS1",
            model=Model(
                "SSVEP_Model.ckpt",
                ["O1", "O2", "Oz", "POz"],
                {1: "freq_7_5hz", 2: "freq_8_57hz", 3: "freq_10hz", 4: "freq_12hz"},
                0.01,
                1,
                20
            ),
            params={
                "classifications": {
                    1: "freq_7_5hz",
                    2: "freq_8_57hz",
                    3: "freq_10hz",
                    4: "freq_12hz",
                },
                "frequencies": {1: 7.5, 2: 8.57, 3: 10.0, 4: 12.0},
            },
        ),
    }
)

am I sure the DSI channels are being read at the correct order? Like is s1
mapping correctly?

print("nfc_reader")
nfc_reader = NfcReader()

print("classifier")
classifier = Classifier()

print("bci_model_uploader")
bci_model_uploader = BciModelUploader( application_dict, nfc_reader, classifier )

print("run")
bci_model_uploader.run()