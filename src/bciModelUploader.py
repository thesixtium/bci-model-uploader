from .applicationClass import ApplicationClass
from .nfcReader import NfcReader

class BciModelUploader:
    def __init__(
            self,
            application_dict: dict[int, ApplicationClass],
            nfc_reader: NfcReader
    ):
        self.application_dict = application_dict
        self.nfc_reader = nfc_reader