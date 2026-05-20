import time
import random

from .applicationDictionary import ApplicationDictionary
from .nfcReader import NfcReader
from .classifier import Classifier

## Todo: Care about the user number

class BciModelUploader:
    def __init__(
            self,
            application_dictionary: ApplicationDictionary,
            nfc_reader: NfcReader,
            classifier: Classifier | None
    ):
        self.application_dictionary = application_dictionary
        self.nfc_reader = nfc_reader
        self.classifier = classifier

    def run( self ):
        while True:
            if self.nfc_reader.is_new_data():
                # Get User ID and Application Number
                user_id, application_number = self.nfc_reader.get_data()
                print(f"user_id: { user_id }, application_number: { application_number }")

                # Open Specified Application
                self.application_dictionary.open_application( application_number )

                # Update Model
                if self.classifier is not None:
                    self.classifier.update_model(
                        self.application_dictionary.get_model()
                    )

                time.sleep(3)
                print("Updated!")

            if self.classifier is not None and self.classifier.is_ready():
                classification = self.classifier.get_classification()
                print( classification )
                self.application_dictionary.send_classification( classification )
            else:
                self.application_dictionary.send_classification( random.randint( 0, 1 ) )

            time.sleep(1)