import time

from applicationDictionary import ApplicationDictionary
from nfcReader import NfcReader
from taskHeadManager import TaskHeadManager


class BciModelUploader:
    def __init__(
            self,
            application_dictionary: ApplicationDictionary,
            nfc_reader: NfcReader,
            task_head_manager: TaskHeadManager
    ):
        self.application_dictionary = application_dictionary
        self.nfc_reader = nfc_reader
        self.task_head_manager = task_head_manager

        self.task_head_location = None

    def run( self ):
        while True:
            if self.nfc_reader.is_new_data():
                # Get User ID and Application Number
                user_id, application_number = self.nfc_reader.get_data()
                print(f"user_id: { user_id }, application_number: { application_number }")

                # Open Specified Application
                self.application_dictionary.open_application( application_number )

                # Get Task Header
                self.task_head_location = self.task_head_manager.get_task_head_location( user_id, application_number )

                time.sleep(1)