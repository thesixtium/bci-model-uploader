from bciModelUploader import BciModelUploader
from nfcReader import NfcReader
from applicationDictionary import ApplicationDictionary
from taskHeadManager import TaskHeadManager
from applicationClass import ApplicationClass

application_dict = ApplicationDictionary(
    {
        0: ApplicationClass( "Fake Application", 4, "P300" )
    }
)
nfc_reader = NfcReader()
task_head_manager = TaskHeadManager()

bci_model_uploader = BciModelUploader( application_dict, nfc_reader, task_head_manager )
bci_model_uploader.run()