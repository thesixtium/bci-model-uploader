import threading
from time import sleep

from .classifierSource import ClassifierSource
from .eegSource import EegSource

from bci_essentials.io.messenger import Messenger
from bci_essentials.data_tank.data_tank import DataTank
from bci_essentials.bci_controller import BciController

class ClassifierRunner:

    def __init__(self, classifierSource: ClassifierSource, eegSource: EegSource, messenger: Messenger):
        self.eegSource = eegSource
        self.classifierSource = classifierSource
        self.dataTank = DataTank()
        self.messenger = messenger
        self.bci_controller = None
        self.stop_event = threading.Event()
        self.exception = None

    def setup_bci_controller(self):
        is_none = self.bci_controller is None
        has_new_classifier = self.classifierSource.isThereNewData()
        has_new_eeg = self.eegSource.isThereNewData()

        if is_none or has_new_classifier or has_new_eeg:
            print("CHANGING MODEL")
            print(self.classifierSource.modelName)

            self.bci_controller = BciController(
                eeg_source=self.eegSource.getSource(),
                marker_source=self.eegSource.getMarker(),
                paradigm=self.classifierSource.getParadigm(),
                classifier=self.classifierSource.getClassifier(),
                data_tank=self.dataTank,
                messenger=self.messenger
            )

            self.bci_controller.setup(online=True, train_complete=True)

            self.bci_controller.event_timestamp_buffer = []
            self.bci_controller.event_marker_buffer = []

            self.classifierSource.thereIsNewData = False
            self.eegSource.thereIsNewData = False

    def run(self):
        self.thread = threading.Thread(target=self.step_loop)
        self.thread.start()

        while self.thread.is_alive():
            if self.exception:
                print("RAISING EXCEPTION")
                raise self.exception

            sleep(0.1)

    def step_loop(self):
        while not self.stop_event.is_set():
            try:
                self.setup_bci_controller()
                #self.bci_controller.step()
            except Exception as e:
                self.shutdown()
                self.exception = Exception(f"Error in Bessy processing loop: {e}")

            sleep(0.1)

    def set_stop(self):
        '''
        Tell the processing loop to stop execution
        '''
        if self.thread:
            self.stop_event.set()
            self.bci_controller = None
            self.thread.join()

    def shutdown(self):
        self.stop_event.set()