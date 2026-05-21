import time
import torch
import serial
import threading
import subprocess
import collections
import serial.tools.list_ports

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from src.core.genericEEGPTModel import GenericEEGPTModel
from src.run_models.model import Model
from src.core.getLibPaths import GetLibPaths


class Classifier:
    def __init__( self, window_size=1024 ):
        self.current_classification = 0
        self.current_confidence = 0

        self.__window_size = window_size
        self.__class_names = {}
        self.__number_of_channels = None
        self.__thread = None
        self.__running = False
        self.__model = None
        self.__port = None
        self.__dsi2lsl_process = None
        self.__glp = GetLibPaths()

        self.find_dsi7_port()
        print("Running DSI2LSL")
        self.run_dsi2lsl()
        time.sleep(5)

        print("Getting LSL Stream")
        self.buffer = collections.deque( maxlen=self.__window_size )
        streams = resolve_byprop( "type", "EEG" )
        self.inlet = StreamInlet( streams[0] )

    def is_ready(self):
        return self.__model is not None

    def find_dsi7_port(self):
        ports = list(serial.tools.list_ports.comports())
        print(f"Ports: {ports}")
        for p in ports:
            if "Silicon Labs" in p.description:
                self.__port = p.name
        print(f"DSI-7: {self.__port}")

    def run_dsi2lsl(self):
        try:
            dsi2lsl_path = self.__glp.get_dsi2lsl_path() / "dsi2lsl.exe"
            self.__dsi2lsl_process = subprocess.Popen(
                [
                    dsi2lsl_path,
                    f'port={self.__port}',
                    'lsl-stream-name=DSI7',
                    'montage=F4,C4,S1,S3,C3,F3'
                ]
            )
        except subprocess.CalledProcessError as e:
            print(f"DSI2LSL failed with error code {e.returncode}")
            raise Exception(f"Error running DSI2LSL: {e}")

    def update_model(
            self,
            model: Model
    ):
        # stop thread
        self.__running = False
        if self.__thread is not None:
            self.__thread.join()

        self.__number_of_channels = len( model.get_use_channels_names() )
        self.__class_names = model.get_output_class_names()

        # update model
        self.__model = GenericEEGPTModel(
            load_path= self.__glp.get_checkpoints_path() / model.get_model_name(),
            use_channels_names=model.get_use_channels_names(),
            output_classes=model.get_output_classes(),
            max_lr=model.get_max_lr(),
            steps_per_epoch=model.get_steps_per_epoch(),
            max_epochs=model.get_max_epochs()
        )
        self.__model.eval()

        # start thread
        self.__running = True
        self.__thread = threading.Thread( target=self.__classifier_step_loop )
        self.__thread.start()

    def __classifier_step_loop(self):
        print("Step")
        while self.__running:
            chunk, timestamps = self.inlet.pull_chunk()
            if not timestamps:
                continue

            # chunk shape from LSL: [n_new_samples, n_channels]
            for sample in chunk:
                self.buffer.append(sample[:self.__number_of_channels])

            # once we have a full window, run inference
            if len(self.buffer) == self.__window_size:
                # shape: [WINDOW_SIZE, CHANNELS] -> [1, CHANNELS, WINDOW_SIZE]
                window = np.array(self.buffer, dtype=np.float32)
                window = window.T  # [CHANNELS, WINDOW_SIZE]
                x = torch.tensor(window).unsqueeze(0)  # [1, CHANNELS, WINDOW_SIZE]

                with torch.no_grad():
                    _, logits = self.__model(x)
                    probs = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0, predicted_class].item()

                self.current_classification = self.__class_names[predicted_class]
                self.current_confidence = confidence * 100

    def get_classification(self) -> tuple[int, float]:
        key = next((k for k, v in self.__class_names.items() if v == self.current_classification), 0)
        return key, self.current_confidence