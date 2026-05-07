import os
import joblib
import serial
import time
import serial.tools.list_ports
import threading

from .classifierSource import ClassifierSource

class NfcSource(ClassifierSource):

    def __init__(self, baudrate=9600):
        self.modelName = None
        self.newModel = None
        self.needToUpdate = False
        self.port = None

        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if "Arduino" in p.description:
                self.port = p.name

        self.baudrate = baudrate
        self._serial_buffer = ""
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        super().__init__()

    def setup(self):
        self.arduino = serial.Serial(self.port, self.baudrate, timeout=5)
        time.sleep(2)
        self._running = True
        self._thread = threading.Thread(target=self._read_serial_loop, daemon=True)
        self._thread.start()

    def _read_serial_loop(self):
        while self._running:
            try:
                waiting = self.arduino.in_waiting
                if waiting > 0:
                    chunk = self.arduino.read(waiting).decode("utf-8")
                    with self._lock:
                        self._serial_buffer += chunk
                    new_model_name = self._parse_buffer()
                    if new_model_name:
                        self._load_model(new_model_name)
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)

    def _parse_buffer(self):
        with self._lock:
            start = self._serial_buffer.find("!")
            end = self._serial_buffer.find("?")

            if start == -1 or end == -1 or end < start:
                if start == -1:
                    self._serial_buffer = ""
                return None

            model_name = self._serial_buffer[start + 1:end]
            self._serial_buffer = self._serial_buffer[end + 1:]
            return model_name if model_name else None

    def _load_model(self, model_name):
        try:
            if model_name != self.modelName:
                model = joblib.load(os.path.join(r"C:\Users\ajrbe\Documents\Git\bci-model-uploader\src\models", f"{model_name}.pk1"))
                with self._lock:
                    self.modelName = model_name
                    self.newModel = model
                    self.needToUpdate = True
                self.update()
        except FileNotFoundError:
            print(f"Model file not found: {model_name}.pk1")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.arduino.is_open:
            self.arduino.close()


