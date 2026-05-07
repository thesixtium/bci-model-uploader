import os
import joblib
import serial
import time
import serial.tools.list_ports
import threading

from .classifierSource import ClassifierSource


class NfcSource( ClassifierSource ):

    def __init__( self, baudrate=9600 ):
        self.modelName = None
        self.newModel = None
        self._thread = None
        self.port = None

        self.needToUpdate = False
        self._running = False

        self.baudrate = baudrate
        self._serial_buffer = ""
        self._lock = threading.Lock()

        ports = list( serial.tools.list_ports.comports() )
        for p in ports:
            if "Arduino" in p.description:
                self.port = p.name

        super().__init__()


    def setup( self ):
        self.arduino = serial.Serial( self.port, self.baudrate, timeout=5 )
        time.sleep( 2 )

        self._running = True
        self._thread = threading.Thread( target=self._read_serial_loop, daemon=True )
        self._thread.start()


    def _read_serial_loop( self ):
        while self._running:
            try:
                waiting = self.arduino.in_waiting
                if waiting > 0:
                    chunk = self.arduino.read( waiting ).decode( "utf-8" )
                    with self._lock:
                        self._serial_buffer += chunk
                    newModelName = self._parse_buffer()
                    if newModelName:
                        self._load_model( newModelName )
                else:
                    time.sleep( 0.05 )
            except Exception as e:
                print( f"Serial read error: {e}" )
                time.sleep( 0.1 )


    def _parse_buffer( self ):
        with self._lock:
            start = self._serial_buffer.find( "!" )
            end = self._serial_buffer.find( "?" )

            if start == -1 or end == -1 or end < start:
                if start == -1:
                    self._serial_buffer = ""
                return None

            modelName = self._serial_buffer[start + 1:end]
            self._serial_buffer = self._serial_buffer[end + 1:]
            return modelName if modelName else None


    def _load_model( self, modelName ):
        try:
            if modelName != self.modelName:
                model = joblib.load(
                    os.path.join(
                        r"C:\Users\ajrbe\Documents\Git\bci-model-uploader\src\models", f"{modelName}.pk1"
                    )
                )

                with self._lock:
                    self.modelName = modelName
                    self.newModel = model
                    self.needToUpdate = True

                self.update()

        except FileNotFoundError:
            print(f"Model file not found: {modelName}.pk1")

        except Exception as e:
            print(f"Error loading model '{modelName}': {e}")


    def stop( self ):
        self._running = False

        if self._thread:
            self._thread.join( timeout=2 )

        if self.arduino.is_open:
            self.arduino.close()


