import serial
import time
import serial.tools.list_ports
import threading

class NfcReader:
    def __init__( self, baudrate=9600 ):
        # Data Variables
        self.new_data = False
        self.user_id = 0  # 2 chars / 8 bits
        self.application_number = 0  # 2 chars / 8 bits

        # Arduino Variables
        port = "COM3"
        ports = list( serial.tools.list_ports.comports() )
        for p in ports:
            if "Arduino" in p.description:
                port = p.name
        self.arduino = serial.Serial( port, baudrate, timeout=5 )
        time.sleep( 2 )

        # Thread Variables
        self._running = True
        self._thread = threading.Thread( target=self._read_serial, daemon=True )
        self._thread.start()


    def _read_serial( self ):
        while self._running:
            if self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode( "utf-8" ).strip()
                if "\n" in line:
                    data = line.split( "\n" )[0]
                    if len( data ) == 4:
                        self.user_id = ( ord( data[0] ) << 8 ) | ord( data[1] )
                        self.application_number = ( ord( data[2] ) << 8 ) | ord( data[3] )
                        self.new_data = True
                        print( f"user_id: { self.user_id }, application_number: { self.application_number }" )

    def is_new_data( self ) -> bool:
        return self.new_data

    def get_data( self ) -> (int, int):
        return self.user_id, self.application_number
