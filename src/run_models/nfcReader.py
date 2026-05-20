import serial
import time
import serial.tools.list_ports
import threading

class NfcReader:
    def __init__( self, baud_rate=115200 ):
        self.user_id = -1
        self.new_data = False
        self.application_number = -1

        # Arduino Variables
        port = ""
        ports = list( serial.tools.list_ports.comports() )
        for p in ports:
            if "Arduino" in p.description:
                port = p.name
        print( f"Arduino: {port}" )
        self.arduino = serial.Serial( port, baud_rate, timeout=5 )
        time.sleep( 2 )

        # Thread Variables
        self.__running = True
        self.__thread = threading.Thread( target=self.__read_serial, daemon=True )
        self.__thread.start()

    def __read_serial(self):
        while self.__running:

            if self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode( "utf-8" ).strip()

                if len( line ) == 4:
                    new_user_id = ( ord( line[0] ) << 8 ) | ord( line[1] )
                    new_application_number = ( ord( line[2] ) << 8 ) | ord( line[3] )

                    if new_user_id != self.user_id or new_application_number != self.application_number:
                        self.user_id = new_user_id
                        self.application_number = new_application_number
                        self.new_data = True
            else:
                time.sleep( 0.01 )

    def is_new_data( self ) -> bool:
        return self.new_data

    def get_data( self ) -> ( int, int ):
        self.new_data = False
        return self.user_id, self.application_number
