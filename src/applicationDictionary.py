from applicationClass import ApplicationClass


class ApplicationDictionary:
    def __init__( self, application_dictionary: dict[ int, ApplicationClass ] ):
        self.application_dictionary = application_dictionary
        self.current_application = None

    def open_application( self, application_number: int ):
        # Close old application
        if self.current_application is not None and self.current_application in self.application_dictionary:
            self.application_dictionary[ self.current_application ].close()
        else:
            print( f"ApplicationDictionary: Couldn't close Application { self.current_application }" )

        # Open new application
        if application_number in self.application_dictionary:
            self.application_dictionary[ application_number ].open()
            self.current_application = application_number
        else:
            print( f"ApplicationDictionary: Couldn't open Application { application_number }" )