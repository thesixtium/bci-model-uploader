class ApplicationClass:
    def __init__(
            self,
            name: str,
            output_classes: int,
            bci_paradigm:str
    ):
        self.name = name
        self.output_classes = output_classes
        self.bci_paradigm = bci_paradigm

    def open( self ):
        print( f"Opened {self.name}" )

    def close( self ):
        print( f"Closed {self.name}" )
