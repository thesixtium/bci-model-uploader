from src.run_models.model import Model


class ApplicationClass:
    def __init__(
            self,
            name: str,
            model: Model | None,
            params: dict[str, str | float | int | dict] | None
    ):
        self.name = name
        self.model = model
        self.params = params

    def get_model( self ):
        return self.model

    def open( self ):
        print( f"Opened {self.name}" )

    def close( self ):
        print( f"Closed {self.name}" )

    def receive_classification( self, classification ):
        print(f"Got {classification}")
