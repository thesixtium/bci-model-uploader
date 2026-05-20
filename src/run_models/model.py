class Model:
    def __init__(
            self,
            model_name: str,
            use_channels_names: list[str],
            output_classes: int,
            max_lr: float,
            steps_per_epoch: int,
            max_epochs: int
    ):
        self.model_name = model_name
        self.use_channels_names = use_channels_names
        self.output_classes = output_classes
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs

    def get_model_name(self):
        return self.model_name

    def get_use_channels_names(self):
        return self.use_channels_names

    def get_output_classes(self):
        return self.output_classes

    def get_max_lr(self):
        return self.max_lr

    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def get_max_epochs(self):
        return self.steps_per_epoch