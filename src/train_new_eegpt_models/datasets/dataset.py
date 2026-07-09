class Dataset:
    def __init__(
            self,
            name,
            dataset,
            n_classes,
            fmin,          # Cutoff frequency of the highpass filter
            fmax,          # Cutoff frequency of the lowpass filter
            tmin,          # Start of the trial window (relative to cue), in seconds
            tmax,          # End of the trial window (relative to cue), in seconds
            sample_rate,   # Native/original sampling rate of the raw recording, in Hz
            use_channels_names
    ):
        self.name = name
        self.dataset = dataset
        self.n_classes = n_classes
        self.fmin = fmin
        self.fmax = fmax
        self.tmin = tmin
        self.tmax = tmax
        self.native_sample_rate = sample_rate
        self.use_channels_names = use_channels_names

        # Every dataset is epoched at its native rate over [tmin, tmax], then we
        # crop the middle 4s (in native-rate samples) and resample that fixed
        # 4s crop to 256 Hz -> always exactly 1024 timepoints, regardless of
        # native sample rate or how long tmax - tmin actually is.
        self.crop_seconds = 4
        self.target_hz = 256
        self.target_samples = self.target_hz * self.crop_seconds  # 1024

        duration = tmax - tmin
        if duration < self.crop_seconds:
            raise ValueError(
                f"{name}: epoch window (tmax - tmin = {duration}s) is shorter than "
                f"the required {self.crop_seconds}s crop window. Widen tmin/tmax."
            )

        # Batch size no longer needs to scale to a dataset-specific resample
        # rate since every dataset ends up at the same 256Hz/1024-sample shape.
        self.batch_size = 64

    def get_name(self):
        return self.name

    def get_dataset(self):
        return self.dataset

    def get_n_classes(self):
        return self.n_classes

    def get_fmin(self):
        return self.fmin

    def get_fmax(self):
        return self.fmax

    def get_tmin(self):
        return self.tmin

    def get_tmax(self):
        return self.tmax

    def get_native_sample_rate(self):
        return self.native_sample_rate

    def get_crop_seconds(self):
        return self.crop_seconds

    def get_target_hz(self):
        return self.target_hz

    def get_target_samples(self):
        return self.target_samples

    def get_batch_size(self):
        return self.batch_size

    def get_use_channels_names(self):
        return self.use_channels_names

    def get_subjects(self):
        return self.dataset.subject_list