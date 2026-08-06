import math
import torch
from moabb.paradigms import SSVEP
from sklearn.preprocessing import LabelEncoder

from .datasets.dataset import Dataset
from .eegPreprocessing import select_and_order_channels, get_data_single_subject


class MoabbSSVEPDataLoader:
    """
    SSVEP counterpart to MoabbMotorImageryDataLoader.

    Same overall pipeline (moabb paradigm -> restrict/reorder channels ->
    crop/resample -> EegDataset split -> DataLoaders), just backed by
    moabb's SSVEP paradigm instead of MotorImagery. moabb's SSVEP datasets
    label each trial with the stimulation frequency that was flickering
    during it (e.g. "7.5", "10.0", "12.0", ...), so get_class_names() below
    naturally comes back as a {index: frequency_string} mapping — same
    shape as the MI loader's {index: class_name} mapping, just with
    frequency strings instead of movement names.

    `dataset` should expose the same interface as the one used for MI
    (get_n_classes, get_fmin/get_fmax, get_tmin/get_tmax,
    get_use_channels_names, get_dataset, get_subjects,
    get_native_sample_rate, get_crop_seconds, get_target_samples,
    get_batch_size, get_name), just pointed at an SSVEP dataset from
    moabb.datasets (e.g. Wang2016, Nakanishi2015, MAMEM1) with fmin/fmax
    tuned for SSVEP's typically wider stimulation band (moabb's SSVEP
    paradigm defaults to 7-45 Hz vs MotorImagery's 8-32 Hz) and
    use_channels_names restricted to the occipital electrodes you actually
    intend to run inference with.
    """

    def __init__(self, dataset: Dataset):
        paradigm = SSVEP(
            n_classes=dataset.get_n_classes(),
            fmin=dataset.get_fmin(),
            fmax=dataset.get_fmax(),
            tmin=dataset.get_tmin(),
            tmax=dataset.get_tmax(),
            channels=dataset.get_use_channels_names(),
            resample=None  # keep native rate; crop + resample happens manually below
        )

        epochs, y_str, metadata = paradigm.get_data(
            dataset=dataset.get_dataset(),
            subjects=dataset.get_subjects(),
            return_epochs=True,
        )

        epochs = select_and_order_channels(epochs, dataset.get_use_channels_names())
        X = epochs.get_data()

        le = LabelEncoder()
        y = le.fit_transform(y_str)
        self.class_names = {}
        for i in range(len(le.classes_)):
            self.class_names[i] = str(le.classes_[i])

        train_dataset, valid_dataset, test_dataset = get_data_single_subject(
            X=X,
            y=y,
            native_sample_rate=dataset.get_native_sample_rate(),
            crop_seconds=dataset.get_crop_seconds(),
            target_samples=dataset.get_target_samples(),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=dataset.get_batch_size(),
            num_workers=0,
            shuffle=True
        )

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=dataset.get_batch_size(),
            num_workers=0,
            shuffle=False
        )

        self.steps_per_epoch = math.ceil(len(self.train_loader))

    def get_class_names(self):
        return self.class_names

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

    def get_steps_per_epoch(self):
        return self.steps_per_epoch