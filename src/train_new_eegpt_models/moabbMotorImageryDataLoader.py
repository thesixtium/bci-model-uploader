from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
import torch
from scipy.signal import resample as scipy_resample
from sklearn.model_selection import train_test_split
from src.core.eegDataset import EegDataset

from .datasets.dataset import Dataset


def crop_middle(x, native_sample_rate, crop_seconds):
    """
    Crop the middle `crop_seconds` out of each trial.
    x: [n_trials, n_channels, n_timepoints] at native_sample_rate Hz
    """
    crop_samples = int(round(native_sample_rate * crop_seconds))
    total_samples = x.shape[-1]

    if total_samples < crop_samples:
        raise ValueError(
            f"Trial has {total_samples} samples ({total_samples / native_sample_rate:.2f}s "
            f"@ {native_sample_rate}Hz), which is shorter than the required "
            f"{crop_seconds}s crop window."
        )

    start = (total_samples - crop_samples) // 2
    return x[..., start:start + crop_samples]


def resample_to_target(x, target_samples, use_avg=True):
    """
    Properly resample a FIXED-DURATION window (the middle crop) to exactly
    `target_samples` timepoints.

    Because the input duration is now known and constant (crop_seconds), this
    is a genuine sample-rate conversion (e.g. 500Hz -> 256Hz over the same 4s
    window), NOT an arbitrary stretch/squash of a variable-length trial.

    Uses scipy.signal.resample, which performs FFT-based band-limited
    resampling (implicitly low-pass filters before decimating), avoiding the
    aliasing artifacts that naive nearest/linear interpolation (e.g.
    torch.nn.functional.interpolate) introduces when downsampling.
    """
    if len(x.shape) not in (2, 3):
        raise ValueError("resample_to_target only supports sequences of single dim channels with optional batch")

    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)

    x_np = x.detach().cpu().numpy()
    x_resampled = scipy_resample(x_np, target_samples, axis=-1)

    return torch.from_numpy(np.ascontiguousarray(x_resampled)).to(dtype=x.dtype)


def get_data_single_subject(X, y, native_sample_rate, crop_seconds=4, target_samples=1024):
    # X shape: [n_trials, n_channels, n_timepoints] at native_sample_rate Hz
    # y shape: [n_trials]

    x = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    x = crop_middle(x, native_sample_rate, crop_seconds)
    x = resample_to_target(x, target_samples)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, stratify=train_y
    )

    return EegDataset(train_x, train_y), \
           EegDataset(valid_x, valid_y), \
           EegDataset(test_x, test_y)


class MoabbMotorImageryDataLoader:
    def __init__(self, dataset: Dataset):
        paradigm = MotorImagery(
            n_classes=dataset.get_n_classes(),
            fmin=dataset.get_fmin(),
            fmax=dataset.get_fmax(),
            tmin=dataset.get_tmin(),
            tmax=dataset.get_tmax(),
            channels=dataset.get_use_channels_names(),
            resample=None  # keep native rate; crop + resample happens manually below
        )

        X, y_str, metadata = paradigm.get_data(
            dataset=dataset.get_dataset(),
            subjects=dataset.get_subjects()
        )

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