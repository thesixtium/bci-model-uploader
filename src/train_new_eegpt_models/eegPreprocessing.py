import numpy as np
import torch
from scipy.signal import resample as scipy_resample
from sklearn.model_selection import train_test_split

from src.core.eegDataset import EegDataset


def select_and_order_channels(epochs, use_channels_names):
    """
    Restrict `epochs` to exactly `use_channels_names`, in that exact order.

    GenericEEGPTModel indexes channels positionally — channel i of the
    input tensor is assumed to correspond to use_channels_names[i]. So it's
    not enough for a recording to merely *contain* the requested channels;
    the training data has to be sliced down to exactly that channel set, in
    exactly that order, or the model silently learns the wrong
    channel-to-electrode mapping and performs worse (or nonsensically) once
    it's run live against a device that only streams those channels, in
    that order (see classifier.py / nfcReader-driven apps).

    Passing `channels=` to the moabb paradigm already filters down to the
    requested set, but doesn't guarantee the returned order matches the
    order they were requested in — so this is called on top of that as a
    belt-and-suspenders guarantee.
    """
    available = set(epochs.info["ch_names"])
    missing = [ch for ch in use_channels_names if ch not in available]
    if missing:
        raise ValueError(
            f"Requested channel(s) {missing} not present in this dataset's "
            f"recording. Available channels: {sorted(available)}"
        )
    return epochs.copy().reorder_channels(list(use_channels_names))


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