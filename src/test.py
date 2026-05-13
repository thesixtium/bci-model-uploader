import numpy as np
import torch
import mne
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery
from sklearn.model_selection import train_test_split
from genericEEGPTModel import get_data_single_subject

# Step 1: Use MOABB's paradigm to epoch the data automatically
# This handles event extraction, epoching, and resampling for you
paradigm = MotorImagery(
    n_classes=2,
    fmin=0,
    fmax=38,
    tmin=0,
    tmax=4,          # 4 second trials
    resample=256     # resample to 256Hz → 1024 timepoints (256 * 4)
)

dataset = BNCI2014_004()

# Step 2: Get epoched data — this returns numpy arrays directly
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

# X shape: [n_trials, n_channels, n_timepoints]  e.g. [400, 7, 1024]
# y shape: [n_trials]  with string labels like 'left_hand', 'right_hand'
print(X.shape, y.shape)

# Step 3: Convert string labels to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_int = le.fit_transform(y)  # 'left_hand'->0, 'right_hand'->1
print(le.classes_)  # so you know which number means what


train_dataset, valid_dataset, test_dataset = get_data_single_subject(X, y_int)
