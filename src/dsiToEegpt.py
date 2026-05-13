import math
import pytorch_lightning as pl
import torch
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery

from genericEEGPTModel import GenericEEGPTModel, seed_torch, get_data_single_subject
from pytorch_lightning import loggers as pl_loggers

use_channels_names = [ "C3", "Cz", "C4" ]
load_path = "eegpt_mcae_58chs_4s_large4E.ckpt"

batch_size = 64
max_epochs = 100
max_lr = 4e-4
output_classes=2

seed_torch( 7_11_2002 )


"""
X shape: [n_trials, n_channels, n_timepoints]
y shape: [n_trials]

n_trials
One trial = one recording epoch — a single window of EEG data captured around an event. For example if you asked someone to imagine moving their left hand 100 times and recorded a 4-second window each time, you'd have 100 trials.
Trial 1:  [imagine left hand]  → 4 seconds of EEG recorded
Trial 2:  [imagine left hand]  → 4 seconds of EEG recorded
Trial 3:  [imagine right hand] → 4 seconds of EEG recorded
...
Trial 100

n_channels: Total number of electrodes
One channel = one electrode.

n_timepoints
The number of individual voltage samples recorded per electrode per trial. 
At 300Hz over 4 seconds that's 1200 raw samples, or 1024 after resampling.
"""



if __name__ == '__main__':
    # Step 1: Use MOABB's paradigm to epoch the data automatically
    # This handles event extraction, epoching, and resampling for you
    paradigm = MotorImagery(
        n_classes=2,
        fmin=0,
        fmax=38,
        tmin=0,
        tmax=3.996,          # 4 second trials
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


    train_dataset, valid_dataset, test_dataset = get_data_single_subject(
        X=X,
        y=y_int,
        target_sample=256 * 4
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    steps_per_epoch = math.ceil(len(train_loader))

    # init model
    model = GenericEEGPTModel(
        load_path=load_path,
        use_channels_names=use_channels_names,
        output_classes=output_classes,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        max_epochs=max_epochs
    )

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]

    trainer = pl.Trainer(accelerator='cpu',
                         #precision=16,
                         max_epochs=max_epochs,
                         log_every_n_steps=1,
                         num_sanity_val_steps=0,
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         logger=[
                             pl_loggers.TensorBoardLogger(
                                 './logs/',
                                 name="EEGPT_BCIC2B_tb",
                                 version=f"subject1"
                             ),
                             pl_loggers.CSVLogger(
                                 './logs/',
                                 name="EEGPT_BCIC2B_csv"
                             )
                         ]
                         )

    print("Testing dataloader...")
    batch = next(iter(train_loader))
    print(f"Got batch: x={batch[0].shape}, y={batch[1].shape}")

    print("Testing forward pass...")
    with torch.no_grad():
        out = model(batch[0])
    print(f"Forward pass done: {out[1].shape}")

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')