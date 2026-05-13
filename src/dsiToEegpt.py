import pytorch_lightning as pl
from genericEEGPTModel import GenericEEGPTModel, seed_torch
from pytorch_lightning import loggers as pl_loggers
from moabb.datasets import BNCI2014_004
from moabbMotorImageryDataLoader import MoabbMotorImageryDataLoader

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
    model_name = "DSI7_EEGPT"
    use_channels_names = ["C3", "Cz", "C4"]
    base_model = "eegpt_mcae_58chs_4s_large4E.ckpt"

    max_epochs = 3  # 100
    max_lr = 4e-4
    output_classes = 2

    seed_torch(7_11_2002)

    data = MoabbMotorImageryDataLoader(
        BNCI2014_004(),
        2,
        0,
        38,
        0,
        4,
        256,
        64
    )

    # init model
    model = GenericEEGPTModel(
        load_path=base_model,
        use_channels_names=use_channels_names,
        output_classes=output_classes,
        max_lr=max_lr,
        steps_per_epoch=data.get_steps_per_epoch(),
        max_epochs=max_epochs
    )

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    save = pl.callbacks.ModelCheckpoint(
                                 dirpath='./checkpoints/',
                                 filename=model_name + '-{epoch:02d}-{valid_acc:.2f}',
                                 save_top_k=0,
                                 save_last=True  # also always save the most recent
                             )

    trainer = pl.Trainer(accelerator='cpu',
                         max_epochs=max_epochs,
                         log_every_n_steps=1,
                         num_sanity_val_steps=0,
                         enable_checkpointing=True,  # change from False
                         callbacks=[ lr_monitor, save ],
                         logger=[
                             pl_loggers.TensorBoardLogger(
                                 './logs/',
                                 name=f"{model_name}_tb",
                                 version=f"subject1"
                             ),
                             pl_loggers.CSVLogger(
                                 './logs/',
                                 name=f"{model_name}_csv"
                             )
                         ]
                         )

    trainer.fit(model, data.get_train_loader(), data.get_valid_loader() )

    results = trainer.validate(model, data.get_valid_loader())
    print(results)