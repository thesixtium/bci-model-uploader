from pylsl import StreamInlet, resolve_byprop
import numpy as np
import torch
import collections
from src.core.genericEEGPTModel import GenericEEGPTModel
from src.core.generic_eegpt_model_lib.modelMethods import seed_torch

# Must match exactly what you trained with
CHANNELS = 3  # C3, Cz, C4
SAMPLE_RATE = 256  # DSI-7 resampled rate
WINDOW_SIZE = 1024  # 4 seconds * 256Hz
STRIDE = 256  # new prediction every 1 second

CLASS_NAMES = ['left_hand', 'right_hand']

max_epochs = 3  # 100
max_lr = 4e-4
output_classes = 2

seed_torch(7_11_2002)

model = GenericEEGPTModel(
        load_path="checkpoints/last.ckpt",
        use_channels_names= [ "C3", "Cz", "C4" ],
        output_classes=output_classes,
        max_lr=max_lr,
        steps_per_epoch=9,
        max_epochs=max_epochs
    )

model.eval()

# Rolling buffer to accumulate samples
buffer = collections.deque(maxlen=WINDOW_SIZE)

print("Looking for EEG stream...")
streams = resolve_byprop("type", "EEG")
inlet = StreamInlet(streams[0])

print("Streaming — waiting for enough data...")
while True:
    chunk, timestamps = inlet.pull_chunk()
    if not timestamps:
        continue

    # chunk shape from LSL: [n_new_samples, n_channels]
    for sample in chunk:
        buffer.append(sample[:CHANNELS])  # take only your 3 channels

    # once we have a full window, run inference
    if len(buffer) == WINDOW_SIZE:
        # shape: [WINDOW_SIZE, CHANNELS] -> [1, CHANNELS, WINDOW_SIZE]
        window = np.array(buffer, dtype=np.float32)
        window = window.T  # [CHANNELS, WINDOW_SIZE]
        x = torch.tensor(window).unsqueeze(0)  # [1, CHANNELS, WINDOW_SIZE]

        with torch.no_grad():
            _, logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_class].item()

        print(f"Prediction: {CLASS_NAMES[predicted_class]} ({confidence * 100:.1f}% confidence)")