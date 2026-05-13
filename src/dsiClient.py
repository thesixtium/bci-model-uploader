import subprocess
from os.path import join
from pylsl import StreamInlet, resolve_byprop
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import torch
import collections

DSI_PORT = "COM3"
DSI_STREAMER_NAME = "DSI-Streamer-v.1.08.119.exe"

def run_dsi_streamer():
    # DSI Streamer (for headset impedance checking)
    try:
        dsi_streamer_path = join("DSI_Streamer", "DSI-Streamer-v.1.08.119.exe")
        subprocess.run([dsi_streamer_path], check = True)
    except subprocess.CalledProcessError as e:
        print(f"DSI Streamer failed with error code {e.returncode}")
        raise Exception(f"Error running DSI Streamer: {e}")
    print("DSI Streamer closed")

def run_dsi2lsl():
    # DSI2LSL
    try:
        dsi2lsl_path = join("dsi2lsl", "dsi2lsl.exe")
        subprocess.run(
            [
                dsi2lsl_path,
                f'port={DSI_PORT}',
                'lsl-stream-name=DSI7',
                'montage=F4,C4,S1,S3,C3,F3'
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"DSI2LSL failed with error code {e.returncode}")
        raise Exception(f"Error running DSI2LSL: {e}")

def main():
    # start DSI Streamer
    #try:
     #   run_dsi_streamer()
    #except Exception as e:
     #   error_msg = f"DSI Streamer failed: {e}"
      #  print(error_msg)

    # start DSI2LSL
    #try:
       # run_dsi2lsl()
    #except Exception as e:
     #   error_msg = f"DSI2LSL failed: {e}"
      #  print(error_msg)

    if True:
        print("looking for an EEG stream...")
        streams = resolve_byprop("type", "EEG")

        # create a new inlet to read from the stream
        inlet = StreamInlet(streams[0])

        while True:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            chunk, timestamps = inlet.pull_chunk()
            if timestamps:
                print(timestamps, chunk)
    else:
        # Must match exactly what you trained with
        CHANNELS = 3  # C3, Cz, C4
        SAMPLE_RATE = 256  # DSI-7 resampled rate
        WINDOW_SIZE = 1024  # 4 seconds * 256Hz
        STRIDE = 256  # new prediction every 1 second

        CLASS_NAMES = ['left_hand', 'right_hand']

        model = GenericEEGPTModel.load_from_checkpoint(
            'checkpoints/eegpt-epoch=99-valid_acc=0.72.ckpt',
            # must pass any args that __init__ requires:
            load_path=load_path,
            use_channels_names=use_channels_names,
            output_classes=2,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
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

if __name__ == "__main__":
    main()