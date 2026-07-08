import time
import random
import numpy as np
from pylsl import StreamInfo, StreamOutlet

# --- Configuration ---
N_CHANNELS = 8
SAMPLE_RATE = 250  # Hz
EEG_STREAM_NAME = "MyEEGStream"
MARKER_STREAM_NAME = "MyMarkerStream"

# --- EEG Stream ---
eeg_info = StreamInfo(
    name=EEG_STREAM_NAME,
    type="EEG",
    channel_count=N_CHANNELS,
    nominal_srate=SAMPLE_RATE,
    channel_format="float32",
    source_id="fake_eeg_001"
)
eeg_outlet = StreamOutlet(eeg_info)

# --- Marker Stream ---
marker_info = StreamInfo(
    name=MARKER_STREAM_NAME,
    type="Markers",
    channel_count=1,
    nominal_srate=0,
    channel_format="string",
    source_id="fake_markers_001"
)
marker_outlet = StreamOutlet(marker_info)

print(f"EEG stream online:    '{EEG_STREAM_NAME}' ({N_CHANNELS} ch @ {SAMPLE_RATE} Hz)")
print(f"Marker stream online: '{MARKER_STREAM_NAME}'")
print("Streaming... Ctrl+C to stop.\n")

MARKERS = ["stimulus/left", "stimulus/right", "response/correct", "response/error"]

interval        = 1.0 / SAMPLE_RATE
next_sample_t   = time.perf_counter()
next_marker_t   = time.perf_counter() + random.uniform(1.5, 3.0)

try:
    while True:
        now = time.perf_counter()

        # --- Send EEG sample if due ---
        if now >= next_sample_t:
            # Simple pink-ish noise: 8 channels, ~50 µV amplitude
            sample = (np.random.randn(N_CHANNELS) * 50).tolist()
            eeg_outlet.push_sample(sample)
            next_sample_t += interval

        # --- Send marker if due ---
        if now >= next_marker_t:
            label = random.choice(MARKERS)
            marker_outlet.push_sample([label])
            print(f"Marker: {label}")
            next_marker_t = now + random.uniform(1.5, 3.0)

        # Sleep just under one sample period to avoid busy-waiting too hard
        sleep_until = min(next_sample_t, next_marker_t)
        remaining = sleep_until - time.perf_counter()
        if remaining > 0.0005:
            time.sleep(remaining - 0.0005)

except KeyboardInterrupt:
    print("\nStopped.")