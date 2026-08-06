"""
check_moabb_channels.py

Given a list of channel names, checks every Motor Imagery AND SSVEP dataset
registered in MOABB and prints the name of each dataset that:
  1. has a channel set equal to, or a superset of, the specified channel
     list, AND
  2. has a native sampling rate >= MIN_SAMPLING_RATE_HZ (see below).

Usage:
    1. Edit MY_CHANNELS below with the channel names you care about.
    2. Edit MIN_SAMPLING_RATE_HZ if your target sample rate differs from the
       default (see the comment above it for how that default was picked).
    3. Run: python check_moabb_channels.py

Notes:
    - Comparison is case-insensitive and ignores leading/trailing whitespace.
    - Some datasets require downloading data the first time they are accessed,
      which can be slow or fail without internet access / registration
      (e.g. some BBCI/PhysioNet mirrors). These are reported as ERROR and
      skipped, rather than crashing the whole run.
    - "Subset" here means: every channel you specified is present in the
      dataset (dataset may have additional channels too). If you want an
      EXACT match instead, set EXACT_MATCH_ONLY = True below.
    - Sampling rate is read from the dataset's raw recording (raw.info["sfreq"]),
      i.e. the NATIVE rate before any resampling done later in the training
      pipeline (see resample_to_target in eegPreprocessing.py).
"""

from moabb.paradigms import MotorImagery, SSVEP

# ---------------------------------------------------------------------------
# EDIT THIS: put the channel names you want to check for
# ---------------------------------------------------------------------------
MY_CHANNELS = ["C4", "O1", "O2", "C3"]

# If True, only print datasets whose channel set is EXACTLY MY_CHANNELS
# (same channels, nothing more). If False (default), datasets whose channels
# are a superset of MY_CHANNELS also count as a match.
EXACT_MATCH_ONLY = False

# ---------------------------------------------------------------------------
# EDIT THIS if needed: minimum acceptable native sampling rate, in Hz.
#
# The training pipeline crops each trial to a fixed duration and then
# resamples it to a fixed number of timepoints (get_data_single_subject /
# resample_to_target in eegPreprocessing.py) so every dataset lines up on
# the same input shape EEGPT expects. That resample can only throw
# information away safely (downsampling); it can't manufacture real signal
# detail that wasn't recorded (upsampling just interpolates). So a dataset's
# NATIVE rate needs to be >= your target rate, or you're training on
# fabricated resolution.
#
# Default here (256 Hz) matches the eegpt_mcae_*_4s_*.ckpt checkpoints,
# which are trained on 1024-sample windows over a 4-second crop
# (1024 samples / 4s = 256 Hz). If you're training with a different
# crop_seconds / target_samples combination, set this to
# target_samples / crop_seconds for that config instead.
MIN_SAMPLING_RATE_HZ = 256

# Which MOABB paradigms to check. Each entry is (label, paradigm instance).
PARADIGMS = [
    ("Motor Imagery", MotorImagery()),
    ("SSVEP", SSVEP()),
]


def normalize(names):
    """Lowercase + strip whitespace for robust comparison."""
    return {str(n).strip().lower() for n in names}


def get_dataset_channels_and_sfreq(dataset):
    """
    Try to pull channel names and native sampling rate for a dataset without
    processing all subjects. Uses the first available subject only, to keep
    this reasonably fast.
    """
    subject = dataset.subject_list[0]
    data = dataset.get_data(subjects=[subject])

    # data structure: {subject: {session: {run: raw}}}
    for sessions in data.values():
        for runs in sessions.values():
            for raw in runs.values():
                return raw.ch_names, raw.info["sfreq"]
    raise RuntimeError("Could not find any raw data to extract channel names/sfreq from.")


def check_paradigm(label, paradigm, wanted):
    datasets = paradigm.datasets

    print(f"Checking {len(datasets)} MOABB {label} datasets...")
    print("-" * 60)

    matches = []

    for dataset in datasets:
        name = type(dataset).__name__
        try:
            ch_names, sfreq = get_dataset_channels_and_sfreq(dataset)
            available = normalize(ch_names)

            channels_ok = (wanted == available) if EXACT_MATCH_ONLY else wanted.issubset(available)
            sfreq_ok = sfreq >= MIN_SAMPLING_RATE_HZ

            if channels_ok and sfreq_ok:
                print(f"[MATCH] {name}  (sfreq: {sfreq:g}Hz, channels: {ch_names})")
                matches.append((name, sfreq))
            else:
                reasons = []
                if not channels_ok:
                    missing = wanted - available
                    reasons.append(f"missing channels: {sorted(missing)}")
                if not sfreq_ok:
                    reasons.append(f"sfreq {sfreq:g}Hz < required {MIN_SAMPLING_RATE_HZ}Hz")
                print(f"[no match] {name}  ({'; '.join(reasons)})")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("-" * 60)
    print(f"{label}: {len(matches)} matching dataset(s):")
    for m, sfreq in matches:
        print(f"  - {m}  ({sfreq:g}Hz)")
    print()

    return matches


def main():
    wanted = normalize(MY_CHANNELS)

    print(f"Target channels: {sorted(MY_CHANNELS)}")
    print(f"Channel mode: {'EXACT match' if EXACT_MATCH_ONLY else 'SUBSET (dataset may have extra channels)'}")
    print(f"Minimum sampling rate: {MIN_SAMPLING_RATE_HZ}Hz")
    print("=" * 60)

    all_matches = {}
    for label, paradigm in PARADIGMS:
        all_matches[label] = check_paradigm(label, paradigm, wanted)

    print("=" * 60)
    print("Summary:")
    for label, matches in all_matches.items():
        print(f"  {label}: {len(matches)} matching dataset(s)")
        for name, sfreq in matches:
            print(f"    - {name}  ({sfreq:g}Hz)")


if __name__ == "__main__":
    main()