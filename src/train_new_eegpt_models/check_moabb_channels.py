"""
check_moabb_channels.py

Given a list of channel names, checks every Motor Imagery dataset registered
in MOABB and prints the name of each dataset whose channel set is equal to,
or a superset of, the specified channel list.

Usage:
    1. Edit MY_CHANNELS below with the channel names you care about.
    2. Run: python check_moabb_channels.py

Notes:
    - Comparison is case-insensitive and ignores leading/trailing whitespace.
    - Some datasets require downloading data the first time they are accessed,
      which can be slow or fail without internet access / registration
      (e.g. some BBCI/PhysioNet mirrors). These are reported as ERROR and
      skipped, rather than crashing the whole run.
    - "Subset" here means: every channel you specified is present in the
      dataset (dataset may have additional channels too). If you want an
      EXACT match instead, set EXACT_MATCH_ONLY = True below.
"""

from moabb.paradigms import MotorImagery

# ---------------------------------------------------------------------------
# EDIT THIS: put the channel names you want to check for
# ---------------------------------------------------------------------------
MY_CHANNELS = ["F4", "C4", "P4", "P3", "C3", "F3"]

# If True, only print datasets whose channel set is EXACTLY MY_CHANNELS
# (same channels, nothing more). If False (default), datasets whose channels
# are a superset of MY_CHANNELS also count as a match.
EXACT_MATCH_ONLY = False


def normalize(names):
    """Lowercase + strip whitespace for robust comparison."""
    return {str(n).strip().lower() for n in names}


def get_dataset_channels(dataset):
    """
    Try to pull channel names for a dataset without processing all subjects.
    Uses the first available subject only, to keep this reasonably fast.
    """
    subject = dataset.subject_list[0]
    data = dataset.get_data(subjects=[subject])

    # data structure: {subject: {session: {run: raw}}}
    for sessions in data.values():
        for runs in sessions.values():
            for raw in runs.values():
                return raw.ch_names
    raise RuntimeError("Could not find any raw data to extract channel names from.")


def main():
    wanted = normalize(MY_CHANNELS)

    paradigm = MotorImagery()
    datasets = paradigm.datasets

    print(f"Checking {len(datasets)} MOABB motor imagery datasets...")
    print(f"Target channels: {sorted(MY_CHANNELS)}")
    print(f"Mode: {'EXACT match' if EXACT_MATCH_ONLY else 'SUBSET (dataset may have extra channels)'}")
    print("-" * 60)

    matches = []

    for dataset in datasets:
        name = type(dataset).__name__
        try:
            ch_names = get_dataset_channels(dataset)
            available = normalize(ch_names)

            if EXACT_MATCH_ONLY:
                is_match = wanted == available
            else:
                is_match = wanted.issubset(available)

            if is_match:
                print(f"[MATCH] {name}  (channels: {ch_names})")
                matches.append(name)
            else:
                missing = wanted - available
                print(f"[no match] {name}  (missing: {sorted(missing)})")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("-" * 60)
    print(f"Done. {len(matches)} matching dataset(s):")
    for m in matches:
        print(f"  - {m}")


if __name__ == "__main__":
    main()