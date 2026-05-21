"""
XDF to CSV Converter
Converts XDF (Extensible Data Format) files to CSV.
Each stream in the XDF file is saved as a separate CSV.

Requirements:
    pip install pyxdf pandas
"""

import sys
from pathlib import Path

try:
    import pyxdf
except ImportError:
    print("Error: pyxdf not installed. Run: pip install pyxdf")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas/numpy not installed. Run: pip install pandas numpy")
    sys.exit(1)


def list_streams(xdf_path: str):
    """Print a summary of all streams in the XDF file."""
    streams, header = pyxdf.load_xdf(xdf_path)
    print(f"\nFound {len(streams)} stream(s) in: {xdf_path}\n")
    for i, stream in enumerate(streams):
        info = stream["info"]
        name = info["name"][0]
        stype = info["type"][0]
        channel_count = int(info["channel_count"][0])
        nominal_srate = float(info["nominal_srate"][0])
        sample_count = len(stream["time_stamps"])
        print(f"  [{i}] Name: {name!r}")
        print(f"       Type: {stype}, Channels: {channel_count}, "
              f"Nominal rate: {nominal_srate} Hz, Samples: {sample_count}")
    print()


def get_channel_labels(stream) -> list[str]:
    """Extract channel labels from stream metadata, or generate defaults."""
    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = []
        for ch in channels:
            label = ch.get("label", [None])[0] or ch.get("name", [None])[0]
            labels.append(label or f"ch_{len(labels)}")
        return labels
    except (KeyError, IndexError, TypeError):
        n = int(stream["info"]["channel_count"][0])
        return [f"ch_{i}" for i in range(n)]


def stream_to_dataframe(stream) -> pd.DataFrame:
    """Convert a single XDF stream to a pandas DataFrame."""
    timestamps = stream["time_stamps"]          # shape: (n_samples,)
    time_series = stream["time_series"]         # shape: (n_samples, n_channels)

    labels = get_channel_labels(stream)

    # Marker/string streams come as lists of lists
    if isinstance(time_series, list):
        # Flatten inner lists to strings
        flat = [" | ".join(str(v) for v in row) if isinstance(row, list) else str(row)
                for row in time_series]
        df = pd.DataFrame({"timestamp": timestamps, "value": flat})
    else:
        time_series = np.array(time_series)
        if time_series.ndim == 1:
            time_series = time_series[:, np.newaxis]

        # Trim or pad labels to match actual channel count
        n_cols = time_series.shape[1]
        if len(labels) < n_cols:
            labels += [f"ch_{i}" for i in range(len(labels), n_cols)]
        labels = labels[:n_cols]

        df = pd.DataFrame(time_series, columns=labels)
        df.insert(0, "timestamp", timestamps)

    return df


def convert(xdf_path: str, output_dir: str | None = None, stream_index: int | None = None):
    """
    Convert XDF streams to CSV files.

    Args:
        xdf_path:     Path to the input .xdf file.
        output_dir:   Directory for output CSVs (defaults to same dir as input).
        stream_index: If set, only convert that stream index; otherwise convert all.
    """
    xdf_path = Path(xdf_path)
    if not xdf_path.exists():
        print(f"Error: File not found — {xdf_path}")
        sys.exit(1)

    out_dir = Path(output_dir) if output_dir else xdf_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {xdf_path} ...")
    streams, header = pyxdf.load_xdf(str(xdf_path))
    print(f"  {len(streams)} stream(s) found.")

    indices = [stream_index] if stream_index is not None else range(len(streams))

    for i in indices:
        if i >= len(streams):
            print(f"Warning: stream index {i} out of range (max {len(streams) - 1}), skipping.")
            continue

        stream = streams[i]
        name = stream["info"]["name"][0]
        stype = stream["info"]["type"][0]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        out_path = out_dir / f"{xdf_path.stem}_stream{i}_{safe_name}.csv"

        print(f"  Converting stream [{i}] {name!r} ({stype}) → {out_path.name}")
        df = stream_to_dataframe(stream)
        df.to_csv(out_path, index=False)
        print(f"    Saved {len(df):,} rows × {len(df.columns)} columns.")

    print("\nDone.")


def main():
    # ── Edit these variables ──────────────────────────────────────────────────
    XDF_FILE    = "run1.xdf"   # Path to the input .xdf file
    OUTPUT_DIR  = None              # Output folder for CSVs, or None to use same dir as input
    STREAM      = None              # Stream index to convert (0-based), or None for all streams
    LIST_ONLY   = False             # Set True to print stream info and exit without converting
    # ─────────────────────────────────────────────────────────────────────────

    if LIST_ONLY:
        list_streams(XDF_FILE)
        return

    convert(XDF_FILE, output_dir=OUTPUT_DIR, stream_index=STREAM)


if __name__ == "__main__":
    main()