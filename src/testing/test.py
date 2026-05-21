from pathlib import Path


def get_dsi2lsl_path() -> Path:
    """
    Returns the absolute path to the dsi2lsl directory.
    Works from any file within the src folder or its subdirectories.
    """
    # Start from this file's location and walk up until we find the project root
    # (identified by having both 'src' and 'lib' as siblings)
    current = Path(__file__).resolve().parent

    while current != current.parent:  # Stop at filesystem root
        if (current / "src").is_dir() and (current / "lib").is_dir():
            # Found the project root
            return (current / "lib" / "dsi2lsl").resolve()
        current = current.parent

    raise FileNotFoundError("Could not find project root (directory containing both 'src' and 'lib')")


if __name__ == "__main__":
    path = get_dsi2lsl_path()
    print(f"dsi2lsl path: {path}")