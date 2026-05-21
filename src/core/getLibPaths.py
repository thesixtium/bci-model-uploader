from pathlib import Path

import sys
from pathlib import Path

class GetLibPaths:

    def __init__(self):
        self.dsi2lsl_path = None
        self.checkpoints_path = None
        self.logs_path = None
        self.imgs_path = None
        self._find_lib_paths()

    def _get_base_dir(self) -> Path:
        if getattr(sys, 'frozen', False):
            # Running as a PyInstaller bundle
            return Path(sys.executable).resolve().parent
        else:
            # Running in development — walk up to find project root
            current = Path(__file__).resolve().parent
            while current != current.parent:
                if (current / "src").is_dir() and (current / "lib").is_dir():
                    return current
                current = current.parent
            raise FileNotFoundError(
                "Could not find project root (directory containing both 'src' and 'lib')"
            )

    def _find_lib_paths(self):
        base = self._get_base_dir()
        lib = base / "lib"
        self.dsi2lsl_path     = (lib / "dsi2lsl").resolve()
        self.checkpoints_path = (lib / "checkpoints").resolve()
        self.logs_path        = (lib / "logs").resolve()
        self.imgs_path        = (lib / "imgs").resolve()

    def get_dsi2lsl_path(self) -> Path:
        return self.dsi2lsl_path

    def get_checkpoints_path(self) -> Path:
        return self.checkpoints_path

    def get_logs_path(self) -> Path:
        return self.logs_path

    def get_imgs_path(self) -> Path:
        return self.imgs_path