import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

from .applicationClass import ApplicationClass
from src.run_models.model import Model

BG         = "#0d0f1a"
DIM_BG     = "#13152a"
DIM_BORDER = "#2a2d45"
DIM_FG     = "#3a3d5c"

BLUE       = "#55CDFC"
PINK       = "#F7A8B8"
WHITE      = "#FFFFFF"

ACTIVE_COLORS = [
    (BLUE,  "#0a1f2e", "#1a3d55"),
    (PINK,  "#2e0a14", "#552030"),
    (WHITE, "#1e1e2e", "#3a3a5c"),
]


class DummyMiApplication2(ApplicationClass):
    def __init__(self, name: str, model: Model, params: dict | None):
        super().__init__(name, model, params)
        self._classifications = (params or {}).get("classifications", {})
        self._active_class: int | None = None

        self._root: tk.Tk | None = None
        self._labels: dict = {}
        self._cmd_queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self):
        if self._thread and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_ui, name="tk-ui", daemon=False)
        self._thread.start()
        self._ready.wait()

    def close(self):
        if self._root:
            self._enqueue(self._on_destroy)
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def receive_classification(self, classification: int):
        self._active_class = classification
        self._enqueue(self._refresh_labels)

    def _enqueue(self, fn):
        self._cmd_queue.put(fn)

    def _run_ui(self):
        self._root = tk.Tk()
        self._root.title(self.name)
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_destroy)
        self._build_ui()
        self._ready.set()
        self._poll_queue()
        self._root.mainloop()
        # Only reached after mainloop exits — Tk thread, safe to clear
        self._root = None
        self._labels = {}

    def _poll_queue(self):
        try:
            while True:
                fn = self._cmd_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        except Exception:
            return  # a queued callback raised — stop polling
        try:
            if self._root and self._root.winfo_exists():
                self._root.after(50, self._poll_queue)
        except tk.TclError:
            pass  # interpreter already destroyed — stop polling

    def _on_destroy(self):
        if self._root:
            self._root.destroy()

    def _build_ui(self):
        root = self._root

        title_font = tkfont.Font(family="Courier", size=13, weight="bold")
        label_font = tkfont.Font(family="Courier", size=18, weight="bold")
        hint_font  = tkfont.Font(family="Courier", size=9)

        stripe_frame = tk.Frame(root, bg=BG)
        stripe_frame.pack(fill="x")
        for color, h in [(BLUE, 4), (PINK, 4), (WHITE, 4), (PINK, 4), (BLUE, 4)]:
            tk.Frame(stripe_frame, bg=color, height=h).pack(fill="x")

        tk.Label(
            root, text="[ MI CLASSIFIER ]",
            font=title_font, bg=BG, fg=WHITE, pady=14,
        ).pack(fill="x")

        tk.Frame(root, bg=DIM_BORDER, height=1).pack(fill="x", padx=20)

        container = tk.Frame(root, bg=BG, padx=30, pady=24)
        container.pack()

        self._labels = {}
        for i, key in enumerate(self._classifications.keys()):
            text = self._classifications[key]
            color_set = ACTIVE_COLORS[i % len(ACTIVE_COLORS)]

            frame = tk.Frame(
                container, bg=DIM_BG,
                highlightthickness=1, highlightbackground=DIM_BORDER,
            )
            frame.pack(fill="x", pady=8, ipadx=18, ipady=14)

            lbl = tk.Label(
                frame,
                text=text.upper().replace("_", " "),
                font=label_font, bg=DIM_BG, fg=DIM_FG, width=20,
            )
            lbl.pack()
            self._labels[key] = (frame, lbl, color_set)

        tk.Label(
            root, text="awaiting classification…",
            font=hint_font, bg=BG, fg=DIM_BORDER, pady=10,
        ).pack()

        bottom = tk.Frame(root, bg=BG)
        bottom.pack(fill="x", side="bottom")
        for color, h in [(BLUE, 4), (PINK, 4), (WHITE, 4), (PINK, 4), (BLUE, 4)]:
            tk.Frame(bottom, bg=color, height=h).pack(fill="x")

    def _refresh_labels(self):
        for key, (frame, lbl, color_set) in self._labels.items():
            fg_color, active_bg, active_border = color_set
            if key == self._active_class:
                frame.configure(bg=active_bg, highlightbackground=active_border, highlightthickness=2)
                lbl.configure(bg=active_bg, fg=fg_color)
            else:
                frame.configure(bg=DIM_BG, highlightbackground=DIM_BORDER, highlightthickness=1)
                lbl.configure(bg=DIM_BG, fg=DIM_FG)
