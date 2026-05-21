import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

from .applicationClass import ApplicationClass
from src.run_models.model import Model


class DummyMiApplication1(ApplicationClass):
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
        print(f"Got {classification}")
        self._active_class = classification
        self._enqueue(self._refresh_labels)

    def _enqueue(self, fn):
        self._cmd_queue.put(fn)

    def _run_ui(self):
        self._root = tk.Tk()
        self._root.title(self.name)
        self._root.configure(bg="#0d0d0f")
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

        tk.Label(
            root, text="[ MI CLASSIFIER ]",
            font=title_font, bg="#0d0d0f", fg="#39ff14", pady=14,
        ).pack(fill="x")

        tk.Frame(root, bg="#39ff14", height=1).pack(fill="x", padx=20)

        container = tk.Frame(root, bg="#0d0d0f", padx=30, pady=24)
        container.pack()

        self._labels = {}
        for key, text in self._classifications.items():
            frame = tk.Frame(
                container, bg="#111115", bd=0,
                highlightthickness=1, highlightbackground="#2a2a35",
            )
            frame.pack(fill="x", pady=8, ipadx=18, ipady=14)

            lbl = tk.Label(
                frame,
                text=text.upper().replace("_", " "),
                font=label_font, bg="#111115", fg="#3a3a50", width=20,
            )
            lbl.pack()
            self._labels[key] = (frame, lbl)

        tk.Label(
            root, text="awaiting classification…",
            font=hint_font, bg="#0d0d0f", fg="#333344", pady=10,
        ).pack()

    def _refresh_labels(self):
        for key, (frame, lbl) in self._labels.items():
            if key == self._active_class:
                frame.configure(bg="#0a1f0a", highlightbackground="#39ff14", highlightthickness=2)
                lbl.configure(bg="#0a1f0a", fg="#39ff14")
            else:
                frame.configure(bg="#111115", highlightbackground="#2a2a35", highlightthickness=1)
                lbl.configure(bg="#111115", fg="#3a3a50")
