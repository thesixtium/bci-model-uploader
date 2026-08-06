import tkinter as tk
from tkinter import font as tkfont

from .testingMiApplication import TestingMiApplication
from src.run_models.model import Model


# Common SSVEP stimulation frequencies (Hz). Used as a default if the caller
# doesn't pass its own "frequencies" mapping in params.
DEFAULT_SSVEP_FREQUENCIES = [7.5, 8.57, 10.0, 12.0]


class TestingSsvepApplication(TestingMiApplication):
    """
    SSVEP variant of the target-trial testing application.

    Reuses all of TestingMiApplication's trial timing / CSV accuracy-logging
    logic (a target is picked, N classifications are logged on a fixed
    schedule per trial, rows are written to CSV) but swaps the static
    "highlight the active class" UI for flickering stimulus boxes, since
    SSVEP is driven by the subject looking at a box flickering at its
    tagged frequency.

    Unlike motor imagery, which can have any number of classes,
    SSVEP here always expects exactly 4 classes (one flicker box per class).
    """

    REQUIRED_CLASS_COUNT = 4

    def __init__(self, name: str, model: Model, params: dict | None):
        params = params or {}
        classifications = params.get("classifications", {})
        if len(classifications) != self.REQUIRED_CLASS_COUNT:
            raise ValueError(
                f"TestingSsvepApplication requires exactly "
                f"{self.REQUIRED_CLASS_COUNT} classifications, got "
                f"{len(classifications)}: {classifications}"
            )

        super().__init__(name, model, params)

        # Hz per class key. Accepts either a dict keyed the same as
        # classifications, or a flat list matched up in key order. Falls
        # back to DEFAULT_SSVEP_FREQUENCIES if nothing was supplied.
        frequencies = params.get("frequencies", DEFAULT_SSVEP_FREQUENCIES)
        keys = list(self._classifications.keys())
        if isinstance(frequencies, dict):
            self._frequencies = frequencies
        else:
            self._frequencies = dict(zip(keys, frequencies))

        self._flicker_on: dict = {k: False for k in keys}
        self._flicker_ids: list = []

    # ---------------- Tk thread bootstrap ----------------

    def _run_ui(self):
        self._root = tk.Tk()
        self._root.title(self.name)
        self._root.configure(bg="#0d0d0f")
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_destroy)
        self._build_ui()
        self._ready.set()
        self._poll_queue()
        self._new_target()   # kick off the first trial
        self._start_flicker()  # flicker boxes run continuously, independent of trials
        self._root.mainloop()
        self._root = None
        self._labels = {}

    def _on_destroy(self):
        self._cancel_flicker()
        super()._on_destroy()

    # ---------------- UI ----------------

    def _build_ui(self):
        root = self._root

        title_font = tkfont.Font(family="Courier", size=13, weight="bold")
        target_font = tkfont.Font(family="Courier", size=15, weight="bold")
        label_font = tkfont.Font(family="Courier", size=16, weight="bold")
        freq_font = tkfont.Font(family="Courier", size=9)
        hint_font = tkfont.Font(family="Courier", size=9)

        tk.Label(
            root, text="[ SSVEP CLASSIFIER — ACCURACY TRIAL ]",
            font=title_font, bg="#0d0d0f", fg="#39ff14", pady=14,
        ).pack(fill="x")

        tk.Frame(root, bg="#39ff14", height=1).pack(fill="x", padx=20)

        self._target_label = tk.Label(
            root, text="TARGET: —",
            font=target_font, bg="#0d0d0f", fg="#ffd23f", pady=10,
        )
        self._target_label.pack(fill="x")

        container = tk.Frame(root, bg="#0d0d0f", padx=20, pady=24)
        container.pack()

        # Fixed 2x2 grid — one flicker box per SSVEP class.
        self._labels = {}
        keys = list(self._classifications.keys())
        for idx, key in enumerate(keys):
            text = self._classifications[key]
            row, col = divmod(idx, 2)

            frame = tk.Frame(
                container, bg="#111115", bd=0,
                highlightthickness=1, highlightbackground="#2a2a35",
                width=170, height=120,
            )
            frame.grid(row=row, column=col, padx=10, pady=10)
            frame.grid_propagate(False)

            lbl = tk.Label(
                frame,
                text=text.upper().replace("_", " "),
                font=label_font, bg="#111115", fg="#3a3a50",
            )
            lbl.place(relx=0.5, rely=0.4, anchor="center")

            freq_lbl = tk.Label(
                frame,
                text=f"{self._frequencies.get(key, 0):.2f} Hz",
                font=freq_font, bg="#111115", fg="#555566",
            )
            freq_lbl.place(relx=0.5, rely=0.75, anchor="center")

            self._labels[key] = (frame, lbl, freq_lbl)

        tk.Label(
            root, text="awaiting classification…",
            font=hint_font, bg="#0d0d0f", fg="#333344", pady=10,
        ).pack()

    def _refresh_labels(self):
        # Same correct/incorrect/target semantics as the MI version, but
        # only touches border + text color — background is owned by the
        # flicker loop below, so the two don't fight over the same pixels.
        if self._target_label is not None:
            target_text = self._classifications.get(self._current_target, "—")
            self._target_label.configure(
                text=f"TARGET: {target_text.upper().replace('_', ' ')}"
            )

        for key, (frame, lbl, freq_lbl) in self._labels.items():
            is_active = key == self._active_class
            is_target = key == self._current_target

            if is_active and is_target:
                border = fg = "#39ff14"
            elif is_active and not is_target:
                border = fg = "#ff3b3b"
            elif is_target:
                border = fg = "#ffd23f"
            else:
                border, fg = "#2a2a35", "#3a3a50"

            frame.configure(
                highlightbackground=border,
                highlightthickness=3 if (is_active or is_target) else 1,
            )
            lbl.configure(fg=fg)

    # ---------------- Flicker loop (runs on Tk thread) ----------------

    def _start_flicker(self):
        for key in self._classifications.keys():
            self._schedule_flicker(key)

    def _schedule_flicker(self, key):
        freq = self._frequencies.get(key)
        if not freq or freq <= 0:
            return
        half_period_ms = max(1, round(1000 / (2 * freq)))

        def _tick():
            if not (self._root and self._root.winfo_exists()):
                return
            self._flicker_on[key] = not self._flicker_on[key]
            entry = self._labels.get(key)
            if entry:
                frame, lbl, freq_lbl = entry
                bg = "#e8e8e8" if self._flicker_on[key] else "#111115"
                frame.configure(bg=bg)
                lbl.configure(bg=bg)
                freq_lbl.configure(bg=bg)
            try:
                tid = self._root.after(half_period_ms, _tick)
                self._flicker_ids.append(tid)
            except tk.TclError:
                pass

        _tick()

    def _cancel_flicker(self):
        if self._root:
            for tid in self._flicker_ids:
                try:
                    self._root.after_cancel(tid)
                except tk.TclError:
                    pass
        self._flicker_ids = []