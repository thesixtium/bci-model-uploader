import csv
import os
import queue
import random
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import font as tkfont

from .applicationClass import ApplicationClass
from src.run_models.model import Model


class TestingMiApplication(ApplicationClass):
    def __init__(self, name: str, model: Model, params: dict | None):
        super().__init__(name, model, params)
        params = params or {}
        self._classifications = params.get("classifications", {})
        self._active_class: int | None = None

        # --- Target-trial / accuracy-logging config ---
        self._trial_duration_ms = int(params.get("trial_duration", 4.0) * 1000)
        self._classifications_per_trial = max(1, int(params.get("classifications_per_trial", 4)))
        self._csv_path = params.get("csv_path", f"{name}_accuracy_log.csv")
        self._current_target: int | None = None
        self._trial_number = 0
        self._csv_lock = threading.Lock()
        self._csv_file = None
        self._csv_writer = None
        self._log_tick_ids: list = []

        self._root: tk.Tk | None = None
        self._labels: dict = {}
        self._target_label: tk.Label | None = None
        self._cmd_queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self):
        if self._thread and self._thread.is_alive():
            return
        self._open_csv()
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
        self._close_csv()

    def receive_classification(self, classification: int):
        # NOTE: this no longer logs directly. Logging happens on the app's own
        # fixed schedule (see _schedule_log_ticks), so the number of rows
        # written per trial is guaranteed regardless of how often/irregularly
        # the caller invokes this method. This just records the latest known
        # classification so a log tick has something current to grab.
        print(f"Got {classification}")
        self._active_class = classification
        self._enqueue(self._refresh_labels)

    # ---------------- CSV handling ----------------

    def _open_csv(self):
        file_exists = os.path.isfile(self._csv_path) and os.path.getsize(self._csv_path) > 0
        self._csv_file = open(self._csv_path, mode="a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow(
                ["timestamp", "trial_number", "target", "target_label",
                 "predicted", "predicted_label", "correct"]
            )
            self._csv_file.flush()

    def _close_csv(self):
        with self._csv_lock:
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def _log_result(self, predicted: int):
        target = self._current_target
        if target is None:
            return  # no trial active yet, nothing to score against
        target_label = self._classifications.get(target, "")
        predicted_label = self._classifications.get(predicted, "")
        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            self._trial_number,
            target,
            target_label,
            predicted,
            predicted_label,
            int(target == predicted),
        ]
        with self._csv_lock:
            if self._csv_writer:
                self._csv_writer.writerow(row)
                self._csv_file.flush()

    # ---------------- Tk thread plumbing ----------------

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
        self._new_target()  # kick off the first trial
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
        self._cancel_log_ticks()
        if self._root:
            self._root.destroy()

    # ---------------- Target trial logic (runs on Tk thread) ----------------

    def _new_target(self):
        if not self._classifications:
            return
        # a new trial is starting — any ticks still pending from the
        # previous trial must not fire during this one
        self._cancel_log_ticks()
        self._trial_number += 1
        self._current_target = random.choice(list(self._classifications.keys()))
        self._refresh_labels()
        self._schedule_log_ticks()
        try:
            if self._root and self._root.winfo_exists():
                self._root.after(self._trial_duration_ms, self._new_target)
        except tk.TclError:
            pass

    def _schedule_log_ticks(self):
        """Schedule exactly self._classifications_per_trial log events, evenly
        spaced across the trial window. This is what guarantees the count —
        it no longer depends on how often receive_classification() is called."""
        if not (self._root and self._root.winfo_exists()):
            return
        n = self._classifications_per_trial
        interval = self._trial_duration_ms / n
        for k in range(1, n + 1):
            # clamp so the last tick fires strictly before the next trial's
            # _new_target reset, even if rounding pushes it to the boundary
            delay = min(round(interval * k), max(self._trial_duration_ms - 1, 0))
            try:
                tid = self._root.after(delay, self._log_tick)
                self._log_tick_ids.append(tid)
            except tk.TclError:
                pass

    def _cancel_log_ticks(self):
        if self._root:
            for tid in self._log_tick_ids:
                try:
                    self._root.after_cancel(tid)
                except tk.TclError:
                    pass
        self._log_tick_ids = []

    def _log_tick(self):
        """Fires on the app's own schedule. Logs whatever classification is
        currently known (self._active_class), which may be a repeat of the
        last one if no new classification has come in — that's expected and
        is what guarantees a fixed count per trial."""
        self._log_result(self._active_class)

    # ---------------- UI ----------------

    def _build_ui(self):
        root = self._root

        title_font = tkfont.Font(family="Courier", size=13, weight="bold")
        target_font = tkfont.Font(family="Courier", size=15, weight="bold")
        label_font = tkfont.Font(family="Courier", size=18, weight="bold")
        hint_font = tkfont.Font(family="Courier", size=9)

        tk.Label(
            root, text="[ MI CLASSIFIER — ACCURACY TRIAL ]",
            font=title_font, bg="#0d0d0f", fg="#39ff14", pady=14,
        ).pack(fill="x")

        tk.Frame(root, bg="#39ff14", height=1).pack(fill="x", padx=20)

        self._target_label = tk.Label(
            root, text="TARGET: —",
            font=target_font, bg="#0d0d0f", fg="#ffd23f", pady=10,
        )
        self._target_label.pack(fill="x")

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
        # Update target banner
        if self._target_label is not None:
            target_text = self._classifications.get(self._current_target, "—")
            self._target_label.configure(
                text=f"TARGET: {target_text.upper().replace('_', ' ')}"
            )

        for key, (frame, lbl) in self._labels.items():
            is_active = key == self._active_class
            is_target = key == self._current_target

            if is_active and is_target:
                # correct: currently-classified class matches the target
                frame.configure(bg="#0a1f0a", highlightbackground="#39ff14", highlightthickness=2)
                lbl.configure(bg="#0a1f0a", fg="#39ff14")
            elif is_active and not is_target:
                # incorrect: active but not the target
                frame.configure(bg="#241010", highlightbackground="#ff3b3b", highlightthickness=2)
                lbl.configure(bg="#241010", fg="#ff3b3b")
            elif is_target:
                # target but not yet classified as active
                frame.configure(bg="#111115", highlightbackground="#ffd23f", highlightthickness=2)
                lbl.configure(bg="#111115", fg="#ffd23f")
            else:
                frame.configure(bg="#111115", highlightbackground="#2a2a35", highlightthickness=1)
                lbl.configure(bg="#111115", fg="#3a3a50")