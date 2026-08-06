"""
Standalone dummy version of the MI classifier UI.

No dependency on Model / ApplicationClass / csv logging — this just
shows the window and randomly picks a "classification" on a timer so
you can see how the UI looks and behaves.

Run directly:
    python dummy_mi_ui.py
"""

import random
import tkinter as tk
from tkinter import font as tkfont

# How often (ms) the dummy "classifier" picks a new active class
CLASSIFY_INTERVAL_MS = 800

# How long (ms) each target trial lasts before a new target is picked
TRIAL_DURATION_MS = 4000

CLASSIFICATIONS = {
    0: "left_hand",
    1: "right_hand",
    2: "rest",
}


class DummyMiUI:
    def __init__(self, root: tk.Tk, classifications: dict):
        self.root = root
        self.classifications = classifications
        self.active_class: int | None = None
        self.current_target: int | None = None
        self.labels: dict = {}
        self.target_label: tk.Label | None = None

        self._build_ui()
        self._new_target()
        self._tick_classifier()

    # ---------------- UI ----------------

    def _build_ui(self):
        root = self.root
        root.title("Dummy MI Demo")
        root.configure(bg="#0d0d0f")
        root.resizable(False, False)

        title_font = tkfont.Font(family="Courier", size=13, weight="bold")
        target_font = tkfont.Font(family="Courier", size=15, weight="bold")
        label_font = tkfont.Font(family="Courier", size=18, weight="bold")
        hint_font = tkfont.Font(family="Courier", size=9)

        tk.Label(
            root, text="[ MI CLASSIFIER ]",
            font=title_font, bg="#0d0d0f", fg="#39ff14", pady=14,
        ).pack(fill="x")

        tk.Frame(root, bg="#39ff14", height=1).pack(fill="x", padx=20)

        self.target_label = tk.Label(
            root, text="TARGET: —",
            font=target_font, bg="#0d0d0f", fg="#ffd23f", pady=10,
        )
        self.target_label.pack(fill="x")

        container = tk.Frame(root, bg="#0d0d0f", padx=30, pady=24)
        container.pack()

        for key, text in self.classifications.items():
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
            self.labels[key] = (frame, lbl)

        tk.Label(
            root, text="randomly classifying… (dummy mode)",
            font=hint_font, bg="#0d0d0f", fg="#333344", pady=10,
        ).pack()

    def _refresh_labels(self):
        if self.target_label is not None:
            target_text = self.classifications.get(self.current_target, "—")
            self.target_label.configure(
                text=f"TARGET: {target_text.upper().replace('_', ' ')}"
            )

        for key, (frame, lbl) in self.labels.items():
            is_active = key == self.active_class
            is_target = key == self.current_target

            if is_active and is_target:
                frame.configure(bg="#0a1f0a", highlightbackground="#39ff14", highlightthickness=2)
                lbl.configure(bg="#0a1f0a", fg="#39ff14")
            elif is_active and not is_target:
                frame.configure(bg="#241010", highlightbackground="#ff3b3b", highlightthickness=2)
                lbl.configure(bg="#241010", fg="#ff3b3b")
            elif is_target:
                frame.configure(bg="#111115", highlightbackground="#ffd23f", highlightthickness=2)
                lbl.configure(bg="#111115", fg="#ffd23f")
            else:
                frame.configure(bg="#111115", highlightbackground="#2a2a35", highlightthickness=1)
                lbl.configure(bg="#111115", fg="#3a3a50")

    # ---------------- Dummy random logic ----------------

    def _new_target(self):
        self.current_target = random.choice(list(self.classifications.keys()))
        self._refresh_labels()
        self.root.after(TRIAL_DURATION_MS, self._new_target)

    def _tick_classifier(self):
        self.active_class = random.choice(list(self.classifications.keys()))
        self._refresh_labels()
        self.root.after(CLASSIFY_INTERVAL_MS, self._tick_classifier)


if __name__ == "__main__":
    root = tk.Tk()
    app = DummyMiUI(root, CLASSIFICATIONS)
    root.mainloop()