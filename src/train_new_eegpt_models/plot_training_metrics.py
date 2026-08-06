from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, norm

# ----------------------------------------------------------------------------
# Style (white background, clean/simple)
# ----------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

TRAIN_COLOR = "#2b6cb0"
VALID_COLOR = "#dd6b20"
ACCENT_COLORS = ["#38a169", "#805ad5", "#d53f8c", "#718096"]
CHANCE_COLOR = "#999999"


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------

def wilson_ci(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    k = number correct, n = number of trials, confidence = e.g. 0.95 for a
    95% CI. More reliable than a normal-approximation +/- for small n or
    accuracies near 0 or 1.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def binomial_chance_threshold(n: int, c: int, alpha: float) -> float:
    """Minimum accuracy (fraction, not %) needed to exceed chance level at
    significance level `alpha`, for `n` trials and `c` classes, using the
    binomial cumulative distribution (Combrisson & Jerbi, 2015):

        P(z) = sum_{i=z}^{n} C(n,i) * (1/c)^i * ((c-1)/c)^(n-i)

    Returns the smallest z/n such that P(z) < alpha, i.e. the empirical
    (sample-size-corrected) chance level -- not the naive theoretical 1/c.
    """
    if n == 0:
        return float("nan")
    p = 1.0 / c
    for z in range(n, -1, -1):
        p_value = 1 - binom.cdf(z - 1, n, p)
        if p_value > alpha:
            return (z + 1) / n
    return 0.0


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_metrics(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # The logger writes some metrics under two different column names at
    # different points in training (e.g. an early sanity-check row uses
    # "valid_acc" before "valid_accuracy" exists). Coalesce them.
    if "valid_accuracy" in df.columns and "valid_acc" in df.columns:
        df["valid_accuracy"] = df["valid_accuracy"].fillna(df["valid_acc"])

    return df


def col_series(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Return (step, value) for one metric column, with rows where that
    column is NaN dropped. The CSV logs different metric groups on
    different rows, so each column must be filtered independently --
    dropping rows where *any* requested column is NaN would leave gaps
    inside a line and break it into invisible fragments."""
    d = df[["step", col]].dropna(subset=[col])
    return d["step"], d[col]


# ----------------------------------------------------------------------------
# Dashboard (loss / accuracy / f1 / kappa only -- no data-load or LR panels)
# ----------------------------------------------------------------------------

def fig_training_dashboard(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    # --- Loss ---
    ax = axes[0, 0]
    step, val = col_series(df, "train_loss")
    ax.plot(step, val, label="Train", color=TRAIN_COLOR, linewidth=1.8)
    step, val = col_series(df, "valid_loss")
    ax.plot(step, val, label="Valid", color=VALID_COLOR, linewidth=1.8)
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.legend(frameon=False)

    # --- Balanced Accuracy ---
    ax = axes[0, 1]
    step, val = col_series(df, "train_acc")
    ax.plot(step, val, label="Train", color=TRAIN_COLOR, linewidth=1.8)
    step, val = col_series(df, "valid_balanced_accuracy")
    ax.plot(step, val, label="Valid", color=VALID_COLOR, linewidth=1.8)
    ax.set_title("Balanced Accuracy")
    ax.set_xlabel("Step")
    ax.legend(frameon=False)

    # --- Cohen's Kappa ---
    ax = axes[1, 0]
    step, val = col_series(df, "valid_cohen_kappa")
    ax.plot(step, val, color=ACCENT_COLORS[1], linewidth=1.8)
    ax.set_title("Cohen's Kappa")
    ax.set_xlabel("Step")

    # --- F1 scores ---
    ax = axes[1, 1]
    step, val = col_series(df, "valid_f1_macro")
    ax.plot(step, val, label="Macro", color=ACCENT_COLORS[0], linewidth=1.8)
    step, val = col_series(df, "valid_f1_micro")
    ax.plot(step, val, label="Micro", color=ACCENT_COLORS[2], linewidth=1.8)
    step, val = col_series(df, "valid_f1_weighted")
    ax.plot(step, val, label="Weighted", color=ACCENT_COLORS[1], linewidth=1.8)
    ax.set_title("F1 Scores")
    ax.set_xlabel("Step")
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Single-bar valid balanced accuracy w/ CI + binomial chance level
# ----------------------------------------------------------------------------

def fig_valid_balanced_accuracy(df: pd.DataFrame, out_path: str, confidence_interval: float,
                                 valid_n_trials: int, n_classes: int):
    alpha = 1 - confidence_interval

    _, balanced_acc_vals = col_series(df, "valid_balanced_accuracy")
    accuracy = float(balanced_acc_vals.iloc[-1])
    k_correct = round(accuracy * valid_n_trials)

    lo, hi = wilson_ci(k_correct, valid_n_trials, confidence_interval)
    margin = (hi - lo) / 2
    chance = binomial_chance_threshold(valid_n_trials, n_classes, alpha)

    fig, ax = plt.subplots(figsize=(4.5, 5))
    bar = ax.bar([0], [accuracy], width=0.5, color=VALID_COLOR,
                 yerr=[[accuracy - lo], [hi - accuracy]], capsize=6,
                 ecolor="#333333", error_kw={"elinewidth": 1.4})

    ax.text(bar[0].get_x() + bar[0].get_width() / 2, hi + 0.02,
            f"{accuracy * 100:.1f}% \u00b1 {margin * 100:.1f}%",
            ha="center", va="bottom", fontsize=10)

    ax.axhline(chance, color=CHANCE_COLOR, linestyle="--", linewidth=1,
               label=f"Chance level (binomial, n={valid_n_trials}, p<{alpha:.3g}): {chance * 100:.1f}%")

    ax.set_xticks([0])
    ax.set_xticklabels(["Final valid\nbalanced accuracy"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Valid Balanced Accuracy")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(metrics_csv_path: str, output_dir: str, confidence_interval: float,
         valid_n_trials: int, n_classes: int):
    os.makedirs(output_dir, exist_ok=True)
    df = load_metrics(metrics_csv_path)

    fig_training_dashboard(df, os.path.join(output_dir, "training_metrics_dashboard.png"))
    fig_valid_balanced_accuracy(df, os.path.join(output_dir, "valid_balanced_accuracy.png"),
                                 confidence_interval, valid_n_trials, n_classes)

    print(f"Figures written to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    confidence_interval = 0.95  # e.g. 0.95 for a 95% confidence interval

    # Number of trials in the validation set for this run, and number of
    # classes actually being classified (needed for the binomial chance
    # level below). NOTE: these can't be read off the metrics CSV or the
    # training scripts alone -- they depend on which subject(s)/dataset
    # config were loaded at runtime, which isn't in the provided files.
    # The value below is a rough estimate from the Schirrmeister2017 dataset
    # docs (~963 trials/subject, 4 classes, "feet" excluded post-split,
    # 80/20 test split then 90/10 train/valid split of the remainder) --
    # replace it with the real count, e.g. by adding
    # `print(len(valid_loader.dataset))` in trainEEGPTModelFromDataset.py.
    valid_n_trials = 58
    n_classes = 3  # left hand / right hand / rest ("feet" excluded)

    metrics_csv_path = r"C:\Users\ajrbe\Documents\Git\bci-model-uploader\lib\logs\Schirrmeister2017_csv\version_0\metrics.csv"
    output_dir = "./training_metrics_output"

    main(metrics_csv_path, output_dir, confidence_interval, valid_n_trials, n_classes)