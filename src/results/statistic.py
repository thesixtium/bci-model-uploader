from __future__ import annotations

import glob
import math
import os
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom, norm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ----------------------------------------------------------------------------
# Style
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

SUBJECT_PALETTE = ["#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#d53f8c", "#718096"]
CHANCE_COLOR = "#999999"
CORRECT_COLOR = "#38a169"
INCORRECT_COLOR = "#e53e3e"


def pretty_label(label: str) -> str:
    """'right_hand' -> 'Right Hand'"""
    return str(label).replace("_", " ").title()


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

@dataclass
class SubjectData:
    name: str                # real name, used for folders/filenames
    df: pd.DataFrame
    class_ids: list[int] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    display_name: str = ""   # anonymized label ("Subject 1", ...), used in plots


def subject_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ("_accuracy_log.csv",):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def clean_events(df: pd.DataFrame, subject_name: str) -> pd.DataFrame:
    """Drop rows with missing target/predicted values, or with an
    'unknown' target_label/predicted_label (case-insensitive), before
    collapsing to trial level."""
    n_before = len(df)

    key_cols = ["target", "target_label", "predicted", "predicted_label"]
    not_nan = df[key_cols].notna().all(axis=1)

    def is_unknown(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower() == "unknown"

    not_unknown = ~(is_unknown(df["target_label"]) | is_unknown(df["predicted_label"]))

    cleaned = df[not_nan & not_unknown].copy()

    n_dropped = n_before - len(cleaned)
    if n_dropped:
        print(f"  {subject_name}: dropped {n_dropped}/{n_before} rows "
              f"(NaN or 'unknown' target/predicted)", file=sys.stderr)

    return cleaned


def load_subject(path: str) -> SubjectData:
    df = pd.read_csv(path)
    required = {"trial_number", "target", "target_label", "predicted", "predicted_label", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    subject_name = subject_name_from_path(path)
    df = clean_events(df, subject_name)

    # Class id -> name map, built from whatever appears in the cleaned data (targets ∪ predictions)
    id_to_name = dict(zip(df["target"], df["target_label"]))
    id_to_name.update(dict(zip(df["predicted"], df["predicted_label"])))
    class_ids = sorted(id_to_name.keys())
    class_names = [pretty_label(id_to_name[i]) for i in class_ids]

    return SubjectData(name=subject_name, df=df,
                        class_ids=class_ids, class_names=class_names)


def trial_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-event rows into one row per trial (majority-vote prediction)."""
    def mode_or_first(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    trials = (
        df.groupby("trial_number")
        .agg(
            target=("target", "first"),
            target_label=("target_label", "first"),
            predicted=("predicted", mode_or_first),
        )
        .reset_index()
    )
    trials["correct"] = (trials["target"] == trials["predicted"]).astype(int)
    return trials


# ----------------------------------------------------------------------------
# Individual-subject figures
# ----------------------------------------------------------------------------

def plot_confusion(ax, y_true, y_pred, class_ids, class_names, title, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_disp = np.nan_to_num(cm_disp)
        fmt, vmax = "{:.0%}", 1.0
    else:
        cm_disp = cm
        fmt, vmax = "{:d}", cm.max() if cm.max() > 0 else 1

    im = ax.imshow(cm_disp, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(title)
    ax.grid(False)

    thresh = vmax / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_disp[i, j] > thresh else "#222222"
            label = fmt.format(cm_disp[i, j]) if normalize else fmt.format(cm[i, j])
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=10)
    return im


def fig_confusion_matrix(subj: SubjectData, trials: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(5.8, 5))
    plot_confusion(ax, trials["target"], trials["predicted"],
                    subj.class_ids, subj.class_names,
                    f"{subj.display_name} — Confusion Matrix (n={len(trials)} trials)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subj.name}_confusion_matrix.png"))
    plt.close(fig)


def fig_class_metrics(subj: SubjectData, trials: pd.DataFrame, out_dir: str):
    precision, recall, f1, support = precision_recall_fscore_support(
        trials["target"], trials["predicted"], labels=subj.class_ids, zero_division=0
    )
    x = np.arange(len(subj.class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width, precision, width, label="Precision", color="#2b6cb0")
    ax.bar(x, recall, width, label="Recall", color="#dd6b20")
    ax.bar(x + width, f1, width, label="F1", color="#38a169")
    ax.axhline(1 / len(subj.class_ids), color=CHANCE_COLOR, linestyle="--", linewidth=1,
               label="Chance level")
    ax.set_xticks(x)
    ax.set_xticklabels(subj.class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{subj.display_name} — Per-Class Precision / Recall / F1 (trial-level)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subj.name}_class_metrics.png"))
    plt.close(fig)

    return pd.DataFrame({
        "class": subj.class_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def fig_accuracy_over_trials(subj: SubjectData, trials: pd.DataFrame, out_dir: str, window: int = 5):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = trials["correct"].map({1: CORRECT_COLOR, 0: INCORRECT_COLOR})
    ax.scatter(trials["trial_number"], trials["correct"], c=colors, s=28, zorder=3)

    if len(trials) >= 2:
        roll = trials["correct"].rolling(window=window, min_periods=1, center=True).mean()
        ax.plot(trials["trial_number"], roll, color="#2b6cb0", linewidth=2,
                 label=f"Rolling accuracy (window={window})", zorder=2)

    ax.axhline(1 / len(subj.class_ids), color=CHANCE_COLOR, linestyle="--", linewidth=1,
               label="Chance level", zorder=1)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Correct (per-trial)")
    ax.set_title(f"{subj.display_name} — Accuracy Across the Session")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subj.name}_accuracy_over_trials.png"))
    plt.close(fig)


def fig_class_distribution(subj: SubjectData, trials: pd.DataFrame, out_dir: str):
    counts = trials["target_label"].map(pretty_label).value_counts().reindex(subj.class_names, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color="#2b6cb0")
    ax.set_ylabel("Number of trials")
    ax.set_title(f"{subj.display_name} — Target Class Distribution")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{subj.name}_class_distribution.png"))
    plt.close(fig)


def summarize_subject(subj: SubjectData, trials: pd.DataFrame) -> dict:
    n_classes = len(subj.class_ids)
    n_trials = len(trials)
    n_correct = int(trials["correct"].sum())
    return {
        "subject": subj.name,                 # real name, for the CSV
        "subject_display": subj.display_name,  # anonymized label, for plots
        "n_trials": n_trials,
        "n_correct": n_correct,
        "n_classes": n_classes,
        "chance_level": 1 / n_classes if n_classes else np.nan,   # naive theoretical 1/c, for reference
        "accuracy": n_correct / n_trials if n_trials else np.nan,
    }


def process_subject(path: str, out_root: str, display_name: str) -> tuple[dict, pd.DataFrame, SubjectData]:
    subj = load_subject(path)
    subj.display_name = display_name
    trials = trial_level(subj.df)

    out_dir = os.path.join(out_root, "individual", subj.name)
    os.makedirs(out_dir, exist_ok=True)

    fig_confusion_matrix(subj, trials, out_dir)
    class_metrics = fig_class_metrics(subj, trials, out_dir)
    fig_accuracy_over_trials(subj, trials, out_dir)
    fig_class_distribution(subj, trials, out_dir)

    class_metrics.insert(0, "subject", subj.name)
    class_metrics.insert(1, "subject_display", subj.display_name)
    class_metrics.to_csv(os.path.join(out_dir, f"{subj.name}_class_metrics.csv"), index=False)

    return summarize_subject(subj, trials), class_metrics, subj


# ----------------------------------------------------------------------------
# Collective figures
# ----------------------------------------------------------------------------

def fig_accuracy_by_subject(summary: pd.DataFrame, out_dir: str, confidence_interval: float):
    alpha = 1 - confidence_interval

    x = np.arange(len(summary))
    width = 0.5

    accuracies = summary["accuracy"].to_numpy()
    lowers, uppers = [], []
    for k, n in zip(summary["n_correct"], summary["n_trials"]):
        lo, hi = wilson_ci(int(k), int(n), confidence_interval)
        lowers.append(lo)
        uppers.append(hi)
    lowers, uppers = np.array(lowers), np.array(uppers)
    yerr = np.vstack([accuracies - lowers, uppers - accuracies])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(x, accuracies, width, color="#dd6b20",
                   yerr=yerr, capsize=4, ecolor="#333333",
                   error_kw={"elinewidth": 1.2})

    # Accuracy +/- CI half-width label above each bar
    for bar, acc, lo, hi in zip(bars, accuracies, lowers, uppers):
        margin = (hi - lo) / 2
        ax.text(bar.get_x() + bar.get_width() / 2, hi + 0.02,
                 f"{acc * 100:.1f}% \u00b1 {margin * 100:.1f}%",
                 ha="center", va="bottom", fontsize=9)

    # Binomial chance level: use the smallest trial count across subjects so
    # every subject is compared against the same (most conservative) threshold.
    min_n = int(summary["n_trials"].min())
    n_classes = int(summary["n_classes"].iloc[0])  # assumes a shared class count across subjects
    chance = binomial_chance_threshold(min_n, n_classes, alpha)
    ax.axhline(chance, color=CHANCE_COLOR, linestyle="--", linewidth=1,
               label=f"Chance level (binomial, n={min_n}, p<{alpha:.3g}): {chance * 100:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["subject_display"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Subject")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_by_subject.png"))
    plt.close(fig)


def fig_pooled_confusion(all_subjects: list[SubjectData], all_trials: dict[str, pd.DataFrame], out_dir: str):
    # Use the union of classes across subjects (assumes a shared label scheme)
    class_map = {}
    for s in all_subjects:
        for cid, cname in zip(s.class_ids, s.class_names):
            class_map[cid] = cname
    class_ids = sorted(class_map)
    class_names = [class_map[c] for c in class_ids]

    y_true = pd.concat([all_trials[s.name]["target"] for s in all_subjects], ignore_index=True)
    y_pred = pd.concat([all_trials[s.name]["predicted"] for s in all_subjects], ignore_index=True)

    fig, ax = plt.subplots(figsize=(5.8, 5))
    plot_confusion(ax, y_true, y_pred, class_ids, class_names,
                    f"Pooled Confusion Matrix — All Subjects (n={len(y_true)} trials)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pooled_confusion_matrix.png"))
    plt.close(fig)


def fig_per_trial_confusion_grid(all_subjects: list[SubjectData], all_trials: dict[str, pd.DataFrame],
                                  out_dir: str):
    """One panel per subject, each showing that subject's trial-level
    (majority-vote) confusion matrix."""
    n = len(all_subjects)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 5.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    total_n = 0
    for i, subj in enumerate(all_subjects):
        trials = all_trials[subj.name]
        total_n += len(trials)
        plot_confusion(axes_flat[i], trials["target"], trials["predicted"],
                        subj.class_ids, subj.class_names,
                        f"{subj.display_name} — (n={len(trials)})")

    # Hide any unused axes if subject count doesn't fill the grid
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Per-Trial Confusion Matrices — All Subjects (total n={total_n})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(out_dir, "per_trial_confusion_matrices_grid.png"))
    plt.close(fig)


def fig_f1_heatmap(all_class_metrics: pd.DataFrame, out_dir: str):
    pivot = all_class_metrics.pivot_table(index="subject_display", columns="class", values="f1")
    fig, ax = plt.subplots(figsize=(1.4 * len(pivot.columns) + 2, 0.6 * len(pivot) + 2))
    im = ax.imshow(pivot.values, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Per-Class F1 Score by Subject")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.5 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_by_subject_heatmap.png"))
    plt.close(fig)


def fig_accuracy_trend_overlay(all_subjects: list[SubjectData], all_trials: dict[str, pd.DataFrame],
                                out_dir: str, window: int = 5):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, subj in enumerate(all_subjects):
        trials = all_trials[subj.name]
        roll = trials["correct"].rolling(window=window, min_periods=1, center=True).mean()
        color = SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)]
        ax.plot(trials["trial_number"], roll, label=subj.display_name, color=color, linewidth=2)

    chance_levels = {1 / len(s.class_ids) for s in all_subjects}
    if len(chance_levels) == 1:
        ax.axhline(chance_levels.pop(), color=CHANCE_COLOR, linestyle="--", linewidth=1, label="Chance level")

    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Trial number")
    ax.set_ylabel(f"Rolling accuracy (window={window})")
    ax.set_title("Accuracy Trend Across Subjects")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_trend_overlay.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def find_input_files(input_dir: str, pattern: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    return paths


def main(input_dir: str, pattern: str, output_dir: str, confidence_interval: float):
    paths = find_input_files(input_dir, pattern)
    if not paths:
        print(f"No files matching '{pattern}' found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    collective_dir = os.path.join(output_dir, "collective")
    os.makedirs(collective_dir, exist_ok=True)

    summaries, all_class_metrics, all_subjects, all_trials = [], [], [], {}
    for i, path in enumerate(paths, start=1):
        display_name = f"Subject {i}"
        print(f"Processing {os.path.basename(path)} as {display_name} ...")
        summary, class_metrics, subj = process_subject(path, output_dir, display_name)
        summaries.append(summary)
        all_class_metrics.append(class_metrics)
        all_subjects.append(subj)
        all_trials[subj.name] = trial_level(subj.df)

    summary_df = pd.DataFrame(summaries)
    class_metrics_df = pd.concat(all_class_metrics, ignore_index=True)

    summary_df.to_csv(os.path.join(collective_dir, "subject_summary.csv"), index=False)
    class_metrics_df.to_csv(os.path.join(collective_dir, "all_class_metrics.csv"), index=False)

    fig_accuracy_by_subject(summary_df, collective_dir, confidence_interval)
    fig_pooled_confusion(all_subjects, all_trials, collective_dir)
    fig_per_trial_confusion_grid(all_subjects, all_trials, collective_dir)
    fig_f1_heatmap(class_metrics_df, collective_dir)
    fig_accuracy_trend_overlay(all_subjects, all_trials, collective_dir)

    print("\n=== Summary ===")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary_df.to_string(index=False))
    print(f"\nFigures and tables written to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    confidence_interval = 0.95  # e.g. 0.95 for a 95% confidence interval

    input_dir = os.path.dirname(os.path.abspath(__file__))  # directory containing *_accuracy_log.csv files
    pattern = "*_accuracy_log.csv"
    output_dir = "./mi_stats"

    main(input_dir, pattern, output_dir, confidence_interval)