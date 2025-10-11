import os
import pandas as pd
import matplotlib.pyplot as plt

# ---- CONFIG ----
METRICS_DIR = "metrics_out"
PER_FILE_CSV = os.path.join(METRICS_DIR, "metrics_per_file.csv")
BY_LLM_CSV   = os.path.join(METRICS_DIR, "metrics_by_llm.csv")
PLOTS_DIR    = os.path.join(METRICS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---- LOAD ----
per_file = pd.read_csv(PER_FILE_CSV)
by_llm   = pd.read_csv(BY_LLM_CSV)

# Make a compact label for per-file bars: "LLM | Dataset"
def label_row(r):
    return f"{r['llm']} | {r['dataset']}"

per_file["label"] = per_file.apply(label_row, axis=1)

# Order per-file bars by F1 desc (helps readability)
per_file = per_file.sort_values("f1", ascending=False)

# Metrics to plot
metrics = [
    ("accuracy",    "Accuracy"),
    ("specificity", "Specificity (TNR)"),
    ("recall",      "Recall (TPR)"),
    ("f1",          "F1-Score"),
]

# Small helper for consistent save+show
def make_bar(df, x_col, y_col, title, filename, annotate_counts=False, counts_cols=None):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(df[x_col], rotation=45, ha="right")
    # Optional annotation under x tick labels showing coverage
    if annotate_counts and counts_cols:
        # Add a note below each bar with n_used / n_missing_pred
        for idx, row in df.reset_index().iterrows():
            txt = f"used={int(row[counts_cols[0]])}, miss={int(row[counts_cols[1]])}"
            ax.text(idx, -0.07, txt, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
        plt.subplots_adjust(bottom=0.22)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved {out}")

# ---- 1) Per-file charts (all nine results) ----
for key, pretty in metrics:
    make_bar(
        per_file,
        x_col="label",
        y_col=key,
        title=f"{pretty} — per file",
        filename=f"per_file_{key}.png",
        annotate_counts=True,
        counts_cols=("n_used", "n_missing_pred"),
    )

# ---- 2) By-LLM aggregate charts ----
# Sort by F1 for the aggregate view too
by_llm = by_llm.sort_values("f1", ascending=False)

for key, pretty in metrics:
    make_bar(
        by_llm,
        x_col="llm",
        y_col=key,
        title=f"{pretty} — by LLM (micro-avg)",
        filename=f"by_llm_{key}.png",
        annotate_counts=True,
        counts_cols=("n_used", "n_missing_pred"),
    )

print("Done. Charts saved in:", PLOTS_DIR)
