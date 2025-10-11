import os, glob, re, math
import pandas as pd
from typing import Dict, Optional, Tuple

# === SET ME ===
DATA_DIR = r"F:\From Pendrive\Study\Research Evaluation\all results_updated"   # Folder containing your 9 CSVs
PATTERN  = "*.csv"                   # Or "**/*.csv" for recursive
OUTPUT_DIR = "metrics_out"           # Results will be written here

# Heuristics to infer LLM and dataset from filenames (edit as you like)
LLM_TOKENS = {
    "openai": ["openai", "gpt", "gpt4", "gpt-4o", "gpt5", "gpt-5", "gpt-5-nano", "gpt-4o-mini"],
    "gemini": ["gemini", "g-1.5", "g1.5", "google"],
    "grok":   ["grok", "xai"],
}
# map any substring to a clean dataset label you want to see in tables
DATASET_ALIASES = {
    "dataset 1": "dataset1",
    "dataset1":  "dataset1",
    "dataset_1": "dataset1",
    "dataset-1": "dataset1",
    "dataset 2": "dataset2",
    "dataset2":  "dataset2",
    "dataset_2": "dataset2",
    "dataset-2": "dataset2",
    "dataset 3": "dataset3",
    "dataset3":  "dataset3",
    "dataset_3": "dataset3",
    "dataset-3": "dataset3",
    "kd_10000":  "dataset3",  # example from your earlier path
    "enron":     "enron",
    "phish":     "phish",
    "legit":     "legit",
}

# ---------- helpers ----------
def to01(s) -> Optional[int]:
    """Map yes/no-ish to 1/0; None if not parseable."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in ("yes","y","1","true"):  return 1
    if t in ("no","n","0","false"):  return 0
    return None

def parse_llm_and_dataset(filename: str) -> Tuple[str, str]:
    """
    Very light heuristics:
    - LLM: match any token from LLM_TOKENS
    - Dataset: first matching alias substring in name, else if 'datasetX' pattern exists use that,
               otherwise 'unknown'
    """
    name = os.path.basename(filename).lower()

    llm = "unknown"
    for k, toks in LLM_TOKENS.items():
        if any(tok in name for tok in toks):
            llm = k
            break

    # dataset from aliases
    dataset = None
    for sub, label in DATASET_ALIASES.items():
        if sub in name:
            dataset = label
            break

    # fallback: datasetX or dataset_X
    if dataset is None:
        m = re.search(r"(dataset[\s\-_]*\d+)", name)
        if m:
            dataset = m.group(1).replace(" ", "").replace("_", "").replace("-", "")
        else:
            dataset = "unknown"

    return llm, dataset

def metrics_from_series(gt: pd.Series, pred: pd.Series) -> Dict[str, float]:
    gt = gt.astype(int); pred = pred.astype(int)
    tp = int(((gt==1) & (pred==1)).sum())
    tn = int(((gt==0) & (pred==0)).sum())
    fp = int(((gt==0) & (pred==1)).sum())
    fn = int(((gt==1) & (pred==0)).sum())
    n  = tp + tn + fp + fn

    acc  = (tp + tn) / n if n else float('nan')
    tpr  = tp / (tp + fn) if (tp + fn) else float('nan')   # recall / sensitivity
    tnr  = tn / (tn + fp) if (tn + fp) else float('nan')   # specificity
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    f1   = (2*prec*tpr)/(prec+tpr) if math.isfinite(prec) and math.isfinite(tpr) and (prec+tpr)>0 else float('nan')
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "specificity": tnr, "recall": tpr, "f1": f1}

def evaluate_file(path: str) -> Dict:
    df = pd.read_csv(path, encoding="utf-8")


    # 1) Drop rows that are completely empty across our key columns
    key_cols = ["Email Text", "Email Type", "Phishing", "Ground Truth"]
    df = df[~df[key_cols].isna().all(axis=1)]

    # 2) Keep only rows that actually have a ground-truth label
    df = df[df["Ground Truth"].notna()]
    required = {"Email Text","Email Type","Phishing","Ground Truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    # Map labels
    gt   = df["Ground Truth"].map(to01)
    pred = df["Phishing"].map(to01)

    # Count missing predictions (LLM no response)
    n_total = len(df)
    n_missing_pred = int(pred.isna().sum())

    # Only keep rows where both GT and prediction are valid for metrics
    mask = gt.notna() & pred.notna()
    used = mask.sum()

    if used == 0:
        base = {"n_total": n_total, "n_used": 0, "n_missing_pred": n_missing_pred,
                "tp":0,"tn":0,"fp":0,"fn":0,
                "accuracy": float('nan'), "specificity": float('nan'), "recall": float('nan'), "f1": float('nan')}
    else:
        m = metrics_from_series(gt[mask], pred[mask])
        base = {"n_total": n_total, "n_used": m["n"], "n_missing_pred": n_missing_pred,
                "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
                "accuracy": m["accuracy"], "specificity": m["specificity"], "recall": m["recall"], "f1": m["f1"]}

    llm, dataset = parse_llm_and_dataset(path)
    base.update({"file": os.path.basename(path), "llm": llm, "dataset": dataset})
    return base

def micro_from_counts(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Aggregate by summing counts, then recompute metrics (micro-average)."""
    agg = df.groupby(group_cols, dropna=False)[["tp","tn","fp","fn","n_used","n_total","n_missing_pred"]].sum().reset_index()
    rows = []
    for _, r in agg.iterrows():
        tp, tn, fp, fn = int(r.tp), int(r.tn), int(r.fp), int(r.fn)
        n = tp + tn + fp + fn
        acc  = (tp + tn) / n if n else float('nan')
        tpr  = tp / (tp + fn) if (tp + fn) else float('nan')
        tnr  = tn / (tn + fp) if (tn + fp) else float('nan')
        prec = tp / (tp + fp) if (tp + fp) else float('nan')
        f1   = (2*prec*tpr)/(prec+tpr) if math.isfinite(prec) and math.isfinite(tpr) and (prec+tpr)>0 else float('nan')
        rows.append({
            **{c: r[c] for c in group_cols},
            "n_total": int(r.n_total),
            "n_used":  int(r.n_used),
            "n_missing_pred": int(r.n_missing_pred),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "specificity": tnr, "recall": tpr, "f1": f1
        })
    return pd.DataFrame(rows)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN)))
    if not files:
        raise SystemExit(f"No CSVs found in {DATA_DIR}")

    per_file = []
    for p in files:
        try:
            res = evaluate_file(p)
            per_file.append(res)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(p)} -> {e}")

    if not per_file:
        raise SystemExit("No valid files processed.")

    df = pd.DataFrame(per_file, columns=[
        "file","llm","dataset","n_total","n_used","n_missing_pred",
        "tp","tn","fp","fn","accuracy","specificity","recall","f1"
    ])

    # --- save per-file summary + one CSV per input file
    per_file_csv = os.path.join(OUTPUT_DIR, "metrics_per_file.csv")
    df.to_csv(per_file_csv, index=False)
    print(f"Saved per-file summary -> {per_file_csv}")

    # Also save a tiny per-file CSV next to each input row (1 row per file)
    perfile_dir = os.path.join(OUTPUT_DIR, "per_file")
    os.makedirs(perfile_dir, exist_ok=True)
    for _, row in df.iterrows():
        out_path = os.path.join(perfile_dir, f"{row['file']}_metrics.csv")
        pd.DataFrame([row]).to_csv(out_path, index=False)

    # --- aggregate by LLM (what you asked)
    by_llm = micro_from_counts(df, ["llm"])
    by_llm_csv = os.path.join(OUTPUT_DIR, "metrics_by_llm.csv")
    by_llm.to_csv(by_llm_csv, index=False)
    print(f"Saved by-LLM summary -> {by_llm_csv}")

    # --- (optional but useful) aggregate by dataset
    by_ds = micro_from_counts(df, ["dataset"])
    by_ds_csv = os.path.join(OUTPUT_DIR, "metrics_by_dataset.csv")
    by_ds.to_csv(by_ds_csv, index=False)
    print(f"Saved by-dataset summary -> {by_ds_csv}")

    # Pretty print to console
    pd.set_option("display.max_columns", None)
    print("\n=== Per-file ===")
    print(df.round(4).to_string(index=False))
    print("\n=== By LLM (micro) ===")
    print(by_llm.round(4).to_string(index=False))
    print("\n=== By dataset (micro) ===")
    print(by_ds.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
