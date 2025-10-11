import os
import re
import glob
import math
import pandas as pd
from typing import Optional, Dict, List

# === CONFIG ===
DATA_DIR = r"F:\From Pendrive\Study\Research Evaluation\Response from OpenAI"  # <-- change this
GLOB_PATTERN = "*.csv"                             # or "*/**.csv" for recursive

# --- Label normalization helpers --------------------------------------------

YES_STRINGS = {
    "yes", "y", "1", "true", "phish", "phishing", "malicious", "scam", "fraud", "attack", "unsafe",
    "phishing email", "phishingemail", "phishing_emai", "phishing_emails"
}
NO_STRINGS = {
    "no", "n", "0", "false", "legit", "legitimate", "safe", "benign", "ham",
    "safe email", "safeemail", "legitimate email", "legitimateemail"
}

def _norm(s: str) -> str:
    """Lowercase + collapse spaces + strip punctuation/non-alnum for robust matching."""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def map_ground_truth(val: str, fallback_email_type: Optional[str] = None) -> Optional[int]:
    """
    Ground Truth: positive=1 for phishing, negative=0 for legitimate.
    Primary source: 'Ground Truth' column: 'Yes' => 1, 'no' => 0.
    Fallback: if missing/empty, derive from 'Email Type' ('Phishing Email' => 1, 'Safe Email' => 0).
    """
    if pd.isna(val) or str(val).strip() == "":
        if fallback_email_type is not None:
            ft = _norm(fallback_email_type)
            if "phish" in ft:
                return 1
            if "safe" in ft or "legit" in ft:
                return 0
        return None

    s = _norm(val)
    # Your CSVs say Ground Truth has 'Yes' (= phishing) or 'no' (= not phishing)
    if s in {"yes", "y", "1", "true"}:
        return 1
    if s in {"no", "n", "0", "false"}:
        return 0
    # extra safety
    if "phish" in s:
        return 1
    if "safe" in s or "legit" in s:
        return 0
    return None

def map_prediction(val: str) -> Optional[int]:
    """
    Prediction from 'Phishing' column:
    Treats 'yes'/'phishing' as 1; 'no'/'legitimate'/'safe' as 0.
    Returns None if it can't confidently map.
    """
    if pd.isna(val) or str(val).strip() == "":
        return None
    s = _norm(val)

    if s in YES_STRINGS or "phish" in s:
        return 1
    if s in NO_STRINGS or "legit" in s or "safe" in s:
        return 0

    # handle forms like "Phishing: yes" or "is phishing? yes"
    if re.search(r"\byes\b", s):
        return 1
    if re.search(r"\bno\b", s):
        return 0

    return None

# --- Metrics -----------------------------------------------------------------

def safe_div(num: int, den: int) -> float:
    return float(num) / den if den else float("nan")

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    y_true/y_pred are 1 (phishing) or 0 (legitimate).
    Returns accuracy, specificity (TNR), recall (TPR), f1, and confusion counts.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn

    accuracy = safe_div(tp + tn, total)
    recall = safe_div(tp, tp + fn)          # TPR / Sensitivity
    specificity = safe_div(tn, tn + fp)     # TNR
    precision = safe_div(tp, tp + fp)
    f1 = (2 * precision * recall) / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and (precision + recall) > 0 else float("nan")

    return {
        "n": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "specificity": specificity, "recall": recall, "f1": f1
    }

# --- File evaluation ----------------------------------------------------------

REQUIRED_COLS = {"Email Text", "Email Type", "Phishing", "Ground Truth"}

def evaluate_csv(path: str) -> Dict[str, float]:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    # Map labels
    gt = df.apply(lambda r: map_ground_truth(r["Ground Truth"], r.get("Email Type")), axis=1)
    pred = df["Phishing"].apply(map_prediction)

    mask = gt.notna() & pred.notna()
    if mask.sum() == 0:
        return {
            "file": os.path.basename(path), "n": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "accuracy": float("nan"), "specificity": float("nan"),
            "recall": float("nan"), "f1": float("nan")
        }

    metrics = compute_metrics(gt[mask], pred[mask])
    metrics["file"] = os.path.basename(path)
    return metrics

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
    if not files:
        raise SystemExit(f"No CSV files found in: {DATA_DIR}")

    results: List[Dict] = []
    for p in files:
        try:
            res = evaluate_csv(p)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(p)} due to error: {e}")
            continue
        results.append(res)

    out = pd.DataFrame(results, columns=["file","n","tp","tn","fp","fn","accuracy","specificity","recall","f1"])
    # Round for display; keep raw in a separate df if needed
    shown = out.copy()
    for c in ["accuracy","specificity","recall","f1"]:
        shown[c] = shown[c].astype(float).round(4)
    print("\nPer-file metrics:\n", shown.to_string(index=False))

    # Save summary CSV
    out.to_csv("metrics_summary.csv", index=False)
    print("\nSaved: metrics_summary.csv")

if __name__ == "__main__":
    main()
