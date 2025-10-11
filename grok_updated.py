import csv
import time
import os
import re
from typing import Optional, Dict, Any

from openai import OpenAI  # xAI is OpenAI-compatible via base_url

# ====== CONFIG ======
API_BASE_URL = "https://api.x.ai/v1"
API_ENV_VAR = "XAI_API_KEY"
MODEL_NAME = "grok-4-fast-reasoning"  # or "grok-3"
# ====================

def call_llm(prompt: str, retries: int = 3, initial_retry_delay: int = 5) -> Optional[Dict[str, Any]]:
    """
    Calls Grok via xAI's OpenAI-compatible Chat Completions API.
    Expects text in this exact 3-line format:
      Phishing: yes|no
      Reasoning: ...
      Confidence: 0-100
    Returns: {"phishing": str, "reasoning": str, "confidence": int} or None
    """

    def parse_three_line_answer(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", t, flags=re.S).strip()

        ph = re.search(r"(?im)^\s*phishing\s*[:\-]\s*(yes|no)\s*$", t)
        rs = re.search(r"(?im)^\s*reasoning\s*[:\-]\s*(.+?)\s*$", t)
        cf = re.search(r"(?im)^\s*confidence\s*[:\-]\s*(\d{1,3})\s*$", t)

        if not ph:
            return None
        phishing = ph.group(1).lower().strip()
        reasoning = (rs.group(1).strip() if rs else "")
        try:
            confidence = int(cf.group(1)) if cf else 0
        except ValueError:
            confidence = 0
        confidence = max(0, min(100, confidence))
        return {"phishing": phishing, "reasoning": reasoning, "confidence": confidence}

    api_key = "xai-gwdsBP4vmRPMMuo7qZaqpR9OrWvqks8VIUTecrdLECbQmhTGbYGg6qqR5xoW8fjlVh3CZ5RhYcPyZ6BK";
    if not api_key:
        print(f"Error: {API_ENV_VAR} environment variable not set. Create an xAI key in the console and export it.")
        return None

    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

    for attempt in range(retries):
        try:
            # Push the “exact 3 lines” constraint into a system msg for reliability
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict classifier. Respond with EXACTLY three lines and nothing else:\n"
                        "Phishing: yes|no\nReasoning: <short explanation>\nConfidence: <0-100>"
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,      # keep it deterministic for parsing
                max_tokens=150,     # plenty for the 3-line format
            )

            response_text = (resp.choices[0].message.content or "").strip()
            print("this is the response: " + response_text)

            parsed = parse_three_line_answer(response_text)
            if parsed and all(k in parsed for k in ["phishing", "reasoning", "confidence"]):
                return parsed
            else:
                print(f"Invalid response format: {response_text[:200]}")
                return None

        except Exception as e:
            es = str(e).lower()
            if "status code: 403" in es or "403" in es:
                print("403 Forbidden: check xAI API key, model access, or billing.")
                return None
            elif "429" in es or "rate" in es:
                retry_delay = initial_retry_delay * (2 ** attempt)
                print(f"Rate limit on attempt {attempt + 1}/{retries}. Sleeping {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            elif "deadline exceeded" in es or "timeout" in es or "timed out" in es:
                print(f"Request timed out on attempt {attempt + 1}/{retries}.")
                return None
            else:
                print(f"Error calling Grok API: {e}")
                return None

    print(f"Failed after {retries} attempts.")
    return None
from pathlib import Path
from string import ascii_uppercase

EXPECTED_NAME = "Phishing_Email for chatgpt.csv"  # adjust if needed

def resolve_input_file(preferred: str) -> str:
    p = Path(preferred)

    # 1) Exact path exists?
    if p.exists():
        return str(p)

    # 2) Search likely roots (your pendrive path variants)
    likely_roots = []
    # Try same folder without assuming drive letter
    for d in ascii_uppercase:  # C:..Z:
        base = Path(f"{d}:\\From Pendrive\\Study\\Research Uni Adelaide\\Dataset")
        likely_roots += [
            base / "email dataset",
            base,  # in case the CSV isn't inside "email dataset"
        ]

    # 3) Patterns to catch tiny variations (spaces/underscores/missing spaces)
    patterns = [
        EXPECTED_NAME,
        EXPECTED_NAME.replace(" ", "_"),
        EXPECTED_NAME.replace(" ", ""),
        "*Phishing*Email*chatgpt*.csv",
        "*Phishing*Email*chatgpt*",  # catches .csv.txt
    ]

    for root in likely_roots:
        if root.exists():
            for pat in patterns:
                for f in root.rglob(pat):
                    if f.is_file():
                        print(f"Found input file at: {f}")
                        return str(f)

    # 4) Last resort: suggest a quick manual check
    raise FileNotFoundError(
        "Could not find the CSV. Double-check the drive letter and folder names, or pick it with a file dialog."
    )

# --- use it here ---
input_file = resolve_input_file(
    r"F:\From Pendrive\Study\Research Uni Adelaide\Dataset\email dataset\Phishing_Email for chatgpt.csv"
)


def process_csv(input_file: str, output_file: str, tokens_per_minute: int = 20) -> None:
    """
    Reads CSV with columns: Email Text, Email Type
    Writes CSV with: Email Text, Email Type, Phishing, Ground Truth, Reasoning, Confidence
    """
    delay = 60.0 / max(1, tokens_per_minute)

    try:
        with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:

            csv_reader = csv.DictReader(infile)
            fieldnames = ['Email Text', 'Email Type', 'Phishing', 'Ground Truth', 'Reasoning', 'Confidence']
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            # Validate input CSV structure
            if not all(field in (csv_reader.fieldnames or []) for field in ['Email Text', 'Email Type']):
                raise ValueError("Input CSV must have 'Email Text' and 'Email Type' columns")

            for row_number, row in enumerate(csv_reader, 1):
                raw_text = (row.get('Email Text') or "").strip()
                email_content = raw_text[:500]  # keep short to reduce tokens

                # Ground Truth mapping
                etype = (row.get('Email Type') or "").strip()
                if etype == "Safe Email":
                    ground_truth = "no"
                elif etype == "Phishing Email":
                    ground_truth = "yes"
                else:
                    ground_truth = ""

                if not email_content:
                    print(f"Skipping empty Email Text in row {row_number}")
                    csv_writer.writerow({
                        'Email Text': email_content,
                        'Email Type': etype,
                        'Phishing': '',
                        'Ground Truth': ground_truth,
                        'Reasoning': 'Error: Empty prompt',
                        'Confidence': ''
                    })
                    continue

                # Construct strict prompt
                prompt = (
                    "Is this email content a phishing attempt?\n"
                    "Answer with EXACTLY these three lines (no extra text):\n"
                    "Phishing: yes|no\n"
                    "Reasoning: <short explanation>\n"
                    "Confidence: <0-100>\n\n"
                    f"Email content: {email_content}"
                )

                print(f"Processing row {row_number}: {email_content[:50]}...")
                response = call_llm(prompt)

                if isinstance(response, dict) and all(k in response for k in ['phishing', 'reasoning', 'confidence']):
                    csv_writer.writerow({
                        'Email Text': email_content,
                        'Email Type': etype,
                        'Phishing': response['phishing'],
                        'Ground Truth': ground_truth,
                        'Reasoning': response['reasoning'],
                        'Confidence': response['confidence'],
                    })
                    print(f"Successfully processed row {row_number}")
                else:
                    csv_writer.writerow({
                        'Email Text': email_content,
                        'Email Type': etype,
                        'Phishing': '',
                        'Ground Truth': ground_truth,
                        'Reasoning': f'Error: Invalid or no response from Grok API.',
                        'Confidence': ''
                    })
                    print(f"Failed to process row {row_number}")

                time.sleep(delay)

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Verify the path.")
    except PermissionError:
        print(f"Error: Permission denied when accessing files.")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    # Update these paths as needed
    input_file = r"F:\From Pendrive\Study\Research Uni Adelaide\Dataset\dataset 3\phishing_legit_dataset_KD_10000.csv"
    output_file = r"C:\Users\samma\PyCharmMiscProject\llm_phishing_results_dataset_3_grok_updated.csv"
    tokens_per_minute = 20

    print("Starting Grok email phishing detection...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Rate limit: {tokens_per_minute} tokens per minute")
    print(f"Ensure {API_ENV_VAR} is set and valid. Base URL: {API_BASE_URL}, Model: {MODEL_NAME}")

    process_csv(input_file, output_file, tokens_per_minute)
    print("Processing complete!")


if __name__ == "__main__":
    main()
