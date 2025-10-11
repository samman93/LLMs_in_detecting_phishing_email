import csv
import time
import json
import os
from typing import Optional, Dict
import re
import google.generativeai as genai  # CHANGED: Import for Gemini API


def call_llm(prompt: str, retries: int = 3, initial_retry_delay: int = 5) -> Optional[Dict[str, any]]:
    """
    Calls Gemini API to evaluate if email content is a phishing attempt.
    Expects *text* response in this exact 3-line format:
    Phishing: yes|no
    Reasoning: ...
    Confidence: 0-100

    Returns a dict: {"phishing": "...", "reasoning": "...", "confidence": number}
    """

    def parse_three_line_answer(text: str) -> Optional[Dict[str, any]]:
        """ Parse formats like:
        Phishing: yes
        Reasoning: explains why...
        Confidence: 87
        """
        if not text:
            return None
        # Strip code fences if any
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.S).strip()
        # Robust single-pass extraction
        ph = re.search(r"(?im)^\s*phishing\s*[:\-]\s*(yes|no)\s*$", text)
        rs = re.search(r"(?im)^\s*reasoning\s*[:\-]\s*(.+?)\s*$", text)
        cf = re.search(r"(?im)^\s*confidence\s*[:\-]\s*(\d{1,3})\s*$", text)
        if not ph:
            return None
        phishing = ph.group(1).lower().strip()
        reasoning = (rs.group(1).strip() if rs else "")
        try:
            confidence = int(cf.group(1)) if cf else 0
        except ValueError:
            confidence = 0
        # Clamp 0–100 just in case
        confidence = max(0, min(100, confidence))
        return {"phishing": phishing, "reasoning": reasoning, "confidence": confidence}

    try:
        api_key = "AIzaSyDcOIq-1M4Q4JLVowrDAEMjxghHFw1WC3U"  # CHANGED: Use GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. Get a key from https://ai.google.dev/")

        genai.configure(api_key=api_key)  # CHANGED: Configure Gemini API

        for attempt in range(retries):
            try:
                model = genai.GenerativeModel(
                    "gemini-2.5-flash")  # CHANGED: Use a Gemini model, e.g., 'gemini-1.5-flash' or 'gemini-1.5-pro'
                response = model.generate_content(prompt)  # CHANGED: Call Gemini generate_content
                response_text = response.text if response else ""
                print("this is the response: " + response_text)
                parsed = parse_three_line_answer(response_text)
                if parsed and all(k in parsed for k in ["phishing", "reasoning", "confidence"]):
                    return parsed
                else:
                    print(f"Invalid response format: {response_text[:200]}")
                    return None
            except Exception as e:
                if "403" in str(e):
                    print(
                        f"403 Forbidden error: Check API key, model access (e.g., gemini-1.5-flash), or billing at https://ai.google.dev/. Response: {str(e)}"
                    )
                    return None
                elif "429" in str(e):
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    print(f"Rate limit exceeded on attempt {attempt + 1}/{retries}. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                elif "deadline exceeded" in str(e).lower() or "timeout" in str(e).lower():
                    print(f"Request timed out on attempt {attempt + 1}/{retries}.")
                    return None
                else:
                    print(f"Error calling Gemini API: {e}")
                    return None
        print(f"Failed after {retries} attempts.")
        return None
    except Exception as e:
        print(f"Unexpected error in call_llm: {e}")
        return None


def process_csv(input_file: str, output_file: str, tokens_per_minute: int = 20) -> None:
    """
    Reads CSV with Email Text, Email Type, where Email Text may contain newlines,
    queries Gemini API with a phishing detection prompt, and writes results to a new CSV.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        tokens_per_minute: Token-based rate limit for LLM API calls
    """
    # Calculate delay between requests based on token rate limit
    delay = 60.0 / tokens_per_minute

    try:
        with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
                open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            # Set up CSV reader and writer
            csv_reader = csv.DictReader(infile)
            fieldnames = ['Email Text', 'Email Type', 'Phishing', 'Ground Truth', 'Reasoning', 'Confidence']
            csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            csv_writer.writeheader()

            # Validate input CSV structure
            if not all(field in csv_reader.fieldnames for field in ['Email Text', 'Email Type']):
                raise ValueError("Input CSV must have 'Email Text' and 'Email Type' columns")

            # Process each row
            for row_number, row in enumerate(csv_reader, 1):
                email_content = row['Email Text'].strip()[:500]  # Limit to 500 chars to avoid token issues

                # Determine Ground Truth based on Email Type
                ground_truth = "no" if row['Email Type'] == "Safe Email" else "yes" if row[
                                                                                           'Email Type'] == "Phishing Email" else ""

                if not email_content:
                    print(f"Skipping empty Email Text in row {row_number}")
                    csv_writer.writerow({
                        'Email Text': email_content,
                        'Email Type': row['Email Type'],
                        'Phishing': '',
                        'Ground Truth': ground_truth,
                        'Reasoning': 'Error: Empty prompt',
                        'Confidence': ''
                    })
                    continue

                # Construct the prompt
                prompt = (
                    "Is this email content a phishing attempt?\n"
                    "Answer with EXACTLY these three lines (no extra text):\n"
                    "Phishing: yes|no\n"
                    "Reasoning: <short explanation>\n"
                    "Confidence: <0-100>\n\n"
                    f"Email content: {email_content}"
                )

                print(f"Processing row {row_number}: {email_content[:50]}...")

                # Call Gemini API (function updated for Gemini)
                response = call_llm(prompt)

                # Write to CSV
                if response and isinstance(response, dict) and all(
                        key in response for key in ['phishing', 'reasoning', 'confidence']
                ):
                    csv_writer.writerow({
                        'Email Text': email_content,
                        'Email Type': row['Email Type'],
                        'Phishing': response['phishing'],
                        'Ground Truth': ground_truth,
                        'Reasoning': response['reasoning'],
                        'Confidence': response['confidence']
                    })
                    print(f"Successfully processed row {row_number}")
                else:
                    raw_response = response if response else "No response"
                    csv_writer.writerow({
                        'Email Text': email_content,
                        'Email Type': row['Email Type'],
                        'Phishing': '',
                        'Ground Truth': ground_truth,
                        'Reasoning': f'Error: Invalid or no response from Gemini API. Response: {raw_response}',
                        'Confidence': ''
                    })
                    print(f"Failed to process row {row_number}")

                # Respect rate limit
                time.sleep(delay)

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Verify the path.")
    except PermissionError:
        print(f"Error: Permission denied when accessing files")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    # Configuration
    input_file = r"F:\From Pendrive\Study\Research Uni Adelaide\Dataset\email dataset\twente dataset\Phishing_validation_emails.csv"
    output_file = r"C:\Users\samma\PyCharmMiscProject\llm_email_phishing_results_dataset_twente_gemini.csv"  # CHANGED: Updated output file name for Gemini
    tokens_per_minute = 20  # Keep same pacing logic

    print(f"Starting Gemini email phishing detection...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Rate limit: {tokens_per_minute} tokens per minute")
    print("Ensure GOOGLE_API_KEY is set and valid. See https://ai.google.dev/")

    process_csv(input_file, output_file, tokens_per_minute)
    print("Processing complete!")


if __name__ == "__main__":
    main()