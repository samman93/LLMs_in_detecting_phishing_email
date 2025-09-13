import csv
import time
import json
import os
from typing import Optional, Dict
from openai import OpenAI  # CHANGED
# from xai_sdk import Client
# from xai_sdk.chat import user


def call_llm(prompt: str, retries: int = 3, initial_retry_delay: int = 5) -> Optional[Dict[str, any]]:
    """
    Calls OpenAI API to evaluate if email content is a phishing attempt.
    Expects response in JSON format: {"phishing": "yes|no", "reasoning": "explanation", "confidence": number}
    """
    client = None
    try:
        # Get API key from environment variable
        api_key = "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"  # CHANGED
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Get a key from https://platform.openai.com/")

        # Initialize OpenAI client (kept a timeout like your original)
        client = OpenAI(api_key=api_key, timeout=30)  # CHANGED

        # CHANGED: We’ll call the Chat Completions API directly inside the retry loop
        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(  # CHANGED
                    model="gpt-5-nano",      # CHANGED (choose any available model)
                    temperature=1,
                    max_completion_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Parse response content
                response_text = completion.choices[0].message.content if completion.choices else ""  # CHANGED
                try:
                    parsed_response = json.loads(response_text)
                    if isinstance(parsed_response, dict) and all(
                        key in parsed_response for key in ['phishing', 'reasoning', 'confidence']
                    ):
                        return parsed_response
                    else:
                        print(f"Invalid response format: {response_text}")
                        return None
                except json.JSONDecodeError:
                    print(f"Response is not valid JSON: {response_text}")
                    return None

            except Exception as e:
                if "403" in str(e):
                    print(
                        f"403 Forbidden error: Check API key, model access (e.g., gpt-4o-mini), or billing at https://platform.openai.com/. Response: {str(e)}"
                    )
                    return None
                elif "429" in str(e):
                    retry_delay = initial_retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit exceeded on attempt {attempt + 1}/{retries}. Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                elif "DEADLINE_EXCEEDED" in str(e) or "timeout" in str(e).lower():
                    print(f"Request timed out on attempt {attempt + 1}/{retries}.")
                    return None
                else:
                    print(f"Error calling OpenAI API: {e}")
                    return None

        print(f"Failed after {retries} attempts.")
        return None

    except Exception as e:
        print(f"Unexpected error in call_llm: {e}")
        return None
    finally:
        try:
            if client:
                client.close()  # Ensure client is closed
        except:
            pass


def process_csv(input_file: str, output_file: str, tokens_per_minute: int = 20) -> None:
    """
    Reads CSV with Email Text, Email Type, where Email Text may contain newlines,
    queries OpenAI API with a phishing detection prompt, and writes results to a new CSV.

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
                ground_truth = "no" if row['Email Type'] == "Safe Email" else "yes" if row['Email Type'] == "Phishing Email" else ""

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
                    'Is this email content a phishing attempt? Please respond in JSON format: '
                    '{"phishing":"yes|no","reasoning":"detailed explanation", "confidence": 50}\n'
                    'where "confidence" represents the likelihood of the email being a phishing attempt, '
                    'ranging from 0 to 100%. Ensure the response is valid JSON. \n'
                    f'email content: {email_content}'
                )

                print(f"Processing row {row_number}: {email_content[:50]}...")

                # Call OpenAI API (function unchanged aside from provider)
                response = call_llm(prompt)

                # Write to CSV
                if response and isinstance(response, dict) and all(
                        key in response for key in ['phishing', 'reasoning', 'confidence']):
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
                        'Reasoning': f'Error: Invalid or no response from OpenAI API. Response: {raw_response}',
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
    input_file = r"F:\From Pendrive\Study\Research Uni Adelaide\Dataset\dataset 3\phishing_legit_dataset_KD_10000.csv"
    output_file = r"C:\Users\samma\PyCharmMiscProject\llm_email_phishing_results_dataset_3_gpt_5_nano.csv"
    tokens_per_minute = 20  # Keep same pacing logic

    print(f"Starting OpenAI email phishing detection...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Rate limit: {tokens_per_minute} tokens per minute")
    print("Ensure OPENAI_API_KEY is set and valid. See https://platform.openai.com/")

    process_csv(input_file, output_file, tokens_per_minute)
    print("Processing complete!")


if __name__ == "__main__":
    main()
