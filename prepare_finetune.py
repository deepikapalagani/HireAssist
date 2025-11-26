#!/usr/bin/env python3
"""
prepare_finetune.py

Convert processed CSV into JSONL suitable for instruction/response finetuning.

Outputs two files by default:
 - processed_data/finetune.jsonl         (HF-style with instruction/input/output)
 - processed_data/finetune_openai.jsonl  (OpenAI-style prompt/completion)

Usage examples:
 python prepare_finetune.py --input processed_data/train.csv --max_rows 100

"""
import argparse
import json
import os
from typing import Optional

import pandas as pd

try:
    from peft.data import remove_pii
except Exception:
    # Fallback: basic remover if import fails
    import re

    def remove_pii(text: str):
        if not isinstance(text, str):
            return text
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\b(github\.com|linkedin\.com)\/\S+', '', text)
        return text


def build_examples(row, max_chars: Optional[int] = None):
    role = str(row.get('Role', '') or '')
    jd = str(row.get('Job_Description', '') or '')
    resume = str(row.get('Resume', '') or '')
    decision = str(row.get('Decision', '') or '')
    reason = str(row.get('Reason_for_decision', '') or '')

    # sanitize PII
    resume = remove_pii(resume)
    jd = remove_pii(jd)

    if max_chars:
        resume = resume[:max_chars]
        jd = jd[:max_chars]

    instruction = (
        f"You are a hiring assistant that reviews candidate resumes against a job description."
        f" For the role '{role}', read the job description and the candidate resume, then decide whether to 'select' or 'reject' the candidate and give a brief reason."
    )

    input_text = f"Role: {role}\nJob Description:\n{jd}\n\nResume:\n{resume}"

    output_text = f"Decision: {decision}\nReason: {reason}"

    # For OpenAI-style completion we prefix a space for better tokenization in some settings
    openai_completion = f" {output_text}\n"

    hf_example = {"instruction": instruction, "input": input_text, "output": output_text}
    openai_example = {"prompt": instruction + "\n\n" + input_text, "completion": openai_completion}

    return hf_example, openai_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_data/train.csv", help="Input CSV file")
    parser.add_argument("--output_dir", default="processed_data", help="Directory to write outputs")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit number of rows processed (for testing)")
    parser.add_argument("--max_chars", type=int, default=None, help="Truncate resume/jd to this many chars (optional)")
    parser.add_argument("--dropna", action="store_true", help="Drop rows missing Decision or Resume")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)

    if args.dropna:
        df = df.dropna(subset=['Decision', 'Resume'])

    if args.max_rows:
        df = df.head(args.max_rows)

    hf_path = os.path.join(args.output_dir, "finetune.jsonl")
    openai_path = os.path.join(args.output_dir, "finetune_openai.jsonl")

    written = 0
    with open(hf_path, 'w', encoding='utf-8') as hf_f, open(openai_path, 'w', encoding='utf-8') as oa_f:
        for _, row in df.iterrows():
            hf_example, openai_example = build_examples(row, max_chars=args.max_chars)
            hf_f.write(json.dumps(hf_example, ensure_ascii=False) + "\n")
            oa_f.write(json.dumps(openai_example, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} examples to:\n - {hf_path}\n - {openai_path}")


if __name__ == '__main__':
    main()
