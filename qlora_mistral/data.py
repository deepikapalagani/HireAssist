import json
import pandas as pd
from datasets import load_dataset, Dataset

from config import BOS_TOKEN, EOS_TOKEN

def load_and_format(filepath, model_family="llama3", n = None):

    # Load JSONL data into a DataFrame
    lines = [json.loads(l) for l in open(filepath, encoding="utf-8")]
    df = pd.json_normalize(lines)

    # Define templates for input and output
    input_template = lambda row: (
        f"You are an HR recruiter. Analyze the job description and evaluate if the provided resume is a good fit. "
        f"Provide exactly one sentence of reasoning, followed by a final decision of either [select] or [reject]. Strictly follow the target format.\n\n"
        f"--- JOB ROLE ---\n{row['Role']}\n\n"
        f"--- JOB DESCRIPTION ---\n{row['Job_Description']}\n\n"
        f"--- CANDIDATE RESUME ---\n{row['Resume']}\n"
    )

    output_template = lambda row:f"{row['Decision']}: {row['Reason_for_decision']}"

    final_template = None
    if model_family == "llama3":
        final_template = lambda input, output: (
            f"{BOS_TOKEN}<|start_header_id|>user<|end_header_id|>\n"
            f"{input}{EOS_TOKEN}\n"  # EOS token closes the user message
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{output}{EOS_TOKEN}" # Final EOS token closes the entire sequence
        )
    elif model_family == "mistral":
        # Mistral Instruct Template: <s>[INST] {user} [/INST] {assistant}</s>
        final_template = lambda input, output: (
            f"{BOS_TOKEN}[INST] {input} [/INST] {output}{EOS_TOKEN}"
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    # Build the dataset
    data = []
    for _, row in df.iterrows():
        input_ = input_template(row)
        output_ = output_template(row)

        final_text = final_template(input_, output_)
        data.append({"text": final_text})

    print("Loaded", len(data), "examples from", filepath)
    dataset = Dataset.from_list(data)
    
    if n is not None:
        print("--- WARNING: Limiting dataset to", n, "examples for testing ---")
        dataset = dataset.select(range(min(n, len(dataset))))

    return dataset