# Resume Screening LLM Fine-tuning

This project fine-tunes an LLM (Qwen2.5-1.5B-Instruct) to evaluate candidate resumes against job descriptions and make hiring decisions (Select/Reject) with reasons.

## Project Structure

- `preprocess_data.py`: Script to load, clean (PII removal, deduplication), format, and split the dataset.
- `train.py`: Script to fine-tune the model using LoRA and `trl`.
- `inference.py`: Script to run inference with the fine-tuned model.
- `processed_data/`: Directory containing the processed train/val/test datasets (CSV and JSONL).

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install torch transformers peft trl accelerate datasets pandas scikit-learn
    ```

2.  **Data Preprocessing**:
    The dataset is sourced from Hugging Face (`AzharAli05/Resume-Screening-Dataset`).
    
    To preprocess the data (PII removal, stratified split):
    ```bash
    python preprocess_data.py
    ```
    This will create a `processed_data` directory with `train`, `validation`, and `test` splits.
    
    **Preprocessing Features:**
    - **PII Removal**: Removes Emails, Phone numbers, URLs using Regex
    - **Stratified Split**: Ensures balanced 'Decision' classes across splits

