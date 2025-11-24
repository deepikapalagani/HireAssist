# Resume Screening LLM Fine-tuning

## Project Structure

- `preprocess_data.py`: Script to load, clean (PII removal), and split the dataset.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
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