# Resume Screening LLM Fine-tuning

## Project Structure

- `preprocess_data.py`: Script to load, clean (PII removal), and split the dataset.

## Setup
- For AWS EC2 instance setup, refer to this: https://docs.google.com/presentation/d/1Tw_klO84R9G7CZ3cINAKgy4BfdNm-8dlnRXSBIVD_3A/edit?slide=id.g28302656ccb_0_10#slide=id.g28302656ccb_0_10

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