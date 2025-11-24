import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import os
import re

def remove_pii(text):
    """
    Removes PII (Email, Phone, URL) from text using Regex.
    """
    if not isinstance(text, str):
        return text
        
    # Email regex
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Phone regex
    text = re.sub(r'(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', '', text)
    
    # URL regex (more robust)
    # Matches http/https, www, and common domains like github/linkedin without protocol
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b(github\.com|linkedin\.com)\/\S+', '', text)
    
    return text

def main():
    print("Loading dataset...")
    # Load the dataset
    try:
        dataset = load_dataset("AzharAli05/Resume-Screening-Dataset", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Original dataset size: {len(dataset)}")

    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()

    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['Role', 'Job_Description', 'Resume', 'Decision', 'Reason_for_decision'])

    # 2. PII Removal
    print("Removing PII from Resumes...")
    df['Resume'] = df['Resume'].apply(remove_pii)

    # Stratified Split
    # 80% Train, 10% Validation, 10% Test
    # First split: Train (80%) vs Temp (20%)
    print("Splitting dataset...")
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['Decision'], 
        random_state=42
    )

    # Second split: Validation (50% of Temp -> 10% of total) vs Test (50% of Temp -> 10% of total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['Decision'], 
        random_state=42
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Verify stratification
    print("\nClass distribution in Train:")
    print(train_df['Decision'].value_counts(normalize=True))
    print("\nClass distribution in Validation:")
    print(val_df['Decision'].value_counts(normalize=True))
    print("\nClass distribution in Test:")
    print(test_df['Decision'].value_counts(normalize=True))

    # Convert back to HF Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Save to disk
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Also save as JSONL for easy inspection
    train_df.to_json(os.path.join(output_dir, "train.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(output_dir, "validation.jsonl"), orient="records", lines=True)
    test_df.to_json(os.path.join(output_dir, "test.jsonl"), orient="records", lines=True)

    print(f"\nData saved to {output_dir}/")

if __name__ == "__main__":
    main()
