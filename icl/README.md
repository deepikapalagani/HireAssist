# In-Context Learning (ICL) Inference for HireAssist

This directory contains scripts to evaluate Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.3 on the Resume Screening task using In-Context Learning (Few-Shot).

## Contents

- `icl_inference.py`: The main Python script for running inference.
- `run_icl_eval.sh`: A shell script to automate the evaluation process.

## Setup

1.  **Prerequisites**:
    - Python 3.8+
    - GPU with at least 16GB VRAM (for 4-bit quantization of 7B/8B models).
    - AWS EC2 instance (e.g., `g4dn.xlarge` or `g5.xlarge`) is recommended.

2.  **Install Dependencies**:
    ```bash
    pip install torch transformers accelerate bitsandbytes datasets scikit-learn
    ```

3.  **Data**:
    - Ensure the `processed_data` directory exists in the parent directory (`../processed_data`) containing `train.jsonl` and `validation.jsonl`.

## Usage

1.  **Make the script executable**:
    ```bash
    chmod +x run_icl_eval.sh
    ```

2.  **Run the evaluation**:
    ```bash
    ./run_icl_eval.sh
    ```

    This will:
    - Run Llama-3 (2-shot) on the Validation set.
    - Run Llama-3 (2-shot) on a sample of the Training set.
    - Run Mistral (3-shot) on the Validation set.
    - Run Mistral (3-shot) on a sample of the Training set.

3.  **Results**:
    - Results will be saved in the `results/` directory (created automatically).
    - Files: `llama3_val_results.json`, `mistral_val_results.json`, etc.

## Mac Usage (Apple Silicon)

If you have a Mac with Apple Silicon (M1/M2/M3/M4), you can run the evaluation locally using the `mlx` library.

1.  **Install Mac Dependencies**:
    ```bash
    pip install mlx mlx-lm bert_score scikit-learn datasets tqdm
    ```

2.  **Run the Mac Evaluation**:
    ```bash
    chmod +x run_icl_eval_mac.sh
    ./run_icl_eval_mac.sh
    ```
    This uses 4-bit quantized models from the `mlx-community` (e.g., `mlx-community/Meta-Llama-3-8B-Instruct-4bit`).

## Script Details

### `icl_inference.py`

- **Few-Shot Logic**:
    - **Llama-3**: Uses 2 examples (1 Select, 1 Reject).
    - **Mistral**: Uses 3 examples (2 Select, 1 Reject).
    - **Data Leakage Prevention**: When running on the training set, the specific examples used in the prompt are excluded from the evaluation dataset.
- **Quantization**: Uses `bitsandbytes` 4-bit quantization (`nf4`) to reduce memory usage.

### `run_icl_eval.sh`

- Helper script that calls `icl_inference.py` with appropriate arguments.
- You can modify the `--max_samples` argument in this script to run on the full dataset (remove the flag) or a different number of samples.
