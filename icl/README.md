# In-Context Learning (ICL) Inference for HireAssist

This directory contains scripts to evaluate Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.3 on the Resume Screening task using In-Context Learning (Few-Shot).

## Contents

- `icl_inference.py`: The main Python script for running inference (supports Batch Inference).
- `run_icl_eval.sh`: A shell script to automate the evaluation process.
- `setup_aws.sh`: Helper script to set up the environment on AWS Deep Learning AMI.
- `icl_inference_mac.py` & `run_icl_eval_mac.sh`: Optimized scripts for Apple Silicon (MLX).

## AWS Setup (Recommended)

For fastest results, use an AWS EC2 instance.

1.  **Instance**:
    - **Recommended**: `g5.xlarge` (NVIDIA A10G, 24GB VRAM).
    - **Minimum**: `g4dn.xlarge` (NVIDIA T4, 16GB VRAM) - works but slower.

2.  **Setup**:
    ```bash
    cd icl
    source setup_aws.sh
    ```
    *Note*: You will need to authenticate with Hugging Face (`huggingface-cli login`) to access Llama-3.

3.  **Run Evaluation**:
    ```bash
    ./run_icl_eval.sh
    ```
    This runs 3 experiments per model (Zero-shot, 2-shot Standard, 2-shot Detailed) using **Batch Inference** (default batch size 8).

## Mac Usage (Apple Silicon)

Run locally on M1/M2/M3/M4 Macs using `mlx`.

1.  **Install Dependencies**:
    ```bash
    pip install mlx mlx-lm bert_score scikit-learn datasets tqdm
    ```

2.  **Run Evaluation**:
    ```bash
    ./run_icl_eval_mac.sh
    ```

## Experiments

The scripts evaluate the following configurations on the **Validation** set:

1.  **Zero-Shot**: Standard instruction, no examples.
2.  **2-Shot Standard**: Standard instruction + 2 examples.
3.  **2-Shot Detailed**: Detailed step-by-step instruction + 2 examples.

**Metrics**:
- **Accuracy**: Select/Reject classification.
- **BERTScore**: Quality of the generated reasoning (F1).
