#!/bin/bash

# Install dependencies
# pip install mlx mlx-lm bert_score scikit-learn datasets tqdm

# Create results directory
mkdir -p results

# ==========================================
# VALIDATION EXPERIMENTS (Mac M4)
# ==========================================

# --- Llama-3-8B-Instruct (MLX 4-bit) ---
echo "Running Llama-3 (Zero-Shot) on Validation..."
python icl_inference_mac.py \
    --model_name "mlx-community/Meta-Llama-3-8B-Instruct-4bit" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_zero_shot.json" \
    --prompt_style "zero_shot"

echo "Running Llama-3 (2-Shot Standard) on Validation..."
python icl_inference_mac.py \
    --model_name "mlx-community/Meta-Llama-3-8B-Instruct-4bit" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_2shot_std.json" \
    --prompt_style "two_shot_standard"

echo "Running Llama-3 (2-Shot Detailed) on Validation..."
python icl_inference_mac.py \
    --model_name "mlx-community/Meta-Llama-3-8B-Instruct-4bit" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_2shot_detailed.json" \
    --prompt_style "two_shot_detailed"

# --- Mistral-7B-Instruct-v0.3 (MLX 4-bit) ---
echo "Running Mistral (Zero-Shot) on Validation..."
python icl_inference_mac.py \
    --model_name "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_zero_shot.json" \
    --prompt_style "zero_shot"

echo "Running Mistral (2-Shot Standard) on Validation..."
python icl_inference_mac.py \
    --model_name "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_2shot_std.json" \
    --prompt_style "two_shot_standard"

echo "Running Mistral (2-Shot Detailed) on Validation..."
python icl_inference_mac.py \
    --model_name "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_2shot_detailed.json" \
    --prompt_style "two_shot_detailed"

echo "Mac Evaluation Complete. Results saved in results/"
