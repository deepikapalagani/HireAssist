#!/bin/bash

# Install dependencies if needed (uncomment if running on a fresh instance)
# pip install -r ../requirements.txt

# Create results directory
mkdir -p results

# ==========================================
# VALIDATION EXPERIMENTS
# ==========================================

# --- Llama-3-8B-Instruct ---
echo "Running Llama-3 (Zero-Shot) on Validation..."
python icl_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_zero_shot.json" \
    --prompt_style "zero_shot"

echo "Running Llama-3 (2-Shot Standard) on Validation..."
python icl_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_2shot_std.json" \
    --prompt_style "two_shot_standard"

echo "Running Llama-3 (2-Shot Detailed) on Validation..."
python icl_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_2shot_detailed.json" \
    --prompt_style "two_shot_detailed"

# --- Mistral-7B-Instruct-v0.3 ---
echo "Running Mistral (Zero-Shot) on Validation..."
python icl_inference.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_zero_shot.json" \
    --prompt_style "zero_shot"

echo "Running Mistral (2-Shot Standard) on Validation..."
python icl_inference.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_2shot_std.json" \
    --prompt_style "two_shot_standard"

echo "Running Mistral (2-Shot Detailed) on Validation..."
python icl_inference.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_2shot_detailed.json" \
    --prompt_style "two_shot_detailed"


# ==========================================
# TEST SET EVALUATION (Run after selecting best config)
# ==========================================

# Set the best prompt style for each model based on validation results
# Options: zero_shot, two_shot_standard, two_shot_detailed

BEST_LLAMA_STYLE="two_shot_standard"  # Change this based on validation results
BEST_MISTRAL_STYLE="two_shot_detailed" # Change this based on validation results

# Uncomment to run on Test Set
# echo "Running Llama-3 (Best: $BEST_LLAMA_STYLE) on Test Set..."
# python icl_inference.py \
#     --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
#     --data_path "../processed_data/test.jsonl" \
#     --output_file "results/llama3_test_results.json" \
#     --prompt_style "$BEST_LLAMA_STYLE"

# echo "Running Mistral (Best: $BEST_MISTRAL_STYLE) on Test Set..."
# python icl_inference.py \
#     --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
#     --data_path "../processed_data/test.jsonl" \
#     --output_file "results/mistral_test_results.json" \
#     --prompt_style "$BEST_MISTRAL_STYLE"

echo "Validation Experiments Complete. Check results in results/ folder."
