#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# ==========================================
# TEST SET EVALUATION
# ==========================================

# 1. Llama-3-8B-Instruct (Best: Zero-Shot)
echo "Running Llama-3 (Zero-Shot) on Test Set..."
python icl_inference.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/test.jsonl" \
    --output_file "results/llama3_test_zero_shot.json" \
    --prompt_style "zero_shot"

# 2. Mistral-7B-Instruct-v0.3 (Best: Two-Shot Detailed)
echo "Running Mistral (2-Shot Detailed) on Test Set..."
python icl_inference.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/test.jsonl" \
    --output_file "results/mistral_test_2shot_detailed.json" \
    --prompt_style "two_shot_detailed"

echo "Test Set Evaluation Complete. Check results in results/ folder."
