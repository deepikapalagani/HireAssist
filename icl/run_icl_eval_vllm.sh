#!/bin/bash

# Create results directory
mkdir -p results

# ==========================================
# VALIDATION EXPERIMENTS (vLLM)
# ==========================================

# --- Llama-3-8B-Instruct ---
echo "Running Llama-3 (Zero-Shot) on Validation [vLLM]..."
python icl_inference_vllm.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_zero_shot_vllm.json" \
    --prompt_style "zero_shot" \
    --quantization "bitsandbytes" \
    --gpu_memory_utilization 0.85 \
    --enforce_eager

echo "Running Llama-3 (2-Shot Standard) on Validation [vLLM]..."
python icl_inference_vllm.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_2shot_std_vllm.json" \
    --prompt_style "two_shot_standard" \
    --quantization "bitsandbytes" \
    --gpu_memory_utilization 0.85 \
    --enforce_eager

echo "Running Llama-3 (2-Shot Detailed) on Validation [vLLM]..."
python icl_inference_vllm.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/llama3_val_2shot_detailed_vllm.json" \
    --prompt_style "two_shot_detailed" \
    --quantization "bitsandbytes" \
    --gpu_memory_utilization 0.85 \
    --enforce_eager

# --- Mistral-7B-Instruct-v0.3 ---
echo "Running Mistral (Zero-Shot) on Validation [vLLM]..."
python icl_inference_vllm.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_zero_shot_vllm.json" \
    --prompt_style "zero_shot" \
    --quantization "bitsandbytes" \
    --gpu_memory_utilization 0.85 \
    --enforce_eager

echo "Running Mistral (2-Shot Standard) on Validation [vLLM]..."
python icl_inference_vllm.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_2shot_std_vllm.json" \
    --prompt_style "two_shot_standard" \
    --quantization "bitsandbytes" \
    --gpu_memory_utilization 0.85 \
    --enforce_eager

echo "Running Mistral (2-Shot Detailed) on Validation [vLLM]..."
python icl_inference_vllm.py \
    --model_name "mistralai/Mistral-7B-Instruct-v0.3" \
    --data_path "../processed_data/validation.jsonl" \
    --output_file "results/mistral_val_2shot_detailed_vllm.json" \
    --prompt_style "two_shot_detailed" \
    --quantization "bitsandbytes" \
    --gpu_memory_utilization 0.85 \
    --enforce_eager

echo "vLLM Validation Experiments Complete. Check results in results/ folder."
