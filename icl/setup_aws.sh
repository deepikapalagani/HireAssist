#!/bin/bash

# Setup script for AWS Deep Learning AMI (Ubuntu)
# Usage: source setup_aws.sh

echo "Setting up environment..."

# Activate the pre-installed PyTorch environment (usually pytorch_p310 or similar)
# If not using DLAMI, or if you prefer a fresh env:
# conda create -n icl_env python=3.10 -y
# conda activate icl_env

# Install dependencies
# Install dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
# Force upgrade critical libs for Llama-3/Mistral
python -m pip install --upgrade transformers accelerate sentencepiece protobuf bitsandbytes

# Install rest from requirements
pip install -r ../requirements.txt

# Install bert_score and other missing libs if not in requirements (just in case)
python -m pip install bert_score scikit-learn datasets tqdm vllm

echo "Setup complete! You can now run ./run_icl_eval.sh"
