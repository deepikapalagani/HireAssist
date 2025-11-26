import torch

TEST_MODE = True
# --- 1. Configuration ---

# Model and Tokenizer Setup
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./llama3_qlora_output"
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|eot_id|>"
PAD_TOKEN = "<|padding|>" # Or use EOT/BOS if the tokenizer is configured to do so

# QLoRA/PEFT Parameters
LORA_R = 16          # LoRA attention dimension
LORA_ALPHA = 16      # Alpha parameter for LoRA scaling
LORA_DROPOUT = 0.1   # Dropout probability for LoRA layers
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] #TODO: verify 

# Training Parameters
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 100
LOGGING_STEPS = 10
FP16 = not torch.cuda.is_bf16_supported() # Use FP16 if BF16 is not supported
BF16 = torch.cuda.is_bf16_supported()     # Use BF16 if supported
WARMUP_RATIO = 0.03

# --- TEST MODE OVERRIDES for minimal training time ---
if TEST_MODE:
    print("--- WARNING: TEST MODE ACTIVE. Training parameters heavily reduced. ---")
    NUM_TRAIN_EPOCHS = 0.001  # Run for a tiny fraction of an epoch
    MAX_STEPS = 5             # Only run 5 total update steps (highly minimal)
    MAX_SEQ_LENGTH = 256      # Use a shorter sequence length
    SAVE_STEPS = 999999       # Don't save checkpoints
    LOGGING_STEPS = 1         # Log every step
# --- END TEST MODE ---