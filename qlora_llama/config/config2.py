import torch

TEST_MODE =  False
print("TEST MODE: ", TEST_MODE)
# --- 1. Configuration ---

MODEL_FAMILY = "llama3" # Options: "llama3", "mistral"
USE_FLASH_ATTENTION = False

# Model and Tokenizer Setup
if MODEL_FAMILY == "llama3":
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    BOS_TOKEN = "<|begin_of_text|>"
    EOS_TOKEN = "<|eot_id|>"
    PAD_TOKEN = "<|padding|>" 
    RESPONSE_START_TOKEN = "<|start_header_id|>assistant<|end_header_id|>\n"
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
elif MODEL_FAMILY == "mistral":
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "<unk>" # Mistral often uses unk as pad if not specified, or we can add one.
    RESPONSE_START_TOKEN = " [/INST] "
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
else:
    raise ValueError(f"Unknown MODEL_FAMILY: {MODEL_FAMILY}")

OUTPUT_DIR = f"./{MODEL_FAMILY}_qlora_output"

# QLoRA/PEFT Parameters
LORA_R = 32          # Increased to 32 for better reasoning capabilities
LORA_ALPHA = 64      # Alpha = 2 * R is a common heuristic
LORA_DROPOUT = 0.05  # Reduced dropout slightly for stability

# Training Parameters
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch size = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
NUM_TRAIN_EPOCHS = 3
SAVE_STEPS = 500
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