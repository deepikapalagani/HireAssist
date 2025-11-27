import torch
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

# To import data and config
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import load_and_format
from config.config1 import *
from metrics import EvaluationCallback

# --- Main Execution ---

def run_qlora_finetuning():
    """Executes the QLoRA fine-tuning process."""
    print(f"--- 1. Loading Model and Tokenizer: {MODEL_NAME} ---")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
    
    # Crucially, add a padding token if the tokenizer doesn't have one (Llama 3 doesn't by default)
    # The default Llama 3 tokenizer only has BOS and EOS (eot_id).
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN}) 

    # 1. Define 4-bit Quantization Configuration for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Optimized for fine-tuning
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load the Model with Quantization
    # We load as a CausalLM (Generative Model) because the task is sequence generation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distributes the model across available GPUs
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        use_auth_token=True,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "eager",
    )
    print(f"Using attention implementation: {USE_FLASH_ATTENTION}")
    # 3. Resize embeddings if PAD token was added
    model.resize_token_embeddings(len(tokenizer))
    
    # 4. Freeze all parameters (4-bit weights are already non-trainable)
    model.config.use_cache = False # Required for gradient checkpointing
    model.config.pretraining_tp = 1 # Recommended for Llama
    
    # Prepare model for k-bit training (enables gradient checkpointing, input require grads, etc.)
    model = prepare_model_for_kbit_training(model)

    print("--- 2. Preparing Dataset and Tokenization ---")
    
    # Create / Load Dataset (Using mock data here)
    train_dataset = load_and_format(filepath="./processed_data/train.jsonl", n=5 if TEST_MODE else None)
    validation_dataset = load_and_format(filepath="./processed_data/validation.jsonl", n=5 if TEST_MODE else None)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    tokenized_validation_dataset = validation_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(validation_dataset)}")

    print("--- 3. Configuring QLoRA (PEFT) ---")

    # 5. Define LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none", # Recommended setting for most tasks
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    # 6. Apply LoRA to the Model
    model = get_peft_model(model, peft_config)
    print("Trainable parameters after QLoRA:")
    model.print_trainable_parameters()
    # Should show only a tiny fraction (e.g., 0.04%) of total parameters are trainable.

    print("--- 4. Setting up Training Arguments ---")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        fp16=FP16,
        bf16=BF16,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit", # Optimized AdamW for QLoRA
        eval_strategy="steps",
        eval_steps=SAVE_STEPS if not TEST_MODE else 1, # Evaluate as often as we save (or every step in test)
        load_best_model_at_end=True if not TEST_MODE else False,
        gradient_checkpointing=True,
    )

    # Data Collator (standard language modeling collator for next-token prediction)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False # Causal Language Modeling (next token prediction)
    )

    print("--- 5. Starting Training ---")

    # 7. Initialize Trainer and Start Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        callbacks=[EvaluationCallback(tokenized_validation_dataset, tokenizer, num_samples=5 if TEST_MODE else 20)]
    )

    trainer.train()

    # 8. Save the final PEFT adapter weights
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "config1"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "config1"))
    print("\nTraining complete. PEFT adapter saved.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        run_qlora_finetuning()
    else:
        print("CUDA is not available. Please run this script on a machine with an NVIDIA GPU.")