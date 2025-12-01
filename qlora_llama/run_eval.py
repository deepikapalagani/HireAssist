import torch
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import evaluate
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Import config
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config2 import *
from data import load_and_format

def run_evaluation():
    print(f"--- Loading Base Model: {MODEL_NAME} ---")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True, add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})

    # 2. Load Base Model with Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        use_auth_token=True,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "eager",
        local_files_only=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # 3. Load LoRA Adapter
    adapter_path = os.path.join(OUTPUT_DIR, "config2") # Assuming this is where it was saved
    print(f"--- Loading LoRA Adapter from: {adapter_path} ---")
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # 4. Load Validation Dataset
    print("--- Loading Validation Dataset ---")
    validation_dataset = load_and_format(filepath="./processed_data/test.jsonl", n=None) # Load ALL
    print(f"Loaded {len(validation_dataset)} examples.")

    # 5. Run Evaluation Loop
    print("--- Starting Evaluation ---")
    
    predictions = []
    references = []
    accuracies = []
    logs = []
    gt_labels = []
    pred_labels = []
    
    # Initialize BERTScore
    bertscore = evaluate.load("bertscore")

    for example in tqdm(validation_dataset):
        full_text = example['text']
        split_token = "<|start_header_id|>assistant<|end_header_id|>\n"
        
        if split_token not in full_text:
            continue
            
        prompt, ground_truth = full_text.split(split_token, 1)
        prompt += split_token
        
        ground_truth = ground_truth.replace("<|eot_id|>", "").strip()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        predictions.append(generated_text)
        references.append(ground_truth)
        
        # Accuracy Logic (Same as metrics.py)
        gt_lines = ground_truth.strip().split('\n')
        gt_decision = gt_lines[0].strip().upper() if gt_lines else ""
        
        pred_lines = generated_text.strip().split('\n')
        pred_decision = pred_lines[0].strip().upper() if pred_lines else ""
        
        is_correct = 1 if gt_decision in pred_decision else 0
        accuracies.append(is_correct)
        
        # Collect labels for advanced metrics (1=SELECT, 0=REJECT)
        gt_label = 1 if "SELECT" in gt_decision else 0
        pred_label = 1 if "SELECT" in pred_decision else 0
        gt_labels.append(gt_label)
        pred_labels.append(pred_label)
        
        logs.append({
            "prompt": prompt,
            "ground_truth_full": ground_truth,
            "prediction_full": generated_text,
            "gt_decision": gt_decision,
            "pred_decision": pred_decision,
            "is_correct": is_correct
        })

    # 6. Save Logs
    output_file = "final_evaluation_predictions.jsonl"
    print(f"Saving predictions to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

    # 7. Compute & Print Metrics
    accuracy = np.mean(accuracies)
    print(f"\n>>> Final Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, Confusion Matrix
    try:
        precision = precision_score(gt_labels, pred_labels, zero_division=0)
        recall = recall_score(gt_labels, pred_labels, zero_division=0)
        cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1])
        
        print(f">>> Precision (SELECT): {precision:.4f}")
        print(f">>> Recall (SELECT): {recall:.4f}")
        print(">>> Confusion Matrix:")
        print("             Pred:REJECT   Pred:SELECT")
        print(f"True:REJECT     {cm[0][0]:<12}  {cm[0][1]:<12}")
        print(f"True:SELECT     {cm[1][0]:<12}  {cm[1][1]:<12}")
    except Exception as e:
        print(f"Warning: Failed to compute advanced metrics: {e}")
    
    # BERTScore
    try:
        print("Computing BERTScore...")
        bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
        bert_f1 = np.mean(bert_results['f1'])
        print(f">>> Final BERTScore F1: {bert_f1:.4f}")
    except Exception as e:
        print(f"Warning: Failed to compute BERTScore: {e}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_evaluation()
    else:
        print("CUDA is not available. Please run on a GPU machine.")
