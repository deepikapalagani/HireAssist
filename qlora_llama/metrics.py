import torch
import evaluate
import numpy as np
from transformers import TrainerCallback
from tqdm import tqdm
import math

class EvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, generation_config=None, num_samples=50):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.bertscore = evaluate.load("bertscore")
        self.generation_config = generation_config
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Event called at the end of an evaluation phase.
        We will generate responses for a subset of the eval dataset and compute metrics.
        """
        print("\n--- Running Generation-based Evaluation ---")
        
        # 1. Select a random sample from the validation set to save time
        if len(self.eval_dataset) > self.num_samples:
            indices = np.random.choice(len(self.eval_dataset), self.num_samples, replace=False)
            eval_sample = self.eval_dataset.select(indices)
        else:
            eval_sample = self.eval_dataset

        model = kwargs['model']
        model.eval() # Ensure model is in eval mode
        
        predictions = []
        references = []
        accuracies = []

        print(f"Generating responses for {len(eval_sample)} examples...")
        
        for example in tqdm(eval_sample):
            # The 'text' field contains the full conversation. We need to split it to get the prompt.
            # Format: ... <|start_header_id|>assistant<|end_header_id|>\n{output}<|eot_id|>
            
            full_text = example['text']
            split_token = "<|start_header_id|>assistant<|end_header_id|>\n"
            
            if split_token not in full_text:
                continue # Skip malformed examples
                
            prompt, ground_truth = full_text.split(split_token, 1)
            prompt += split_token # Re-attach the assistant header so the model knows to generate
            
            # Remove the EOS token from ground truth for comparison
            ground_truth = ground_truth.replace("<|eot_id|>", "").strip()
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100, # Limit generation length
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False # Greedy decoding for deterministic evaluation
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            predictions.append(generated_text)
            references.append(ground_truth)
            
            # --- Accuracy Metric (Specific to this task) ---
            # Format: Reasoning\nDecision
            
            # Get Ground Truth Decision
            gt_lines = ground_truth.strip().split('\n')
            gt_decision = gt_lines[0].strip().upper() if gt_lines else ""
            
            # Get Predicted Decision
            pred_lines = generated_text.strip().split('\n')
            pred_decision = pred_lines[0].strip().upper() if pred_lines else ""
            
            # Check if the decision matches
            # We check if the predicted decision line contains the ground truth decision (e.g. "[SELECT]" vs "SELECT")
            is_correct = 1 if gt_decision in pred_decision else 0
            accuracies.append(is_correct)

        # 2. Compute Metrics
        accuracy = np.mean(accuracies)
        
        # BERTScore
        try:
            bert_results = self.bertscore.compute(predictions=predictions, references=references, lang="en")
            bert_f1 = np.mean(bert_results['f1'])
        except Exception as e:
            print(f"Warning: Failed to compute BERTScore: {e}")
            bert_f1 = 0.0

        # Perplexity (derived from eval_loss if available)
        perplexity = math.exp(metrics.get("eval_loss", 0)) if metrics and "eval_loss" in metrics else None

        print(f"\n>>> Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"BERTScore F1: {bert_f1:.4f}")
        if perplexity:
            print(f"Perplexity: {perplexity:.4f}")
        print("---------------------------------------")
        
        # Log to wandb/tensorboard if available (via the metrics dict, though this modifies it in place)
        if metrics is not None:
            metrics["eval_accuracy"] = accuracy
            metrics["eval_bertscore_f1"] = bert_f1
            if perplexity:
                metrics["eval_perplexity"] = perplexity

