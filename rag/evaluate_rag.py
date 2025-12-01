import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from bert_score import score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rag_inference
import re

# Configuration
TEST_DATA_PATH = "processed_data/test.jsonl"
RAG_EXAMPLES_PATH = "processed_data/rag_examples_grouped.json"

def load_validation_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def parse_response(response):
    # Normalize response
    response = response.strip()
    
    # Extract decision
    decision = "unknown"
    if "SELECT" in response.upper():
        decision = "select"
    elif "REJECT" in response.upper():
        decision = "reject"
        
    # Extract reason (assuming reason follows decision or is the whole text if decision is embedded)
    # Simple heuristic: take the whole text as reason for BERTScore
    reason = response
    
    return decision, reason

def main():
    print("Loading test data...")
    val_data = load_validation_data(TEST_DATA_PATH)
    
    print("Loading RAG components...")
    examples = rag_inference.load_rag_examples(RAG_EXAMPLES_PATH)
    embed_model = SentenceTransformer(rag_inference.EMBEDDING_MODEL)
    index, doc_texts = rag_inference.create_index(examples, embed_model)
    
    # Load LLM using the centralized function in rag_inference
    # This will use the CURRENT_MODEL defined in rag_inference.py
    llm_pipe = rag_inference.load_llm(rag_inference.CURRENT_MODEL)
    
    ground_truth_decisions = []
    predicted_decisions = []
    ground_truth_reasons = []
    generated_reasons = []
    
    print(f"\nStarting evaluation on {len(val_data)} examples...")
    
    for i, item in enumerate(val_data):
        print(f"\nProcessing example {i+1}/{len(val_data)}")
        role = item['Role']
        resume = item['Resume']
        job_description = item['Job_Description']
        
        # Ground truth
        gt_decision = item.get('Decision', '').lower()
        gt_reason = item.get('Reason_for_decision', '')
        
        # Generate response
        response = rag_inference.generate_response(
            role, resume, job_description, index, doc_texts, embed_model, llm_pipe
        )
        
        # Parse prediction
        pred_decision, pred_reason = parse_response(response)
        
        # Store results
        ground_truth_decisions.append(gt_decision)
        predicted_decisions.append(pred_decision)
        ground_truth_reasons.append(gt_reason)
        generated_reasons.append(pred_reason)
        
        print(f"Ground Truth: {gt_decision}")
        print(f"Predicted: {pred_decision}")

        # Save detailed result
        result_entry = {
            "Role": role,
            "Resume": resume,
            "Job_Description": job_description,
            "Ground_Truth_Decision": gt_decision,
            "Ground_Truth_Reason": gt_reason,
            "Predicted_Decision": pred_decision,
            "Predicted_Reason": pred_reason,
            "Full_Response": response
        }
        with open("evaluation_rag_results.jsonl", "a") as f:
            f.write(json.dumps(result_entry) + "\n")
            
    print(f"\nDetailed results saved to evaluation_rag_results.jsonl")
    
    # Calculate Metrics
    print("\n--- Evaluation Results ---")
    
    # Filter out unknowns for classification metrics if necessary, or treat as error
    # Here we treat 'unknown' as a wrong class if it happens
    
    acc = accuracy_score(ground_truth_decisions, predicted_decisions)
    prec = precision_score(ground_truth_decisions, predicted_decisions, labels=["select", "reject"], average='macro', zero_division=0)
    rec = recall_score(ground_truth_decisions, predicted_decisions, labels=["select", "reject"], average='macro', zero_division=0)
    cm = confusion_matrix(ground_truth_decisions, predicted_decisions, labels=["select", "reject"])
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {prec:.4f}")
    print(f"Recall (Macro): {rec:.4f}")
    print("Confusion Matrix (Select, Reject):")
    print(cm)
    
    print("\nCalculating BERTScore...")
    P, R, F1 = score(generated_reasons, ground_truth_reasons, lang="en", verbose=True)
    print(f"BERTScore F1 (Mean): {F1.mean().item():.4f}")

if __name__ == "__main__":
    main()
