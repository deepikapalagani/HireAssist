import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configuration
MODELS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2"
}
CURRENT_MODEL = "mistral" # Change this to 'llama' or 'mistral'
print(f"{CURRENT_MODEL=}")
MODEL_NAME = MODELS[CURRENT_MODEL]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_EXAMPLES_PATH = "processed_data/rag_examples_grouped.json"
TOP_K = 1

def load_rag_examples(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_index(examples, model):
    print("Creating FAISS index...")
    documents = []
    for ex in examples:
        # Combine fields for embedding context
        # Grouped format: Role, Select Example, Reject Example
        text = f"Role: {ex['Role']}\n\n" \
               f"Example 1 (Select):\n" \
               f"Job Description: {ex['Job_Description_Select']}\n" \
               f"Resume: {ex['Resume_Select']}\n" \
               f"Decision: {ex['Decision_Select']}\n" \
               f"Reason: {ex['Reason_Select']}\n\n" \
               f"Example 2 (Reject):\n" \
               f"Job Description: {ex['Job_Description_Reject']}\n" \
               f"Resume: {ex['Resume_Reject']}\n" \
               f"Decision: {ex['Decision_Reject']}\n" \
               f"Reason: {ex['Reason_Reject']}"
        documents.append(text)
    
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, documents

def retrieve_examples(query, index, documents, model, k=TOP_K):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    retrieved = []
    for i in indices[0]:
        retrieved.append(documents[i])
    return retrieved

def load_llm(model_key="llama"):
    model_id = MODELS.get(model_key, MODELS["llama"])
    print(f"Loading LLM: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return llm_pipe

def get_prompt(model_key, context, job_description, resume):
    if model_key == "llama":
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Use the following examples to guide your decision and reasoning.

Examples:
{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Given the following Job Description and Resume, please make a hiring decision (SELECT/REJECT) and provide a brief reason.

### Job Description:
{job_description}

### Resume:
{resume}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    elif model_key == "mistral":
        return f"""<s>[INST] You are a helpful assistant. Use the following examples to guide your decision and reasoning.

Examples:
{context}

Given the following Job Description and Resume, please make a hiring decision (SELECT/REJECT) and provide a brief reason.

### Job Description:
{job_description}

### Resume:
{resume} [/INST]"""
    else:
        # Fallback to a generic format
        return f"""System: You are a helpful assistant. Use the following examples to guide your decision and reasoning.

Examples:
{context}

User: Given the following Job Description and Resume, please make a hiring decision (SELECT/REJECT) and provide a brief reason.

### Job Description:
{job_description}

### Resume:
{resume}

Assistant:"""

def generate_response(role, resume, job_description, index, documents, embedding_model, llm_pipeline):
    query = f"Role: {role}\nJob Description: {job_description}\nResume: {resume}"
    
    # Retrieve relevant examples
    retrieved_docs = retrieve_examples(query, index, documents, embedding_model)
    
    # print(f"\n--- Retrieved {len(retrieved_docs)} Documents ---")
    # for i, doc in enumerate(retrieved_docs):
    #     # Extract just the first line or a snippet for brevity
    #     first_line = doc.split('\n')[0]
    #     print(f"Document {i+1}: {first_line}...")

    context = "\n\n".join(retrieved_docs)
    
    # Construct Prompt based on current model
    prompt = get_prompt(CURRENT_MODEL, context, job_description, resume)
    
    # Generate
    outputs = llm_pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=llm_pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    return outputs[0]["generated_text"][len(prompt):]

def main():
    print("Loading RAG examples...")
    examples = load_rag_examples(RAG_EXAMPLES_PATH)
    
    print("Initializing embedding model...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    index, doc_texts = create_index(examples, embed_model)
    
    # Load LLM using the helper function
    llm_pipe = load_llm(CURRENT_MODEL)
    
    # Test Input
    test_role = "Software Engineer"
    test_jd = "We are looking for a Software Engineer with Python and AWS experience."
    test_resume = "Experienced developer with 5 years in Python and cloud infrastructure."
    
    print("\n--- Generating Response for Test Input ---")
    response = generate_response(test_role, test_resume, test_jd, index, doc_texts, embed_model, llm_pipe)
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()
