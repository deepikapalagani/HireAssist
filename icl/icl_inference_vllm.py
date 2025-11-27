import argparse
import json
import os
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from bert_score import score
from vllm import LLM, SamplingParams

# --- Instructions ---

INSTRUCTION_STANDARD = (
    "You are an HR recruiter. Analyze the job description and evaluate if the provided resume is a good fit. "
    "Provide exactly one sentence of reasoning, followed by a final decision of either [select] or [reject]. "
    "Strictly follow the target format.\n\n"
)

INSTRUCTION_DETAILED = (
    "You are an expert HR recruiter. Your task is to rigorously evaluate a candidate's resume against a specific job description.\n"
    "1. Analyze the Job Description to identify key technical skills, required experience years, and core responsibilities.\n"
    "2. Analyze the Resume to find evidence of these specific skills and experience.\n"
    "3. Compare the two. Look for gaps in years of experience or missing critical skills.\n"
    "4. Provide a single sentence reasoning justifying your decision based on this comparison.\n"
    "5. Conclude with a final decision: [select] or [reject].\n"
    "Strictly follow the target format.\n\n"
)

# --- Few-Shot Examples ---

EXAMPLE_1 = {
    "Job_Description": "We need a E-commerce Specialist to enhance our team's technical capabilities and contribute to solving complex complex problems.",
    "Resume": "Here is a professional resume for Tina Roth:\n\nTina Roth\nE-commerce Specialist\n\nContact Information:\n\n* Email: [](mailto:)\n* Phone: \n* LinkedIn: \n* Address: 123 Main St, Anytown, USA 12345\n\nProfessional Summary:\nResults-driven E-commerce Specialist with 5+ years of experience in product listing, inventory management, SEO for e-commerce, online advertising, and customer service. Proven track record of improving website traffic, increasing sales, and enhancing customer satisfaction. Proficient in a range of e-commerce platforms and tools, with a strong understanding of online market trends and consumer behavior.\n\nWork Experience:\n\nE-commerce Specialist, Online Retailer (2018-Present)\n\n* Managed product listings for over 10,000 SKUs, resulting in a 25% increase in product visibility and a 15% increase in sales\n* Implemented and executed SEO strategies, improving website rankings by 30% and organic traffic by 50%\n* Developed and managed online advertising campaigns across Google Ads, Facebook Ads, and Instagram Ads, resulting in a 20% increase in conversions\n* Provided exceptional customer service, responding to 99% of customer inquiries within 2 hours and resolving 95% of issues on the first contact\n* Collaborated with cross-functional teams to improve website user experience, resulting in a 15% increase in conversion rates\n\nE-commerce Coordinator, Retailer (2015-2018)\n\n* Assisted in the creation and maintenance of product listings, resulting in a 10% increase in product visibility\n* Managed inventory levels, ensuring 95% accuracy and reducing stockouts by 25%\n* Conducted keyword research and implemented SEO strategies, improving website rankings by 15%\n* Provided customer support via phone, email, and chat, resolving 90% of issues on the first contact\n\nEducation:\n\n* Bachelor's Degree in Business Administration, Anytown University (2015)\n\nSkills:\n\n* Product listing and management\n* Inventory management\n* SEO for e-commerce\n* Online advertising (Google Ads, Facebook Ads, Instagram Ads)\n* Customer service\n* E-commerce platforms (Shopify, Magento, WooCommerce)\n* Analytics tools (Google Analytics, Adobe Analytics)\n* Content management systems (WordPress, Drupal)\n\nAchievements:\n\n* Winner of the \"E-commerce Excellence Award\" at the 2019 Online Retailer Conference\n* Featured speaker at the 2018 E-commerce Summit, presenting on \"SEO Strategies for E-commerce Success\"\n* Member of the E-commerce Industry Association, participating in industry events and webinars\n\nCertifications:\n\n* Google Analytics Certification, Google (2016)\n* HubSpot Inbound Marketing Certification, HubSpot (2017)\n\nI hope this helps! Remember to tailor your resume to the specific job you're applying for, and highlight your unique skills and experiences.",
    "Reason_for_decision": "Lacks hands-on experience with cloud platforms.",
    "Decision": "Reject"
}

EXAMPLE_2 = {
    "Job_Description": "We are looking for an experienced Data Scientist to join our team and help drive groundbreaking solutions in software engineering.",
    "Resume": "Here's a sample resume for Jose Cortez:\n\nJose Cortez\nContact Information:\n\n* Email: [](mailto:)\n* Phone: \n* LinkedIn: \n* GitHub: \n\nProfessional Summary:\n\nHighly motivated and detail-oriented Data Scientist with 5+ years of experience in developing and deploying machine learning models using Python, TensorFlow, and Deep Learning techniques. Proven track record of delivering high-quality data-driven insights to drive business growth and improvement. Skilled in data visualization, statistical modeling, and SQL database management.\n\nTechnical Skills:\n\n* Programming languages: Python, R, SQL\n* Deep Learning frameworks: TensorFlow, Keras\n* Data Visualization tools: Matplotlib, Seaborn, Plotly\n* Data Analysis libraries: Pandas, NumPy, SciPy\n* Database management: MySQL, PostgreSQL\n* Operating Systems: Windows, Linux, macOS\n* Agile methodologies: Scrum, Kanban\n\nProfessional Experience:\n\nData Scientist, ABC Company (2018-Present)\n\n* Developed and deployed multiple machine learning models using TensorFlow and Keras to predict customer churn, sales, and demand\n* Collaborated with cross-functional teams to design and implement data-driven solutions to business problems\n* Created interactive dashboards using Plotly and Tableau to visualize key business metrics and trends\n* Worked with stakeholders to identify business requirements and develop data-driven insights to inform strategic decisions\n* Built and maintained large datasets using SQL and Pandas, and optimized data processing pipelines using Apache Spark\n\nSenior Data Analyst, DEF Startup (2015-2018)\n\n* Analyzed large datasets using statistical modeling techniques to identify trends and patterns\n* Developed and maintained reports and dashboards using Tableau and Power BI to visualize key business metrics\n* Collaborated with product teams to design and implement A/B testing and experimentation frameworks\n* Built and maintained SQL databases to support data analysis and reporting\n* Worked with stakeholders to develop data-driven insights to inform product development and marketing strategies\n\nEducation:\n\n* Master of Science in Statistics, University of California, Los Angeles (2015)\n* Bachelor of Science in Mathematics, University of California, Berkeley (2013)\n\nAchievements:\n\n* Winner, Kaggle Competition: \"Predicting Customer Churn\" (2019)\n* Featured Speaker, Data Science Conference: \"Applying Deep Learning to Business Problems\" (2020)\n* Published Research Paper: \"A Novel Approach to Predicting Sales using Deep Learning and Time Series Analysis\" (2019)\n\nCertifications:\n\n* Certified Data Scientist, Data Science Council of America (2019)\n* Certified Analytics Professional, Institute for Operations Research and the Management Sciences (2018)\n\nReferences:\n\nAvailable upon request.\n\nI hope this sample resume helps! Remember to customize your resume to fit your specific experience and the job you're applying for.",
    "Reason_for_decision": "Strong technical skills in AI and ML.",
    "Decision": "Select"
}

FEW_SHOT_EXAMPLES = [EXAMPLE_1, EXAMPLE_2]

def format_example(job_desc, resume, reason=None, decision=None, is_target=False):
    """Formats a single example."""
    text = (
        f"Input:\n"
        f"Job Description: {job_desc}\n"
        f"Resume: {resume}\n\n"
        "Question: Is this candidate a good fit?\n"
    )
    if is_target:
        text += "Output format:\nReasoning: <one sentence reasoning>\nDecision: \"select\" or \"reject\""
    else:
        text += (
            f"Output format:\n"
            f"Reasoning: {reason}\n"
            f"Decision: {decision}"
        )
    return text

def get_prompt(model_name, job_description, resume, prompt_style):
    """
    Generates the prompt based on the model type and prompt style.
    """
    
    # Determine Instruction
    if prompt_style == "two_shot_detailed":
        instruction = INSTRUCTION_DETAILED
    else:
        instruction = INSTRUCTION_STANDARD

    # Determine Examples
    examples_text = ""
    if "two_shot" in prompt_style:
        for ex in FEW_SHOT_EXAMPLES:
            examples_text += format_example(ex['Job_Description'], ex['Resume'], ex['Reason_for_decision'], ex['Decision']) + "\n\n"

    current_query = format_example(job_description, resume, is_target=True)

    # Construct Prompt based on Model Template
    if "llama" in model_name.lower():
        # Llama-3 format
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful and precise hiring assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}"
            f"{examples_text}"
            f"{current_query}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
    elif "mistral" in model_name.lower():
        # Mistral format: <s>[INST] Instruction + Examples + Query [/INST]
        prompt = f"<s>[INST] {instruction}{examples_text}{current_query} [/INST]"
    else:
        # Generic fallback
        prompt = f"{instruction}{examples_text}{current_query}"

    return prompt

def parse_output(response):
    """
    Parses the model output to extract Reasoning and Decision.
    """
    response_lower = response.lower()
    
    # Extract Decision
    decision = "Unknown"
    if "decision: select" in response_lower:
        decision = "Select"
    elif "decision: reject" in response_lower:
        decision = "Reject"
    else:
        # Fallback
        if "select" in response_lower and "reject" not in response_lower:
            decision = "Select"
        elif "reject" in response_lower:
            decision = "Reject"
            
    # Extract Reasoning
    reasoning = ""
    try:
        match = re.search(r"reasoning:(.*?)(?:decision:|$)", response_lower, re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
        else:
            parts = response.split("Decision:")
            if len(parts) > 1:
                reasoning = parts[0].strip()
            else:
                reasoning = response.strip()
    except Exception:
        reasoning = response.strip()
        
    return decision, reasoning

def main():
    parser = argparse.ArgumentParser(description="Run ICL Inference with vLLM")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--data_path", type=str, default="../processed_data/validation.jsonl", help="Path to data JSONL file")
    parser.add_argument("--output_file", type=str, default="results.json", help="Path to save results")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--prompt_style", type=str, choices=["zero_shot", "two_shot_standard", "two_shot_detailed"], default="zero_shot", help="Prompting strategy")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quantization", type=str, default=None, help="Quantization method (e.g., bitsandbytes, awq)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Fraction of GPU memory to use")
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs to save memory")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} with vLLM")
    print(f"Prompt Style: {args.prompt_style}")
    
    try:
        # Initialize vLLM
        llm = LLM(
            model=args.model_name,
            quantization=args.quantization,
            trust_remote_code=True,
            dtype="half" if args.quantization is None else "auto",
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager
        )
        
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=256,
            stop=["<|eot_id|>", "</s>"] # Stop tokens for Llama/Mistral
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from {args.data_path}")
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Running on {len(data)} samples")

    # Prepare prompts
    prompts = []
    for item in data:
        job_desc = item.get('Job_Description', '')
        resume = item.get('Resume', '')
        prompt = get_prompt(args.model_name, job_desc, resume, args.prompt_style)
        prompts.append(prompt)

    print("Starting inference...")
    
    # Generate
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    true_labels = []
    pred_labels = []
    true_reasonings = []
    pred_reasonings = []
    
    # Create log file path
    log_file = args.output_file.replace(".json", "_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Intermediate Metrics for {args.model_name} ({args.prompt_style}) [vLLM]\n")
        f.write("================================================\n")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        item = data[i]
        
        true_label = item.get('Decision', '').capitalize()
        true_reason = item.get('Reason_for_decision', '')
        
        decision, reasoning = parse_output(generated_text)
        
        results.append({
            "prompt": prompts[i],
            "generated_text": generated_text,
            "true_label": true_label,
            "predicted_label": decision,
            "true_reasoning": true_reason,
            "predicted_reasoning": reasoning
        })
        
        true_labels.append(true_label)
        pred_labels.append(decision)
        true_reasonings.append(true_reason)
        pred_reasonings.append(reasoning)
        
        # Debug logging
        if args.debug and (i < 5 or (i + 1) % 50 == 0):
            with open(log_file, "a") as f:
                f.write(f"\n[DEBUG Sample {i+1}]\n")
                f.write(f"True: {true_label}\n")
                f.write(f"Pred: {decision}\n")
                f.write(f"Raw Output: {generated_text.strip()}\n")
                f.write("-" * 30 + "\n")

    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    # BERTScore
    print("\nCalculating BERTScore...")
    try:
        P, R, F1 = score(pred_reasonings, true_reasonings, lang="en", verbose=True)
        bertscore_f1 = F1.mean().item()
        print(f"BERTScore F1: {bertscore_f1:.4f}")
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        bertscore_f1 = 0.0

    # Save results
    output_data = {
        "config": {
            "model": args.model_name,
            "prompt_style": args.prompt_style,
            "samples": len(data),
            "backend": "vllm"
        },
        "metrics": {
            "accuracy": accuracy,
            "bertscore_f1": bertscore_f1
        },
        "predictions": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
