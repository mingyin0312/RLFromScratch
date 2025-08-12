import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import numpy as np
from typing import Dict
import json
import logging

# Disable VLLM's progress bars
logging.getLogger("vllm").setLevel(logging.WARNING)

# Constants (updated to match training code)
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."

def extract_xml_answer(text: str) -> str:
    """Extracts the answer from a response using the <answer> tag."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer from the dataset if it uses the '####' delimiter."""
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None

def evaluate_model(
    model_path: str,
    batch_size: int = 8,
    num_samples: int = None,
    save_results: bool = True,
    gpu_memory_utilization: float = 0.8,
) -> Dict:
    print("Initializing evaluation...")

    # Initialize VLLM and tokenizer with a progress indicator
    with tqdm(total=2, desc="Loading model components") as pbar:
        llm = LLM(
            model=model_path,
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=1024,
            device="cuda:0",
            enable_chunked_prefill=True,
        )
        pbar.update(1)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=1024,
            padding_side='right',
            truncation_side='right'
        )
        pbar.update(1)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,  # Matches the max_completion_length from training
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Load test dataset (GSM8K 'main' split)
    print("Loading dataset...")
    #dataset = load_dataset('openai/gsm8k', 'main', split='test')
    dataset = load_from_disk('gsm8k_data/')['test']
    if num_samples:
        dataset = dataset.select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    # Create progress bar for processing samples
    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['question'])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': "What is 2+2?"},
                {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
                {'role': 'user', 'content': q.strip()}
            ]
            for q in batch_data['question']
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        # Generate responses using VLLM
        outputs = llm.generate(
            formatted_prompts,
            sampling_params,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = output.outputs[0].text

            # Extract the generated answer using XML tags
            generated_answer = extract_xml_answer(response)
            true_answer = extract_hash_answer(batch_data['answer'][j])

            # Store the result
            result = {
                'question': batch_data['question'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer == true_answer
            }
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)


        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()

    # Calculate overall metrics
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }

    # Save evaluation results to a JSON file
    if save_results:
        save_path = f"gsm8k_eval_results.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics

if __name__ == "__main__":
    print("Starting GSM8K evaluation...")
    checkpoint_path = "Your_model_path"  # Update path as needed

    metrics = evaluate_model(
        model_path=checkpoint_path,
        batch_size=4,
        num_samples=None,
        save_results=True,
        gpu_memory_utilization=0.8,
    )

    print("\nFinal Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")


