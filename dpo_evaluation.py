import json
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# ─── Settings ────────────────────────────────────────────────────────────────
model_name = "llama-3.2-1b-dpo-epoch2"
local_dir  = ".../dpo_models/" + model_name
data_dir   = "safe_pair_data/"
max_length = 1024
out_file   = f"safe_rlhf_{model_name}.json"

# ─── Load & Filter Test Set ──────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
dataset = load_from_disk(data_dir)["test"]
print(f"Dataset has {len(dataset)} examples\n")

# ─── Load Model ───────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(local_dir)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

# ─── Log‐Likelihood Function ─────────────────────────────────────────────────
def compute_log_likelihood(prompt: str, response: str) -> float:
    input_ids, labels_list = [], []
    # Token IDs for prompt & responses
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    # Build input IDs
    input_ids += [
        torch.tensor(p_ids + c_ids, dtype = torch.long)
    ]
    # Build labels: mask prompt with -100, keep response tokens
    labels_list += [
        torch.tensor([-100]*len(p_ids) + c_ids, dtype = torch.long)
    ]
    #######################
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    labels_tensor = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    assert input_ids.shape == attention_mask.shape and attention_mask.shape == labels_tensor.shape

    with torch.no_grad():
        out = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels_tensor.to(device))
        mean_nll = out.loss.item()
    return np.exp(-mean_nll)

# ─── Streaming Evaluation & Collect LLs ───────────────────────────────────────
results = []
pbar = tqdm(dataset, total=len(dataset), desc="Eval SAFE-RLHF", unit="ex")
pbar.set_postfix({'acc': '0.00%', 'correct': '0'})
correct = 0
total = 0

for ex in pbar:
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]
    # compute ll
    ll_c = compute_log_likelihood(prompt, chosen)
    ll_r = compute_log_likelihood(prompt, rejected)
    # save into results list
    results.append({
        "prompt": prompt,
        "chosen_resp": chosen,
        "rejected_resp": rejected,
        "ll_chosen": ll_c,
        "ll_rejected": ll_r
    })

    if ll_c > ll_r:
        correct += 1
    total += 1

    pbar.set_postfix({
        'acc': f'{(correct/total)*100:.2f}%',
        'correct': f'{correct}/{total}',
    })

accuracy = correct / total
print(f"Preference Accuracy: {accuracy:.4f}")

pbar.close()

# ─── Write to JSON ────────────────────────────────────────────────────────────
with open(out_file, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} log‐likelihood pairs to {out_file}")