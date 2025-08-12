import os
import re
import math
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

# extract the true answer 
def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None

# extract the generated answer
def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def compute_format_score(batch_responses):
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    matches = [bool(re.match(pattern, g_a)) for g_a in batch_responses]
    format_scores = [1.0 if match else 0.0 for match in matches]
    return format_scores

def compute_reward(batch_answers, answers):
    """Reward function that checks if the answer is correct."""
    reward_scores = [2.0 if g_a == a else 0.0 for g_a, a in zip(batch_answers, answers)]
    return reward_scores
    
# learning rate schedule
def get_lr(it, max_steps, warmup_steps = None, max_lr=1e-5, min_lr=1e-6):
    warmup_steps = int(0.1*max_steps)
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# --------------------------------------
# Constants: Chat Prompt Templates
# --------------------------------------
SYSTEM_PROMPT = (
    """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The assistant first thinks about the reasoning process in the mind and then provides the user
    with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
    <answer> </answer> tags, respectively.
    Example:
    <reasoning> ... </reasoning>
    <answer>42</answer>
    """
)
TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."
# --------------------------------------
# Distributed Setup
# --------------------------------------
# Initialize the process group for multi-GPU training
dist.init_process_group(backend="nccl")
world_size = dist.get_world_size()
rank = dist.get_rank()
# Determine local GPU id for this process
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
master_process = rank == 0

# Set random seeds uniquely per process for reproducibility
seed = 42 + rank
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --------------------------------------
# Hyperparameters
# --------------------------------------
max_grad_norm = 0.1          # For gradient clipping
ppo_clip_range = 0.2         # ε for PPO clipping
kl_coef = 0.005                # Multiplier for KL divergence penalty
batch_size = 4               # Prompts per batch per GPU
K = 4                        # Number of samples (responses) per prompt
ppo_epochs = 4               # PPO update epochs per batch
max_new_tokens = 256         # Max tokens to generate per sample
num_epochs = 1               # Number of training epochs
weight_decay = 0.1  
initial_learning_rate = 1e-5

# --------------------------------------
# Load Model & Tokenizer
# --------------------------------------
model_name = "Your_favorite_model" # what I use: Llama-3.2-1B-Instruct
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure pad token is defined (use eos as pad)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------
# Model Setup
# --------------------------------------

# Load policy model (trainable)
policy_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
policy_model.train()
# Load reference model (fixed)
ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(local_rank)
ref_model.eval()
# Freeze reference model parameters
for p in ref_model.parameters():
    p.requires_grad = False

# Wrap the policy model with DDP for gradient synchronization
policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank)

# Setup optimizer
param_dict = {pn: p for pn, p in policy_model.named_parameters() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr = initial_learning_rate)
optimizer.zero_grad(set_to_none = True)

# --------------------------------------
# Prepare Log
# --------------------------------------
if master_process:
    log_dir = ".../grpo_models"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

# --------------------------------------
# Load and Shard Dataset
# --------------------------------------
# Load GSM8K training split
train_data = load_from_disk('gsm8k_data/')["train"]

class WrapperDataset(Dataset):
    """Dataset wrapper for preference data: yields (prompt, answer) pairs."""
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["answer"]

train_data = WrapperDataset(train_data)

def collate_fn(batch):
    prompts, answers = [], []
    for question, answer in batch:
        chat_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "What is 2+2?"},
            {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
            {'role': 'user', 'content': question}
        ]
        prompt = tokenizer.apply_chat_template(
            chat_prompt,
            tokenize = False,
            add_generation_prompt = True
        )
        prompts.append(prompt)
        answers.append(extract_hash_answer(answer))

    return prompts, answers

# Set up DataLoader with DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=world_size, rank=rank, shuffle=True
)
train_loader = DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=4,    
)

max_steps = len(train_loader)
if master_process:
    print(f"⚙️  It will run {max_steps} steps per epoch.")

 
# --------------------------------------
# Training Loop
# --------------------------------------

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for step, (prompts, answers) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        prompt_enc = tokenizer(
            prompts,
            return_tensors = 'pt',
            padding = True,
            padding_side = 'left',
            truncation = True
        )

        input_ids = prompt_enc["input_ids"].to(local_rank) # (B, prompt_len) and left_pad
        attention_mask = prompt_enc["attention_mask"].to(local_rank)

        # ------------------------------
        # Generate K Samples Per Prompt
        # ------------------------------
        policy_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                explore_generations = policy_model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        num_return_sequences=K,
                        top_p=0.9,
                        temperature=1.0,
                        eos_token_id=tokenizer.eos_token_id
                    ) # (batch_size * K, prompt_len + max_new_tokens)
        policy_model.train()
        prompt_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # --------------------------------------
        # Compute Masks & Labels
        # --------------------------------------
        batch_attention_mask = (explore_generations != tokenizer.pad_token_id).long() # [batch_size * K, seq_len]
        batch_action_mask = batch_attention_mask.clone()
        batch_action_mask[:, :prompt_len] = 0 # [batch_size * K, seq_len]
        labels = explore_generations.clone() # [batch_size * K, seq_len]
        labels[batch_action_mask == 0] = -100
        # --------------------------------------
        # Compute Old Logprobs
        # --------------------------------------
        policy_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_old = policy_model(explore_generations,  batch_attention_mask, use_cache=False)
            logits_old = out_old.logits # [batch_size * K, seq_len, vocab_size]
        policy_model.train()
        # shift logits and labels
        logits_old = logits_old[:, :-1, :].contiguous() # [batch_size * K, seq_len - 1, vocab_size]
        labels_old = labels[:, 1:].contiguous() # [batch_size * K, seq_len - 1]
        logprobs_old = -F.cross_entropy(
            logits_old.view(-1, logits_old.shape[-1]), labels_old.view(-1), reduction = 'none', ignore_index=-100
        ).view(logits_old.shape[0], -1) # [batch_size * K, seq_len - 1]
        assert batch_action_mask.shape[-1] == logprobs_old.shape[-1] + 1 and batch_action_mask.shape == batch_attention_mask.shape
        logprobs_old = logprobs_old.view(batch_size, K, -1) # [batch_size, K, seq_len - 1]
        # --------------------------------------
        # Compute Reference Logprobs
        # --------------------------------------
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_ref = ref_model(explore_generations,  batch_attention_mask, use_cache=False)
            logits_ref = out_ref.logits # [batch_size * K, seq_len, vocab_size]
        # shift logits and labels
        logits_ref = logits_ref[:, :-1, :].contiguous() # [batch_size * K, seq_len - 1, vocab_size]
        labels_ref = labels[:, 1:].contiguous() # [batch_size * K, seq_len - 1]
        logprobs_ref = -F.cross_entropy(
            logits_ref.view(-1, logits_ref.shape[-1]), labels_ref.view(-1), reduction = 'none', ignore_index=-100
        ).view(logits_ref.shape[0], -1) # [batch_size * K, seq_len - 1]
        logprobs_ref = logprobs_ref.view(batch_size, K, -1) # [batch_size, K, seq_len - 1]
        assert logprobs_old.shape == logprobs_ref.shape
        # --------------------------------------
        # Compute Advantages
        # --------------------------------------
        batch_responses_ids = explore_generations[:, prompt_len:] # (batch_size*K, response_length) right pad
        batch_responses = tokenizer.batch_decode(batch_responses_ids, skip_special_tokens = True) # (batch_size*K, response_text_length)
        batch_answers = [extract_xml_answer(batch_responses[i]) for i in range(len(batch_responses))] # (batch_size*K, generated_answer_length) str
        answers_K = [a for a in answers for _ in range(K)]
        assert len(batch_answers) == len(answers_K)
        batch_format_scores = compute_format_score(batch_responses) # (batch_size*K, 1)
        batch_reward_scores = compute_reward(batch_answers, answers_K) # (batch_size*K, 1)
        batch_rewards = torch.tensor([bfs + brs for bfs, brs in zip(batch_format_scores, batch_reward_scores)], dtype=torch.float16)
        batch_rewards = batch_rewards.view(batch_size, K) # (batch_size, K)
        batch_advantages = (batch_rewards - batch_rewards.mean(dim = -1, keepdim = True)) / batch_rewards.std(dim = -1, keepdim = True).clamp_min(1e-6)
        batch_advantages = batch_advantages.to(local_rank) # (batch_size, K)
        assert batch_advantages.shape == (batch_size, K)
        batch_advantages = batch_advantages.unsqueeze(2).expand_as(logprobs_ref) # [batch_size, K, seq_len - 1]
        assert batch_advantages.shape == logprobs_ref.shape
        # --------------------------------------
        # PPO Update: multiple epochs per batch
        # --------------------------------------
        for ppo_epoch in range(ppo_epochs):
            # --------------------------------------
            # Compute New Logprobs
            # --------------------------------------
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out_new = policy_model(explore_generations,  batch_attention_mask, use_cache = False)
            logits_new = out_new.logits # [batch_size * K, seq_len, vocab_size]
            # shift logits and labels
            logits_new = logits_new[:, :-1, :].contiguous() # [batch_size * K, seq_len - 1, vocab_size]
            labels_new = labels[:, 1:].contiguous() # [batch_size * K, seq_len - 1]
            logprobs_new = -F.cross_entropy(
                logits_new.view(-1, logits_new.shape[-1]), labels_new.view(-1), reduction = 'none', ignore_index=-100
            ).view(logits_new.shape[0], -1) # [batch_size * K, seq_len - 1]
            logprobs_new = logprobs_new.view(batch_size, K, -1) # [batch_size, K, seq_len - 1]
            assert logprobs_ref.shape == logprobs_new.shape
            # --------------------------------------
            # Compute GRPO loss
            # --------------------------------------
            valid_mask = batch_action_mask[:, :-1].contiguous().float().view(batch_size, K, -1) # [batch_size, K, seq_len - 1] # This is because the default pytorch average is not taken over the valid tokens 
            # compute prob ratios
            ratio = torch.exp(logprobs_new - logprobs_old) # [batch_size, K, seq_len - 1]
            ratio_clipped = torch.clamp(ratio, 1.0 - ppo_clip_range, 1.0 + ppo_clip_range) # [batch_size, K, seq_len - 1]
            individual_ppo_reward = torch.min(ratio * batch_advantages, ratio_clipped * batch_advantages) # [batch_size, K, seq_len - 1]
            # compute KL penalty
            ratio_ref_log = logprobs_ref - logprobs_new # [batch_size, K, seq_len - 1]
            ratio_ref = torch.exp(ratio_ref_log) # [batch_size, K, seq_len - 1]
            individual_kl_penality = ratio_ref - ratio_ref_log - 1 # [batch_size, K, seq_len - 1]
            # compute the overall GRPO loss
            sum_loss_ave_response = (individual_ppo_reward - kl_coef * individual_kl_penality).sum(dim = -1) # [batch_size, K]
            count_ave_response = valid_mask.sum(dim = -1) # [batch_size, K]
            reward_ave_response = sum_loss_ave_response / count_ave_response # [batch_size, K]
            grpo_loss = -reward_ave_response.mean()
            # --------------------------------------
            # Record and log loss
            # --------------------------------------
            loss_ = 0.0
            loss_ += grpo_loss.detach()
            dist.all_reduce(loss_, op=dist.ReduceOp.AVG)
            if master_process :
                print(f'grpo training loss at step {step} with ppo epoch {ppo_epoch} is: {loss_:.4f}')
                with open(log_file, "a") as f:
                    f.write(f'grpo training loss at step {step} with ppo epoch {ppo_epoch} is: {loss_:.4f}\n')
            # --------------------------------------
            # Optimization Step
            # --------------------------------------
            grpo_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm = max_grad_norm)
            lr = get_lr(step, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            
        # Checkpoint
        if master_process and (step % 50 == 0 or step == max_steps - 1):
            ckpt_dir = f"{log_dir}/llama-3.2-1b-grpo-step{step+1}"
            policy_model.module.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)


dist.barrier()
dist.destroy_process_group()    


# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 grpo_train.py
        
