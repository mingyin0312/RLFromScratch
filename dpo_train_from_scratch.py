import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch 
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)

# 1. Load tokenizer
local_dir = "Your_model" # mine is Llama-3.2-1B
tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=True)
# 2. Load the train split of SAFE-RLHF
dataset = load_from_disk("safe_pair_data/")["train"]
print(f"There are {len(dataset)} examples.")
# -------------------------

# Initialize distributed training (NCCL backend for 8 GPUs)
dist.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
master_process = local_rank == 0

# ─── reuse eos_token as pad_token ─────────────────────────────
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# Load model
model = AutoModelForCausalLM.from_pretrained(local_dir)
model.to(local_rank)

# Load a frozen reference model for DPO regularization
ref_model = AutoModelForCausalLM.from_pretrained(local_dir)
ref_model.to(local_rank)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Wrap the training model with DistributedDataParallel
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

class PreferenceDataset(Dataset):
    """Dataset wrapper for preference data: yields (chosen_text, rejected_text) pairs."""
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["prompt"], item["chosen"], item["rejected"]

train_dataset = PreferenceDataset(dataset)

# Collate function to tokenize a batch of (chosen, rejected) pairs
def collate_fn(batch):
    input_ids, labels_list = [], []
    for prompt, chosen_resp, rejected_resp in batch:
        # Token IDs for prompt & responses
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(chosen_resp, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(rejected_resp, add_special_tokens=False)["input_ids"]
        # Build input IDs
        input_ids += [
            torch.tensor(p_ids + c_ids, dtype = torch.long),
            torch.tensor(p_ids + r_ids, dtype = torch.long)
        ]
        # Build labels: mask prompt with -100, keep response tokens
        labels_list += [
            torch.tensor([-100]*len(p_ids) + c_ids, dtype = torch.long),
            torch.tensor([-100]*len(p_ids) + r_ids, dtype = torch.long)
        ]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    labels_tensor = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    assert input_ids.shape == attention_mask.shape and attention_mask.shape == labels_tensor.shape

    return input_ids.to(local_rank), attention_mask.to(local_rank), labels_tensor.to(local_rank)

# set up learning rate
def get_lr(it, max_steps, warmup_steps = None, max_lr=2e-6, min_lr=2e-7):
    warmup_steps = int(0.01*max_steps)
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

# Set up DataLoader with DistributedSampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=train_sampler,
    collate_fn=collate_fn,
)


max_steps = len(train_loader)
if master_process:
    print(f"This epoch will run for {max_steps} steps")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
torch.set_float32_matmul_precision('high')

# Training loop for DPO fine-tuning
num_epochs = 3
beta       = 0.1

if master_process:
    log_dir = "dpo_models"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

gradient_accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)


for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    model.train()
    for step, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        is_grad_accum_step = (step + 1) % gradient_accumulation_steps == 0
        model.require_backward_grad_sync = is_grad_accum_step
        loss_ = 0.0 # used for log purpose only 

        # Forward current model
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
        logits  = outputs.logits #.float()  # [B*2, T, V]

        # **Early check #1: logits**
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(f"NaN/Inf in logits at step {step}")

        # Forward ref model (no grad)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
            ref_logits  = ref_outputs.logits #.float()

        # shift logits and labels
        logits = logits[...,:-1,:].contiguous() # [2B, T-1, V]
        ref_logits = ref_logits[...,:-1,:].contiguous() # [2B, T-1, V]
        labels = labels[...,1:].contiguous() # [2B, T-1]

        # **Early check #2: ref_logits**
        if torch.isnan(ref_logits).any() or torch.isinf(ref_logits).any():
            raise RuntimeError(f"NaN/Inf in ref_logits at step {step}")

        # Compute per-token NLLs
        V = logits.size(-1)
        loss_t  = F.cross_entropy(
            logits.view(-1, V), labels.view(-1),
            ignore_index=-100, reduction="none"
        ).view(logits.size(0), -1)
        ref_loss= F.cross_entropy(
            ref_logits.view(-1, V), labels.view(-1),
            ignore_index=-100, reduction="none"
        ).view(ref_logits.size(0), -1)

        # **Check #3: per-token losses**
        if torch.isnan(loss_t).any() or torch.isinf(loss_t).any():
            raise RuntimeError(f"NaN/Inf in loss_t at step {step}")
        if torch.isnan(ref_loss).any() or torch.isinf(ref_loss).any():
            raise RuntimeError(f"NaN/Inf in ref_loss at step {step}")

        # Sum to get sequence NLL
        nll_seq     = loss_t.sum(dim=1)
        ref_nll_seq = ref_loss.sum(dim=1)

        # **Check #4: sequence-level NLL**
        if torch.isnan(nll_seq).any() or torch.isinf(nll_seq).any():
            raise RuntimeError(f"NaN/Inf in nll_seq at step {step}")
        if torch.isnan(ref_nll_seq).any() or torch.isinf(ref_nll_seq).any():
            raise RuntimeError(f"NaN/Inf in ref_nll_seq at step {step}")

        # Split chosen/rejected
        nll_c = nll_seq[0::2]; nll_r = nll_seq[1::2]
        ref_c = ref_nll_seq[0::2]; ref_r = ref_nll_seq[1::2]

        # DPO loss
        diff_theta = nll_r - nll_c
        diff_ref   = ref_r - ref_c
        inner      = beta * (diff_theta - diff_ref)

        # **Check #5: inner**
        if torch.isnan(inner).any() or torch.isinf(inner).any():
            raise RuntimeError(f"NaN/Inf in inner at step {step}")

        dpo_loss   = -F.logsigmoid(inner).mean() 
        dpo_loss = dpo_loss / gradient_accumulation_steps
        loss_ += dpo_loss.detach() 

        # **Check #6: final loss**
        if torch.isnan(dpo_loss) or torch.isinf(dpo_loss):
            raise RuntimeError(f"NaN/Inf in dpo_loss at step {step}")
        
        dist.all_reduce(loss_, op=dist.ReduceOp.AVG)
        if master_process and step % 5 == 0:
            print(f'dpo loss at step {max_steps * epoch + step} is: {loss_*gradient_accumulation_steps:.4f}')
            with open(log_file, "a") as f:
                f.write(f"{max_steps * epoch + step} dpo train loss: {loss_*gradient_accumulation_steps:.4f}\n")
        # Backprop 
        dpo_loss.backward()

        # **Check #7: gradients**
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                raise RuntimeError(f"NaN/Inf in grad {name} at step {step}")

        if is_grad_accum_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            lr = get_lr(step, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Checkpoint
    if master_process:
        ckpt_dir = f"dpo_models/llama-3.2-1b-dpo-epoch{epoch+1}"
        model.module.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)


dist.barrier()
dist.destroy_process_group()

# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 dpo_train.py
