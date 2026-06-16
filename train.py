import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import math
import numpy as np
from tqdm import tqdm
from decimal import Decimal

import hydra
from omegaconf import DictConfig, OmegaConf

from moe.moe_layer import MoELayer
from moe.losses import compute_auxiliary_loss, compute_routing_entropy

import pandas as pd
import tiktoken

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

class TinyStoriesDataset(torch.utils.data.Dataset):
    """
    Dataset loader for the TinyStories CSV files.
    Reads text data, tokenizes it with OpenAI's tiktoken (gpt2/r50k_base),
    and creates (input, target) pairs for next-token prediction.
    """
    def __init__(self, csv_file, seq_len=32, num_samples=None):
        print(f"Loading dataset from {csv_file}...")
        
        # We only take the 'text' column. 
        # Using a fraction of samples if num_samples is specified for speed testing.
        df = pd.read_csv(csv_file, usecols=["text"])
        if num_samples and num_samples < len(df):
            df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
            
        texts = df['text'].dropna().tolist()
        
        # Simple fast BPE tokenizer used by GPT-2
        enc = tiktoken.get_encoding("r50k_base")
        
        print(f"Tokenizing {len(texts)} stories...")
        # Since this is a demo, we tokenize everything into memory
        self.data = []
        
        # We process story by story. For better language models, we'd 
        # concatenate all stories with an <EOS> token and chunk exactly to seq_len+1.
        for text in texts:
            tokens = enc.encode(text)
            # Create chunks of seq_len + 1 (for next token prediction)
            for i in range(0, len(tokens) - seq_len, seq_len):
                chunk = tokens[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.data.append(chunk)

        # Convert to a single large flat tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.seq_len = seq_len
        self.vocab_size = enc.n_vocab
        print(f"Built dataset with {len(self.data)} sequences of length {seq_len}. Vocab size: {self.vocab_size}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # input is tokens 0 to N-1, target is tokens 1 to N
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


class BaselineModel(nn.Module):
    """
    Parameter-matching Dense Baseline Model.
    If MoE has E experts of size d_ff, this has 1 expert of size E * d_ff.
    """
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_experts: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Massive FFN to match the parameter count of all experts combined
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * num_experts),
            nn.GELU(),
            nn.Linear(d_ff * num_experts, d_model)
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        ffn_out = self.ffn(emb)
        logits = self.fc_out(ffn_out)
        return logits, {} # Empty aux dict to match MoE API

class MoEModel(nn.Module):
    """
    The MoE Model.
    """
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_experts: int, top_k: int,
                 capacity_factor: float = 1.5, noisy_routing: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            noisy_routing=noisy_routing
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        moe_out, aux_metrics = self.moe(emb)
        logits = self.fc_out(moe_out)
        return logits, aux_metrics

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressive text generation.
        Takes a conditioning sequence idx (LongTensor of shape (b,t)) and generates
        max_new_tokens new tokens by feeding predictions back into the model.
        """
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # Pluck the logits at the final step and scale by temperature
            next_token_logits = logits[:, -1, :] / temperature
            # Apply softmax to convert logits to probabilities
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def print_model_efficiency(d_model, d_ff, num_experts, top_k):
    """
    Compute and display the active parameter reduction from sparse MoE routing.
    """
    # Each expert: (d_model -> d_ff) + bias + (d_ff -> d_model) + bias
    expert_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    total_ffn_params = expert_params * num_experts
    active_ffn_params = expert_params * top_k
    
    reduction = 1 - (active_ffn_params / total_ffn_params)
    print(f"\n{'='*50}")
    print(f"  MoE Efficiency Report")
    print(f"{'='*50}")
    print(f"  Total FFN Params (all experts): {total_ffn_params:,}")
    print(f"  Active FFN Params per token:    {active_ffn_params:,}")
    print(f"  FLOPs / Param Reduction:        {reduction * 100:.1f}%")
    print(f"  Experts: {num_experts} | Top-K: {top_k}")
    print(f"{'='*50}\n")


def train(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Model configuration — all values from Hydra YAML config
    d_model = cfg.model.d_model
    d_ff = cfg.model.d_ff
    num_experts = cfg.model.num_experts
    top_k = cfg.model.top_k
    seq_len = cfg.data.seq_len
    model_type = cfg.model.type
    
    # Data Setup
    dataset_path = cfg.data.path
    # Hydra changes the working directory, so resolve relative paths
    original_cwd = hydra.utils.get_original_cwd()
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(original_cwd, dataset_path)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset block not found at {dataset_path}")
        
    num_samples = cfg.training.get("num_samples", None)
    full_dataset = TinyStoriesDataset(csv_file=dataset_path, seq_len=seq_len, num_samples=num_samples)
    vocab_size = full_dataset.vocab_size
    
    # 80/20 train/validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Init model
    if model_type == "moe":
        model = MoEModel(vocab_size, d_model, d_ff, num_experts, top_k,
                         capacity_factor=cfg.model.capacity_factor,
                         noisy_routing=cfg.model.noisy_routing).to(device)
        # Print efficiency report for MoE
        print_model_efficiency(d_model, d_ff, num_experts, top_k)
    else:
        model = BaselineModel(vocab_size, d_model, d_ff, num_experts).to(device)
   # model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Cosine Annealing LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=1e-5)
    
    # Mixed precision and gradient accumulation setups
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")
    
    # Checkpointing: track best validation loss
    best_val_loss = float('inf')
    
    # Logging — resolve output_dir relative to original cwd
    output_dir = cfg.output.dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(original_cwd, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{model_type}_training_log.json")
    history = []
    
    # Flatten the Hydra config for saving into checkpoints
    flat_cfg = {
        "model": model_type,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_experts": num_experts,
        "top_k": top_k,
        "capacity_factor": cfg.model.capacity_factor,
        "noisy_routing": cfg.model.noisy_routing,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "lr": cfg.training.lr,
        "accum_steps": cfg.training.accum_steps,
        "alpha": cfg.training.alpha,
        "beta": cfg.training.beta,
        "seq_len": seq_len,
    }
    
    # Phase 4: Weights & Biases initialization
    use_wandb = cfg.logging.wandb and HAS_WANDB
    if use_wandb:
        wandb.init(
            project="sparse-moe",
            name=f"{model_type}-E{num_experts}-K{top_k}-ep{cfg.training.epochs}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.watch(model, log="gradients", log_freq=100)
        print("W&B run initialized.")
    elif cfg.logging.wandb and not HAS_WANDB:
        print("WARNING: logging.wandb=true but wandb not installed. Install with: pip install wandb")
    
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_entropy = 0.0
        epoch_drop_rate = 0.0
        expert_usage_counts = [0] * num_experts
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")
        
        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with Mixed Precision
            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=True):
                    logits, aux_metrics = model(inputs)
                    
                    # Compute task loss
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    task_loss = criterion(logits_flat, targets_flat)
                    
                    # Compute total loss if MoE
                    total_loss = task_loss
                    aux_loss_val = 0.0
                    entropy_val = 0.0
                    drop_rate_val = 0.0
                    
                    if model_type == "moe" and aux_metrics:
                        f_i = aux_metrics["f_i"]
                        P_i = aux_metrics["P_i"]
                        routing_probs = aux_metrics["routing_probs"]
                        
                        L_aux = compute_auxiliary_loss(f_i, P_i, alpha=cfg.training.alpha)
                        H = compute_routing_entropy(routing_probs)
                        
                        # total_loss = task_loss + \alpha * L_aux - \beta * H
                        total_loss = task_loss + L_aux - (cfg.training.beta * H)
                        
                        aux_loss_val = L_aux.item()
                        entropy_val = H.item()
                        drop_rate_val = float(aux_metrics["drop_rate"])
                        
                        # Accumulate expert usage for histogram logs (fraction of batch seq len)
                        for i in range(num_experts):
                            expert_usage_counts[i] += f_i[i].item() * (inputs.size(0) * inputs.size(1))
            else:
                # CPU Fallback
                logits, aux_metrics = model(inputs)
                logits_flat = logits.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                task_loss = criterion(logits_flat, targets_flat)
                
                total_loss = task_loss
                aux_loss_val = 0.0
                entropy_val = 0.0
                drop_rate_val = 0.0
                
                if model_type == "moe" and aux_metrics:
                    f_i = aux_metrics["f_i"]
                    P_i = aux_metrics["P_i"]
                    routing_probs = aux_metrics["routing_probs"]
                    L_aux = compute_auxiliary_loss(f_i, P_i, alpha=cfg.training.alpha)
                    H = compute_routing_entropy(routing_probs)
                    total_loss = task_loss + L_aux - (cfg.training.beta * H)
                    
                    aux_loss_val = L_aux.item()
                    entropy_val = H.item()
                    drop_rate_val = float(aux_metrics["drop_rate"])
                    for i in range(num_experts):
                         expert_usage_counts[i] += f_i[i].item() * (inputs.size(0) * inputs.size(1))

            # Backward pass with accumulation
            accum_steps = cfg.training.accum_steps
            if device.type == "cuda":
                scaler.scale(total_loss / accum_steps).backward()
            else:
                (total_loss / accum_steps).backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                if device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            # Logging
            epoch_loss += total_loss.item()
            epoch_task_loss += task_loss.item()
            epoch_aux_loss += aux_loss_val
            epoch_entropy += entropy_val
            epoch_drop_rate += drop_rate_val
            
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "Drop": f"{drop_rate_val:.2f}"
            })

        # --- Validation Loop 80/20 Split ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", enabled=True):
                        logits, _ = model(inputs)
                        logits_flat = logits.view(-1, vocab_size)
                        targets_flat = targets.view(-1)
                        loss = criterion(logits_flat, targets_flat)
                else:
                    logits, _ = model(inputs)
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)
                    loss = criterion(logits_flat, targets_flat)
                    
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Phase 2: Compute perplexity from validation loss
        perplexity = math.exp(val_loss)
        
        # Phase 2: Compute Load Balancing CV (Coefficient of Variation)
        load_balance_cv = 0.0
        if model_type == "moe" and sum(expert_usage_counts) > 0:
            usage_array = np.array(expert_usage_counts)
            load_balance_cv = (np.std(usage_array) / np.mean(usage_array)) * 100
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f} | LR: {current_lr:.6f}", end="")
        if model_type == "moe":
            print(f" | CV: {load_balance_cv:.2f}%", end="")
        print()
        
        # Step the cosine LR scheduler
        scheduler.step()

        # Epoch aggregation
        num_steps = len(train_loader)
        epoch_stats = {
            "epoch": epoch + 1,
            "total_loss": epoch_loss / num_steps,
            "val_loss": val_loss,
            "perplexity": perplexity,
            "task_loss": epoch_task_loss / num_steps,
            "aux_loss": epoch_aux_loss / num_steps,
            "entropy": epoch_entropy / num_steps,
            "drop_rate": epoch_drop_rate / num_steps,
            "load_balance_cv": load_balance_cv if model_type == "moe" else None,
            "learning_rate": current_lr,
            "expert_usage": expert_usage_counts if model_type == "moe" else None
        }
        history.append(epoch_stats)
        
        # Phase 4: Log to W&B
        if use_wandb:
            wandb_log = {
                "epoch": epoch + 1,
                "train/total_loss": epoch_stats["total_loss"],
                "train/task_loss": epoch_stats["task_loss"],
                "val/loss": val_loss,
                "val/perplexity": perplexity,
                "lr": current_lr,
            }
            if model_type == "moe":
                wandb_log.update({
                    "train/aux_loss": epoch_stats["aux_loss"],
                    "routing/entropy": epoch_stats["entropy"],
                    "routing/drop_rate": epoch_stats["drop_rate"],
                    "routing/load_balance_cv": load_balance_cv,
                })
            wandb.log(wandb_log)
        
        # Phase 1: Model Checkpointing — save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, f"{model_type}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'perplexity': perplexity,
                'args': flat_cfg
            }, best_model_path)
            print(f"  ✓ Best model saved to {best_model_path} (val_loss: {val_loss:.4f})")
        
    # Phase 1: Save final model checkpoint
    final_model_path = os.path.join(output_dir, f"{model_type}_final.pt")
    torch.save({
        'epoch': cfg.training.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'perplexity': perplexity,
        'args': flat_cfg
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save log
    with open(log_file, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training completed. Log saved to {log_file}")
    
    # Phase 4: Finish W&B run
    if use_wandb:
        wandb.finish()
        print("W&B run finished.")

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()
