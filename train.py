import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import math
import numpy as np
import contextlib
from tqdm import tqdm
from decimal import Decimal

import hydra
from omegaconf import DictConfig, OmegaConf

from moe.moe_layer import MoELayer
from moe.losses import compute_auxiliary_loss, compute_routing_entropy
from moe.multimodal import MultiModalMoEModel

import tiktoken

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ============================================================
# Task Constants
# ============================================================
TASK_NAMES = [
    'Code', 'Maths', 'Science', 'Law', 'Story',       # Text: 0-4
    'ImgClassify', 'ImgReconstruct', 'ImgEdge',        # Vision: 5-7
    'ImgCaption', 'VQA',                                # Multi-modal: 8-9
]
NUM_TASKS = 10
TEXT_TASKS = {0, 1, 2, 3, 4}
IMAGE_TASKS = {5, 6, 7}
MULTIMODAL_TASKS = {8, 9}


# ============================================================
# Multi-Modal Multi-Task Dataset
# ============================================================
class MultiModalMultiTaskDataset(torch.utils.data.Dataset):
    """
    Dataset for the Multi-Modal Multi-Task MoE.
    
    Loads pre-generated data from an .npz file produced by
    scripts/prepare_multimodal.py. Each sample contains:
      - task_id (0-9)
      - text tokens (for text and multi-modal tasks)
      - image pixels (for vision and multi-modal tasks)
      - task-specific targets (class labels, noisy images, edges, VQA answers)
    """
    def __init__(self, npz_file, seq_len=128, num_samples=None):
        print(f"Loading multi-modal dataset from {npz_file}...")
        
        data = np.load(npz_file, allow_pickle=True)
        
        texts = data['texts']              # object array of strings
        task_ids = data['task_ids']         # int array [N]
        images = data['images']            # float32 [N, 3, 32, 32]
        noisy_images = data['noisy_images']  # float32 [N, 3, 32, 32]
        edge_maps = data['edge_maps']      # float32 [N, 1, 32, 32]
        class_labels = data['class_labels']  # int [N]
        vqa_answers = data['vqa_answers']  # int [N]
        
        if num_samples and num_samples < len(task_ids):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(task_ids), size=num_samples, replace=False)
            texts = texts[indices]
            task_ids = task_ids[indices]
            images = images[indices]
            noisy_images = noisy_images[indices]
            edge_maps = edge_maps[indices]
            class_labels = class_labels[indices]
            vqa_answers = vqa_answers[indices]
        
        # Tokenize all text
        enc = tiktoken.get_encoding("r50k_base")
        self.vocab_size = enc.n_vocab
        self.seq_len = seq_len
        
        # Build samples list
        self.samples = []
        
        print(f"Tokenizing text for {len(task_ids)} raw samples...")
        for i in range(len(task_ids)):
            tid = int(task_ids[i])
            text = str(texts[i])
            image = images[i]         # [3, 32, 32]
            noisy_img = noisy_images[i]
            edge_map = edge_maps[i]   # [1, 32, 32]
            cls_label = int(class_labels[i])
            vqa_ans = int(vqa_answers[i])
            
            if tid in TEXT_TASKS:
                # Text-only: create chunked (input, target) pairs
                tokens = enc.encode(text)
                for j in range(0, len(tokens) - seq_len, seq_len):
                    chunk = tokens[j:j + seq_len + 1]
                    if len(chunk) == seq_len + 1:
                        self.samples.append({
                            'task_id': tid,
                            'text_input': torch.tensor(chunk[:-1], dtype=torch.long),
                            'text_target': torch.tensor(chunk[1:], dtype=torch.long),
                            'image': torch.zeros(3, 32, 32, dtype=torch.float32),
                            'noisy_image': torch.zeros(3, 32, 32, dtype=torch.float32),
                            'edge_map': torch.zeros(1, 32, 32, dtype=torch.float32),
                            'class_label': 0,
                            'vqa_answer': 0,
                        })
                        
            elif tid in IMAGE_TASKS:
                # Image-only: one sample per image
                self.samples.append({
                    'task_id': tid,
                    'text_input': torch.zeros(seq_len, dtype=torch.long),
                    'text_target': torch.zeros(seq_len, dtype=torch.long),
                    'image': torch.tensor(image, dtype=torch.float32),
                    'noisy_image': torch.tensor(noisy_img, dtype=torch.float32),
                    'edge_map': torch.tensor(edge_map, dtype=torch.float32),
                    'class_label': cls_label,
                    'vqa_answer': 0,
                })
                
            elif tid in MULTIMODAL_TASKS:
                # Multi-modal: image + text
                tokens = enc.encode(text)
                # Truncate or pad text to seq_len
                if len(tokens) > seq_len:
                    tokens = tokens[:seq_len + 1]
                else:
                    tokens = tokens + [0] * (seq_len + 1 - len(tokens))
                
                self.samples.append({
                    'task_id': tid,
                    'text_input': torch.tensor(tokens[:seq_len], dtype=torch.long),
                    'text_target': torch.tensor(tokens[1:seq_len + 1], dtype=torch.long),
                    'image': torch.tensor(image, dtype=torch.float32),
                    'noisy_image': torch.tensor(noisy_img, dtype=torch.float32),
                    'edge_map': torch.tensor(edge_map, dtype=torch.float32),
                    'class_label': cls_label,
                    'vqa_answer': vqa_ans,
                })
        
        # Print statistics
        task_counts = {}
        for s in self.samples:
            tid = s['task_id']
            task_counts[tid] = task_counts.get(tid, 0) + 1
        
        print(f"\nBuilt dataset with {len(self.samples)} total samples (seq_len={seq_len}):")
        for tid in sorted(task_counts.keys()):
            print(f"  Task {tid} ({TASK_NAMES[tid]}): {task_counts[tid]} samples")
        print(f"Vocab size: {self.vocab_size}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['task_id'],
            s['text_input'],
            s['text_target'],
            s['image'],
            s['noisy_image'],
            s['edge_map'],
            s['class_label'],
            s['vqa_answer'],
        )


# ============================================================
# Legacy Models (kept for backward compatibility)
# ============================================================
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
    The original text-only MoE Model (kept for backward compatibility).
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


# ============================================================
# Training Function
# ============================================================
def train(cfg: DictConfig):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Training on device: {device}")
    
    # Model configuration — all values from Hydra YAML config
    d_model = cfg.model.d_model
    d_ff = cfg.model.d_ff
    num_experts = cfg.model.num_experts
    top_k = cfg.model.top_k
    seq_len = cfg.data.seq_len
    model_type = cfg.model.type
    num_tasks = cfg.data.get("num_tasks", NUM_TASKS)
    
    # Data Setup
    dataset_path = cfg.data.path
    # Hydra changes the working directory, so resolve relative paths
    original_cwd = hydra.utils.get_original_cwd()
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(original_cwd, dataset_path)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    num_samples = cfg.training.get("num_samples", None)
    full_dataset = MultiModalMultiTaskDataset(
        npz_file=dataset_path, seq_len=seq_len, num_samples=num_samples
    )
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
        num_shape_classes = cfg.model.get("num_shape_classes", 5)
        patch_size = cfg.model.get("patch_size", 8)
        
        model = MultiModalMoEModel(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            num_shape_classes=num_shape_classes,
            patch_size=patch_size,
            capacity_factor=cfg.model.capacity_factor,
            noisy_routing=cfg.model.noisy_routing,
        ).to(device)
        # Print efficiency report for MoE
        print_model_efficiency(d_model, d_ff, num_experts, top_k)
    else:
        model = BaselineModel(vocab_size, d_model, d_ff, num_experts).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=0.01)
    
    # Loss functions for different tasks
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    
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
        "num_tasks": num_tasks,
    }
    
    # Phase 4: Weights & Biases initialization
    use_wandb = cfg.logging.wandb and HAS_WANDB
    if use_wandb:
        wandb.init(
            project="sparse-moe",
            name=f"{model_type}-E{num_experts}-K{top_k}-T{num_tasks}-ep{cfg.training.epochs}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.watch(model, log="gradients", log_freq=100)
        print("W&B run initialized.")
    elif cfg.logging.wandb and not HAS_WANDB:
        print("WARNING: logging.wandb=true but wandb not installed. Install with: pip install wandb")
    
    # ====================================================================
    # Training Loop
    # ====================================================================
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_entropy = 0.0
        epoch_drop_rate = 0.0
        expert_usage_counts = [0] * num_experts
        expert_usage_by_task = {t: [0] * num_experts for t in range(num_tasks)}
        
        # Per-task loss tracking
        task_loss_accum = {t: 0.0 for t in range(num_tasks)}
        task_loss_count = {t: 0 for t in range(num_tasks)}
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs} [Train]")
        
        for step, batch in enumerate(pbar):
            task_ids, text_input, text_target, images, noisy_images, edge_maps, class_labels, vqa_answers = batch
            
            task_ids = task_ids.to(device)
            text_input = text_input.to(device)
            text_target = text_target.to(device)
            images = images.to(device)
            noisy_images = noisy_images.to(device)
            edge_maps = edge_maps.to(device)
            class_labels = class_labels.to(device)
            vqa_answers = vqa_answers.to(device)
            
            # Forward pass
            amp_enabled = device.type == "cuda"
            amp_ctx = torch.amp.autocast("cuda", enabled=True) if amp_enabled else contextlib.nullcontext()
            with amp_ctx:
                results, aux_metrics = model(task_ids, text_input=text_input, images=images)
                
                # ---- Compute task-specific losses ----
                task_loss = torch.tensor(0.0, device=device)
                loss_count = 0
                
                # Text LM loss (tasks 0-4, and caption task 8)
                if 'text_logits' in results:
                    text_logits = results['text_logits']
                    # Figure out which samples contributed to text_logits
                    text_mask = torch.zeros(task_ids.shape[0], dtype=torch.bool, device=device)
                    for t in TEXT_TASKS:
                        text_mask |= (task_ids == t)
                    caption_mask = (task_ids == 8)
                    combined_text_mask = text_mask | caption_mask
                    
                    if combined_text_mask.any():
                        relevant_targets = text_target[combined_text_mask]
                        # Match the number of samples
                        n_text = text_logits.shape[0]
                        relevant_targets = relevant_targets[:n_text]
                        lm_loss = ce_criterion(
                            text_logits.reshape(-1, vocab_size),
                            relevant_targets.reshape(-1)
                        )
                        task_loss = task_loss + lm_loss
                        loss_count += 1
                        
                        # Track per-task losses
                        for t in range(5):
                            t_mask = (task_ids == t)
                            if t_mask.any():
                                task_loss_accum[t] += lm_loss.item()
                                task_loss_count[t] += 1
                        if caption_mask.any():
                            task_loss_accum[8] += lm_loss.item()
                            task_loss_count[8] += 1
                
                # Classification loss (task 5)
                if 'class_logits' in results:
                    cls_mask = (task_ids == 5)
                    if cls_mask.any():
                        cls_targets = class_labels[cls_mask]
                        cls_loss = ce_criterion(results['class_logits'], cls_targets.long())
                        task_loss = task_loss + cls_loss
                        loss_count += 1
                        task_loss_accum[5] += cls_loss.item()
                        task_loss_count[5] += 1
                
                # Reconstruction loss (task 6)
                if 'recon' in results:
                    rec_mask = (task_ids == 6)
                    if rec_mask.any():
                        rec_targets = images[rec_mask]  # original clean images
                        rec_loss = mse_criterion(results['recon'], rec_targets)
                        task_loss = task_loss + rec_loss
                        loss_count += 1
                        task_loss_accum[6] += rec_loss.item()
                        task_loss_count[6] += 1
                
                # Edge detection loss (task 7)
                if 'edges' in results:
                    edge_mask = (task_ids == 7)
                    if edge_mask.any():
                        edge_targets = edge_maps[edge_mask]
                        edge_loss = bce_criterion(results['edges'], edge_targets)
                        task_loss = task_loss + edge_loss
                        loss_count += 1
                        task_loss_accum[7] += edge_loss.item()
                        task_loss_count[7] += 1
                
                # VQA loss (task 9)
                if 'vqa_logits' in results:
                    vqa_mask = (task_ids == 9)
                    if vqa_mask.any():
                        vqa_targets = vqa_answers[vqa_mask]
                        vqa_loss = ce_criterion(results['vqa_logits'], vqa_targets.long())
                        task_loss = task_loss + vqa_loss
                        loss_count += 1
                        task_loss_accum[9] += vqa_loss.item()
                        task_loss_count[9] += 1
                
                # Average across contributing losses
                if loss_count > 0:
                    task_loss = task_loss / loss_count
                
                # Compute total loss with MoE auxiliary losses
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
                    
                    # Accumulate expert usage
                    for i in range(num_experts):
                        expert_usage_counts[i] += f_i[i].item()
                    
                    # Track expert usage by task
                    if "dispatch_mask" in aux_metrics:
                        dispatch_mask = aux_metrics["dispatch_mask"]  # [total_tokens, E]
                        
                        # Build a per-token task ID mapping
                        # For text tasks: each sample has seq_len tokens
                        # For image tasks: each sample has 16 patches
                        # For multi-modal tasks: each sample has 16 + seq_len tokens
                        # The model concatenates groups in order: text, image, mm
                        # and pads shorter seqs to max_seq
                        # We need to figure out the same grouping
                        
                        token_task_ids = []
                        batch_size = task_ids.shape[0]
                        
                        # Replicate model's grouping logic
                        text_idx = []
                        img_idx = []
                        mm_idx = []
                        for b_i in range(batch_size):
                            t = task_ids[b_i].item()
                            if t in TEXT_TASKS:
                                text_idx.append(b_i)
                            elif t in IMAGE_TASKS:
                                img_idx.append(b_i)
                            elif t in MULTIMODAL_TASKS:
                                mm_idx.append(b_i)
                        
                        # Compute group sequence lengths
                        text_seq = seq_len if text_idx else 0
                        img_seq = 16 if img_idx else 0  # patch_encoder.num_patches
                        mm_seq = (16 + seq_len) if mm_idx else 0
                        max_seq = max(text_seq, img_seq, mm_seq) if (text_idx or img_idx or mm_idx) else 0
                        
                        # Build token-level task ID array (matching model's batch order)
                        for b_i in text_idx:
                            t = task_ids[b_i].item()
                            token_task_ids.extend([t] * max_seq)
                        for b_i in img_idx:
                            t = task_ids[b_i].item()
                            token_task_ids.extend([t] * max_seq)
                        for b_i in mm_idx:
                            t = task_ids[b_i].item()
                            token_task_ids.extend([t] * max_seq)
                        
                        if len(token_task_ids) > 0:
                            token_task_tensor = torch.tensor(token_task_ids, device=device)
                            # Ensure sizes match (dispatch_mask may differ due to padding)
                            min_len = min(len(token_task_tensor), dispatch_mask.shape[0])
                            token_task_tensor = token_task_tensor[:min_len]
                            dm = dispatch_mask[:min_len]
                            
                            for t in range(num_tasks):
                                t_mask = (token_task_tensor == t)
                                if t_mask.any():
                                    t_usage = dm[t_mask].float().sum(dim=0)
                                    for e in range(num_experts):
                                        expert_usage_by_task[t][e] += t_usage[e].item()
            
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
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                task_ids_v, text_input_v, text_target_v, images_v, noisy_images_v, edge_maps_v, class_labels_v, vqa_answers_v = batch
                
                task_ids_v = task_ids_v.to(device)
                text_input_v = text_input_v.to(device)
                text_target_v = text_target_v.to(device)
                images_v = images_v.to(device)
                edge_maps_v = edge_maps_v.to(device)
                class_labels_v = class_labels_v.to(device)
                vqa_answers_v = vqa_answers_v.to(device)
                
                results_v, _ = model(task_ids_v, text_input=text_input_v, images=images_v)
                
                batch_loss = torch.tensor(0.0, device=device)
                batch_loss_count = 0
                
                if 'text_logits' in results_v:
                    text_mask_v = torch.zeros(task_ids_v.shape[0], dtype=torch.bool, device=device)
                    for t in TEXT_TASKS:
                        text_mask_v |= (task_ids_v == t)
                    caption_mask_v = (task_ids_v == 8)
                    combined_v = text_mask_v | caption_mask_v
                    if combined_v.any():
                        rel_targets = text_target_v[combined_v][:results_v['text_logits'].shape[0]]
                        batch_loss += ce_criterion(
                            results_v['text_logits'].reshape(-1, vocab_size),
                            rel_targets.reshape(-1)
                        )
                        batch_loss_count += 1
                
                if 'class_logits' in results_v:
                    cls_mask_v = (task_ids_v == 5)
                    if cls_mask_v.any():
                        batch_loss += ce_criterion(results_v['class_logits'], class_labels_v[cls_mask_v].long())
                        batch_loss_count += 1
                
                if 'recon' in results_v:
                    rec_mask_v = (task_ids_v == 6)
                    if rec_mask_v.any():
                        batch_loss += mse_criterion(results_v['recon'], images_v[rec_mask_v])
                        batch_loss_count += 1
                
                if 'edges' in results_v:
                    edge_mask_v = (task_ids_v == 7)
                    if edge_mask_v.any():
                        batch_loss += bce_criterion(results_v['edges'], edge_maps_v[edge_mask_v])
                        batch_loss_count += 1
                
                if 'vqa_logits' in results_v:
                    vqa_mask_v = (task_ids_v == 9)
                    if vqa_mask_v.any():
                        batch_loss += ce_criterion(results_v['vqa_logits'], vqa_answers_v[vqa_mask_v].long())
                        batch_loss_count += 1
                
                if batch_loss_count > 0:
                    val_loss += (batch_loss / batch_loss_count).item()
                    val_steps += 1
                    
        val_loss = val_loss / max(val_steps, 1)
        
        # Compute perplexity from validation loss (only meaningful for text tasks)
        perplexity = math.exp(min(val_loss, 20))  # Clamp to prevent overflow
        
        # Compute Load Balancing CV (Coefficient of Variation)
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
        
        # Print per-task losses
        print("  Per-task losses: ", end="")
        for t in range(num_tasks):
            if task_loss_count[t] > 0:
                avg = task_loss_accum[t] / task_loss_count[t]
                print(f"{TASK_NAMES[t]}={avg:.3f} ", end="")
        print()
        
        # Step the cosine LR scheduler
        scheduler.step()

        # Epoch aggregation
        num_steps = len(train_loader)
        
        per_task_losses = {}
        for t in range(num_tasks):
            if task_loss_count[t] > 0:
                per_task_losses[TASK_NAMES[t]] = task_loss_accum[t] / task_loss_count[t]
        
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
            "expert_usage": expert_usage_counts if model_type == "moe" else None,
            "expert_usage_by_task": expert_usage_by_task if model_type == "moe" else None,
            "per_task_losses": per_task_losses,
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
            for t_name, t_loss in per_task_losses.items():
                wandb_log[f"task/{t_name}"] = t_loss
            wandb.log(wandb_log)
        
        # Model Checkpointing — save best model
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
        
    # Save final model checkpoint
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
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()
        print("W&B run finished.")

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()
