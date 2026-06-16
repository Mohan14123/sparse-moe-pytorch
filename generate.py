#!/usr/bin/env python3
"""
Standalone text generation script for the Sparse MoE model.

Usage (Hydra config):
    python generate.py
    python generate.py checkpoint=outputs/moe_final.pt prompt="The little cat"
    python generate.py max_tokens=100 temperature=0.5
"""
import torch
import tiktoken

import hydra
from omegaconf import DictConfig
import os

from train import MoEModel


def generate(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}\n")
    
    # Resolve checkpoint path relative to original cwd (Hydra changes cwd)
    original_cwd = hydra.utils.get_original_cwd()
    checkpoint_path = cfg.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(original_cwd, checkpoint_path)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Retrieve the training args to reconstruct the model
    train_args = checkpoint['args']
    
    # Reconstruct the model with the same architecture
    enc = tiktoken.get_encoding("r50k_base")
    vocab_size = enc.n_vocab
    
    model = MoEModel(
        vocab_size=vocab_size,
        d_model=train_args.get('d_model', 256),  # fallback to defaults if not saved
        d_ff=train_args.get('d_ff', 512),
        num_experts=train_args['num_experts'],
        top_k=train_args['top_k'],
        capacity_factor=train_args.get('capacity_factor', 1.5),
        noisy_routing=train_args.get('noisy_routing', True)
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}, PPL: {checkpoint['perplexity']:.2f})")
    print(f"{'='*50}")
    
    # Tokenize the prompt
    prompt_tokens = enc.encode(cfg.prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"\nPrompt: \"{cfg.prompt}\"")
    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Generating {cfg.max_tokens} tokens (temperature={cfg.temperature})...\n")
    print("-" * 50)
    
    # Generate
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=cfg.max_tokens, 
        temperature=cfg.temperature
    )
    
    # Decode the full output (prompt + generated)
    generated_tokens = output_ids[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    # Decode only the new tokens
    new_tokens = generated_tokens[len(prompt_tokens):]
    new_text = enc.decode(new_tokens)
    
    print(f"{generated_text}")
    print("-" * 50)
    print(f"\n[Generated {len(new_tokens)} new tokens]")


@hydra.main(config_path="configs", config_name="generate", version_base=None)
def main(cfg: DictConfig):
    generate(cfg)


if __name__ == "__main__":
    main()
