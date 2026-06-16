#!/usr/bin/env python3
"""
Export a trained Sparse MoE checkpoint to HuggingFace format.

Usage (local save):
    python export_hf.py --checkpoint outputs/moe_best.pt --output_dir outputs/hf_model

Usage (push to HuggingFace Hub):
    python export_hf.py --checkpoint outputs/moe_best.pt --push_to_hub --hub_repo Mohan14123/sparse-moe
"""
import argparse
import torch
import tiktoken

from moe.hf_wrapper import SparseMoEConfig, SparseMoEForCausalLM


def export(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load the PyTorch checkpoint ---
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]

    enc = tiktoken.get_encoding("r50k_base")
    vocab_size = enc.n_vocab

    # --- Create HuggingFace config from saved training args ---
    config = SparseMoEConfig(
        vocab_size=vocab_size,
        d_model=train_args.get("d_model", 256),
        d_ff=train_args.get("d_ff", 512),
        num_experts=train_args["num_experts"],
        top_k=train_args["top_k"],
        capacity_factor=train_args.get("capacity_factor", 1.5),
        noisy_routing=train_args.get("noisy_routing", True),
    )

    # --- Build the HF model and load weights ---
    model = SparseMoEForCausalLM(config)

    # Map keys from the original MoEModel state_dict to the HF wrapper
    # The HF wrapper has the same structure (embedding, moe, fc_out)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Weights loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")

    # --- Save locally ---
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    print(f"✓ Model saved to {args.output_dir}/")

    # --- Optionally push to HuggingFace Hub ---
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}...")
        model.push_to_hub(args.hub_repo, commit_message="Upload Sparse MoE model")
        config.push_to_hub(args.hub_repo, commit_message="Upload Sparse MoE config")
        print(f"✓ Pushed to https://huggingface.co/{args.hub_repo}")

    # --- Verify round-trip ---
    print("\nVerifying round-trip load...")
    loaded_model = SparseMoEForCausalLM.from_pretrained(args.output_dir)
    model.eval()
    loaded_model.eval()
    test_input = torch.randint(0, vocab_size, (1, 10))
    with torch.no_grad():
        original_out = model(input_ids=test_input).logits
        loaded_out = loaded_model(input_ids=test_input).logits
    assert torch.allclose(original_out, loaded_out, atol=1e-5), "Round-trip verification failed!"
    print("✓ Round-trip verification passed — from_pretrained() works correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Sparse MoE to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, default="outputs/moe_best.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/hf_model",
                        help="Local directory to save HF model")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the model to HuggingFace Hub")
    parser.add_argument("--hub_repo", type=str, default="Mohan14123/sparse-moe",
                        help="HuggingFace Hub repository ID")
    args = parser.parse_args()
    export(args)
