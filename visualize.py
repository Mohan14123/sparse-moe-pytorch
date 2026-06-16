import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

# Use a clean, publication-quality style
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'moe': '#2563EB',       # Blue
    'dense': '#F97316',     # Orange
    'expert0': '#3B82F6',   # Blue
    'expert1': '#F97316',   # Orange
    'expert2': '#10B981',   # Green
    'expert3': '#EF4444',   # Red
    'expert4': '#8B5CF6',   # Purple
    'expert5': '#EC4899',   # Pink
    'expert6': '#14B8A6',   # Teal
    'expert7': '#F59E0B',   # Amber
    'entropy': '#6366F1',   # Indigo
    'drop': '#EF4444',      # Red
    'cv': '#10B981',        # Green
    'ppl': '#8B5CF6',       # Purple
    'lr': '#F59E0B',        # Amber
}


def plot_expert_histograms(history, output_dir):
    """Plot stacked bar chart of expert usage across epochs."""
    epochs = [x["epoch"] for x in history]
    usage = [x["expert_usage"] for x in history]
    num_experts = len(usage[0])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(epochs)
    
    expert_colors = [COLORS.get(f'expert{i}', '#888888') for i in range(num_experts)]
    
    for i in range(num_experts):
        expert_data = [u[i] for u in usage]
        ax.bar(epochs, expert_data, bottom=bottom, label=f"Expert {i}", 
               color=expert_colors[i], edgecolor='white', linewidth=0.5)
        bottom = [b + d for b, d in zip(bottom, expert_data)]
        
    ax.set_title("Expert Usage per Epoch", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tokens Routed")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True, fancybox=True)
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_usage.png"))
    plt.close()

def plot_metrics(history, output_dir):
    """Plot routing entropy, dropped token rates, and load balancing CV."""
    epochs = [x["epoch"] for x in history]
    entropy = [x["entropy"] for x in history]
    drop_rate = [x["drop_rate"] for x in history]
    
    # Check if new metrics exist
    has_cv = "load_balance_cv" in history[0] and history[0]["load_balance_cv"] is not None
    has_lr = "learning_rate" in history[0]
    
    num_plots = 2 + (1 if has_cv else 0) + (1 if has_lr else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5))
    if num_plots == 1:
        axes = [axes]
    
    idx = 0
    
    # Entropy
    axes[idx].plot(epochs, entropy, marker='o', color=COLORS['entropy'], linewidth=2, markersize=5)
    axes[idx].set_title("Routing Entropy ($\\mathcal{H}$)", fontweight='bold')
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Entropy")
    axes[idx].grid(True, linestyle='--', alpha=0.3)
    # Add max entropy reference line
    if len(history) > 0 and history[0].get("expert_usage"):
        num_experts = len(history[0]["expert_usage"])
        max_entropy = math.log(num_experts)
        axes[idx].axhline(y=max_entropy, color='gray', linestyle=':', alpha=0.7, label=f'Max (log {num_experts})')
        axes[idx].legend()
    idx += 1
    
    # Drop Rate
    axes[idx].plot(epochs, drop_rate, marker='o', color=COLORS['drop'], linewidth=2, markersize=5)
    axes[idx].set_title("Token Drop Rate", fontweight='bold')
    axes[idx].set_xlabel("Epoch")
    axes[idx].set_ylabel("Fraction Dropped")
    axes[idx].grid(True, linestyle='--', alpha=0.3)
    idx += 1
    
    # CV (if available)
    if has_cv:
        cv_data = [x["load_balance_cv"] for x in history]
        axes[idx].plot(epochs, cv_data, marker='s', color=COLORS['cv'], linewidth=2, markersize=5)
        axes[idx].set_title("Load Balance CV (%)", fontweight='bold')
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("CV (%)")
        axes[idx].grid(True, linestyle='--', alpha=0.3)
        idx += 1
    
    # LR Schedule (if available)
    if has_lr:
        lr_data = [x["learning_rate"] for x in history]
        axes[idx].plot(epochs, lr_data, marker='D', color=COLORS['lr'], linewidth=2, markersize=5)
        axes[idx].set_title("Learning Rate Schedule", fontweight='bold')
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("LR")
        axes[idx].grid(True, linestyle='--', alpha=0.3)
        idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "routing_metrics.png"))
    plt.close()

def plot_ablation(moe_history, dense_history, output_dir):
    """Compare MoE Task Loss vs Dense Baseline Loss, with perplexity subplot."""
    epochs = [x["epoch"] for x in moe_history]
    
    moe_train_loss = [x["task_loss"] for x in moe_history]
    dense_train_loss = [x["task_loss"] for x in dense_history]
    
    has_val = "val_loss" in moe_history[0]
    has_ppl = "perplexity" in moe_history[0]
    
    num_plots = 1 + (1 if has_ppl else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Loss Comparison
    ax = axes[0]
    ax.plot(epochs, moe_train_loss, marker='o', linestyle='-', color=COLORS['moe'], 
            linewidth=2, markersize=5, label="MoE (Train)")
    ax.plot(epochs, dense_train_loss, marker='x', linestyle='-', color=COLORS['dense'], 
            linewidth=2, markersize=7, label="Dense (Train)")
    
    if has_val:
        moe_val_loss = [x["val_loss"] for x in moe_history]
        dense_val_loss = [x["val_loss"] for x in dense_history]
        ax.plot(epochs, moe_val_loss, marker='o', linestyle='--', color=COLORS['moe'], 
                alpha=0.6, linewidth=1.5, markersize=4, label="MoE (Val)")
        ax.plot(epochs, dense_val_loss, marker='x', linestyle='--', color=COLORS['dense'], 
                alpha=0.6, linewidth=1.5, markersize=6, label="Dense (Val)")
    
    ax.set_title("Ablation: Task Loss Comparison", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=True, fancybox=True)
    
    # Perplexity Comparison
    if has_ppl:
        ax2 = axes[1]
        moe_ppl = [x["perplexity"] for x in moe_history]
        dense_ppl = [x.get("perplexity", math.exp(x["val_loss"])) for x in dense_history]
        
        ax2.plot(epochs, moe_ppl, marker='o', linestyle='-', color=COLORS['moe'], 
                linewidth=2, markersize=5, label="MoE")
        ax2.plot(epochs, dense_ppl, marker='x', linestyle='-', color=COLORS['dense'], 
                linewidth=2, markersize=7, label="Dense")
        ax2.set_title("Validation Perplexity", fontweight='bold')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Perplexity")
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.legend(frameon=True, fancybox=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_loss.png"))
    plt.close()


def print_summary(moe_history):
    """Print a publication-quality summary table of final metrics."""
    final = moe_history[-1]
    
    print(f"\n{'='*55}")
    print(f"  FINAL TRAINING SUMMARY (Epoch {final['epoch']})")
    print(f"{'='*55}")
    print(f"  Validation Loss:      {final['val_loss']:.4f}")
    if 'perplexity' in final:
        print(f"  Perplexity:           {final['perplexity']:.2f}")
    print(f"  Task Loss:            {final['task_loss']:.4f}")
    if final.get('aux_loss'):
        print(f"  Auxiliary Loss:       {final['aux_loss']:.6f}")
    if final.get('entropy'):
        print(f"  Routing Entropy:      {final['entropy']:.4f}")
    if final.get('drop_rate') is not None:
        print(f"  Token Drop Rate:      {final['drop_rate']:.4f}")
    if final.get('load_balance_cv') is not None:
        print(f"  Load Balance CV:      {final['load_balance_cv']:.2f}%")
    if final.get('learning_rate') is not None:
        print(f"  Final Learning Rate:  {final['learning_rate']:.6f}")
    if final.get('expert_usage'):
        print(f"  Expert Usage:")
        for i, u in enumerate(final['expert_usage']):
            print(f"    Expert {i}: {u:,.0f} tokens")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    moe_path = os.path.join(args.output_dir, "moe_training_log.json")
    dense_path = os.path.join(args.output_dir, "dense_training_log.json")
    
    has_moe = os.path.exists(moe_path)
    has_dense = os.path.exists(dense_path)
    
    if has_moe:
        with open(moe_path, "r") as f:
            moe_history = json.load(f)
        plot_expert_histograms(moe_history, args.output_dir)
        plot_metrics(moe_history, args.output_dir)
        print_summary(moe_history)
        print(f"MoE metrics and histograms plotted to {args.output_dir}/")
        
    if has_moe and has_dense:
        with open(dense_path, "r") as f:
            dense_history = json.load(f)
        plot_ablation(moe_history, dense_history, args.output_dir)
        print(f"Ablation comparison plots generated to {args.output_dir}/")
    elif has_moe and not has_dense:
        print("Note: No dense_training_log.json found. Run 'python train.py --model dense' to generate ablation comparison charts.")

if __name__ == "__main__":
    main()
