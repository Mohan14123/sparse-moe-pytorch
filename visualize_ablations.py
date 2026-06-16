#!/usr/bin/env python3
"""
Generate ablation comparison charts from Phase 7 experiments.
Produces publication-quality visuals demonstrating:
  1. Expert Collapse (alpha=0 vs baseline)
  2. Capacity Hard-Drops (capacity_factor=0.5 vs baseline)
  3. Noisy vs Deterministic Routing
  4. K-Scaling Law (K=1, K=2, K=4 with E=8)
"""
import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'baseline': '#2563EB',
    'ablation': '#EF4444',
    'k1': '#EF4444',
    'k2': '#2563EB',
    'k4': '#10B981',
    'no_noise': '#F97316',
}

OUTPUT_DIR = "outputs"

def load_log(subdir):
    path = os.path.join(OUTPUT_DIR, subdir, "moe_training_log.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def plot_expert_collapse():
    """Compare expert usage with and without load balancing."""
    baseline = load_log("ablation_baseline")
    collapse = load_log("ablation_collapse")
    if not baseline or not collapse:
        print("Skipping expert collapse plot: missing data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, data, title in [(axes[0], baseline, "With Load Balancing (α=0.01)"),
                             (axes[1], collapse, "Without Load Balancing (α=0.0)")]:
        final = data[-1]
        usage = final["expert_usage"]
        num_experts = len(usage)
        colors = ['#3B82F6', '#F97316', '#10B981', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F59E0B']
        
        bars = ax.bar(range(num_experts), usage, color=colors[:num_experts], 
                      edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Expert Index")
        ax.set_ylabel("Tokens Routed")
        ax.set_xticks(range(num_experts))
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add CV annotation
        cv = final.get("load_balance_cv", 0)
        ax.annotate(f"CV = {cv:.2f}%", xy=(0.95, 0.95), xycoords='axes fraction',
                   fontsize=12, fontweight='bold', ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    
    fig.suptitle("Ablation: Expert Collapse Without Auxiliary Loss", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_expert_collapse.png"))
    plt.close()
    print("✓ Expert collapse plot saved")


def plot_capacity_ablation():
    """Compare performance with normal vs aggressive capacity dropping."""
    baseline = load_log("ablation_baseline")
    capacity = load_log("ablation_capacity")
    if not baseline or not capacity:
        print("Skipping capacity ablation plot: missing data")
        return
    
    epochs_b = [x["epoch"] for x in baseline]
    epochs_c = [x["epoch"] for x in capacity]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Val Loss
    axes[0].plot(epochs_b, [x["val_loss"] for x in baseline], marker='o', color=COLORS['baseline'], 
                 linewidth=2, label="CF=1.5 (baseline)")
    axes[0].plot(epochs_c, [x["val_loss"] for x in capacity], marker='s', color=COLORS['ablation'], 
                 linewidth=2, label="CF=0.5 (aggressive)")
    axes[0].set_title("Validation Loss", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # Perplexity
    axes[1].plot(epochs_b, [x["perplexity"] for x in baseline], marker='o', color=COLORS['baseline'], 
                 linewidth=2, label="CF=1.5")
    axes[1].plot(epochs_c, [x["perplexity"] for x in capacity], marker='s', color=COLORS['ablation'], 
                 linewidth=2, label="CF=0.5")
    axes[1].set_title("Perplexity", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Drop Rate
    axes[2].plot(epochs_b, [x["drop_rate"] for x in baseline], marker='o', color=COLORS['baseline'], 
                 linewidth=2, label="CF=1.5")
    axes[2].plot(epochs_c, [x["drop_rate"] for x in capacity], marker='s', color=COLORS['ablation'], 
                 linewidth=2, label="CF=0.5")
    axes[2].set_title("Token Drop Rate", fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Drop Rate")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle("Ablation: Impact of Capacity Factor on Model Performance", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_capacity.png"))
    plt.close()
    print("✓ Capacity ablation plot saved")


def plot_noisy_vs_deterministic():
    """Compare noisy routing vs deterministic routing."""
    baseline = load_log("ablation_baseline")
    no_noise = load_log("ablation_no_noise")
    if not baseline or not no_noise:
        print("Skipping noise ablation plot: missing data")
        return
    
    epochs_b = [x["epoch"] for x in baseline]
    epochs_n = [x["epoch"] for x in no_noise]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Routing Entropy
    axes[0].plot(epochs_b, [x["entropy"] for x in baseline], marker='o', color=COLORS['baseline'], 
                 linewidth=2, label="Noisy (default)")
    axes[0].plot(epochs_n, [x["entropy"] for x in no_noise], marker='s', color=COLORS['no_noise'], 
                 linewidth=2, label="Deterministic")
    axes[0].axhline(y=math.log(4), color='gray', linestyle=':', alpha=0.5, label='Max entropy')
    axes[0].set_title("Routing Entropy", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Entropy")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # CV
    axes[1].plot(epochs_b, [x["load_balance_cv"] for x in baseline], marker='o', color=COLORS['baseline'], 
                 linewidth=2, label="Noisy")
    axes[1].plot(epochs_n, [x["load_balance_cv"] for x in no_noise], marker='s', color=COLORS['no_noise'], 
                 linewidth=2, label="Deterministic")
    axes[1].set_title("Load Balance CV (%)", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CV (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # Val Loss
    axes[2].plot(epochs_b, [x["val_loss"] for x in baseline], marker='o', color=COLORS['baseline'], 
                 linewidth=2, label="Noisy")
    axes[2].plot(epochs_n, [x["val_loss"] for x in no_noise], marker='s', color=COLORS['no_noise'], 
                 linewidth=2, label="Deterministic")
    axes[2].set_title("Validation Loss", fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle("Ablation: Noisy vs Deterministic Routing", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_noise.png"))
    plt.close()
    print("✓ Noise ablation plot saved")


def plot_k_scaling():
    """Compare K=1, K=2, K=4 on E=8 experts."""
    k1 = load_log("ablation_k1")
    k2 = load_log("ablation_k2")
    k4 = load_log("ablation_k4")
    if not k1 or not k2 or not k4:
        print("Skipping K-scaling plot: missing data")
        return
    
    epochs = [x["epoch"] for x in k1]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    for data, label, color in [(k1, "K=1 (87.5% reduction)", COLORS['k1']),
                                (k2, "K=2 (75% reduction)", COLORS['k2']),
                                (k4, "K=4 (50% reduction)", COLORS['k4'])]:
        ep = [x["epoch"] for x in data]
        
        axes[0].plot(ep, [x["val_loss"] for x in data], marker='o', color=color, linewidth=2, label=label)
        axes[1].plot(ep, [x["perplexity"] for x in data], marker='o', color=color, linewidth=2, label=label)
        axes[2].plot(ep, [x["entropy"] for x in data], marker='o', color=color, linewidth=2, label=label)
    
    axes[0].set_title("Validation Loss", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    axes[1].set_title("Perplexity", fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    axes[2].set_title("Routing Entropy", fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Entropy")
    axes[2].axhline(y=math.log(8), color='gray', linestyle=':', alpha=0.5, label='Max entropy (E=8)')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    fig.suptitle("Ablation: Scaling Law of K (E=8 Experts)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_k_scaling.png"))
    plt.close()
    print("✓ K-scaling plot saved")


if __name__ == "__main__":
    plot_expert_collapse()
    plot_capacity_ablation()
    plot_noisy_vs_deterministic()
    plot_k_scaling()
    print(f"\nAll ablation plots saved to {OUTPUT_DIR}/")
