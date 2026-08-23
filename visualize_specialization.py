import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

TASK_NAMES = [
    'Code', 'Maths', 'Science', 'Law', 'Story',       # Text: 0-4
    'ImgClassify', 'ImgReconstruct', 'ImgEdge',        # Vision: 5-7
    'ImgCaption', 'VQA'                                 # Multi-modal: 8-9
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="outputs/moe_training_log.json")
    parser.add_argument("--output_image", type=str, default="outputs/specialization_heatmap.png")
    args = parser.parse_args()
    
    with open(args.log_file, "r") as f:
        history = json.load(f)
        
    last_epoch = history[-1]
    
    if "expert_usage_by_task" not in last_epoch or last_epoch["expert_usage_by_task"] is None:
        print("Error: expert_usage_by_task not found in the log.")
        return
        
    usage_by_task = last_epoch["expert_usage_by_task"]
    
    # Determine number of experts from the data
    first_key = list(usage_by_task.keys())[0]
    num_experts = len(usage_by_task[first_key])
    num_tasks = len(usage_by_task)
    
    # Use task names up to the number of tasks found
    task_labels = TASK_NAMES[:num_tasks] if num_tasks <= len(TASK_NAMES) else [f"Task {i}" for i in range(num_tasks)]
    
    # Build matrix: shape (num_tasks, num_experts)
    matrix = np.zeros((num_tasks, num_experts))
    for t_idx in range(num_tasks):
        usage = usage_by_task[str(t_idx)]
        matrix[t_idx] = usage
        
    # Normalize by row (task) to show which experts handle what percentage of each task
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = matrix / (row_sums + 1e-9)
    
    # --- Plot 1: Main Specialization Heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 1]})
    
    sns.heatmap(matrix_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[f"Expert {i}" for i in range(num_experts)],
                yticklabels=task_labels,
                ax=axes[0],
                vmin=0, vmax=0.5,
                linewidths=0.5, linecolor='white')
    axes[0].set_title("Expert Specialization per Task (Normalized by Task)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Experts", fontsize=12)
    axes[0].set_ylabel("Tasks", fontsize=12)
    
    # Add modality group labels
    modality_colors = {'Text': '#2196F3', 'Vision': '#4CAF50', 'Multi-Modal': '#FF9800'}
    for i, label in enumerate(task_labels):
        if i < 5:
            color = modality_colors['Text']
        elif i < 8:
            color = modality_colors['Vision']
        else:
            color = modality_colors['Multi-Modal']
        axes[0].get_yticklabels()[i].set_color(color)
        axes[0].get_yticklabels()[i].set_fontweight('bold')
    
    # --- Plot 2: Expert Load Distribution Bar Chart ---
    expert_total_load = matrix.sum(axis=0)
    expert_load_pct = expert_total_load / expert_total_load.sum() * 100
    
    bars = axes[1].barh([f"Expert {i}" for i in range(num_experts)], expert_load_pct,
                         color=sns.color_palette("YlGnBu", num_experts))
    axes[1].set_xlabel("% of Total Token Load", fontsize=12)
    axes[1].set_title("Expert Load Distribution", fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, max(expert_load_pct) * 1.3)
    
    for bar, pct in zip(bars, expert_load_pct):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300, bbox_inches='tight')
    print(f"Saved specialization heatmap to {args.output_image}")
    
    # --- Plot 3: Per-Modality Breakdown ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    modality_groups = [
        ("Text Tasks", list(range(5)), task_labels[:5]),
        ("Vision Tasks", list(range(5, 8)), task_labels[5:8]),
        ("Multi-Modal Tasks", list(range(8, 10)), task_labels[8:10]),
    ]
    
    for ax, (title, indices, labels) in zip(axes2, modality_groups):
        sub_matrix = matrix_normalized[indices]
        sns.heatmap(sub_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                    xticklabels=[f"E{i}" for i in range(num_experts)],
                    yticklabels=labels,
                    ax=ax, vmin=0, vmax=0.5,
                    linewidths=0.5, linecolor='white')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Experts")
    
    plt.tight_layout()
    modality_path = args.output_image.replace('.png', '_by_modality.png')
    plt.savefig(modality_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-modality heatmap to {modality_path}")

if __name__ == "__main__":
    main()
