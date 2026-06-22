import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="outputs/moe_training_log.json")
    parser.add_argument("--output_image", type=str, default="outputs/specialization_heatmap.png")
    args = parser.parse_args()
    
    with open(args.log_file, "r") as f:
        history = json.load(f)
        
    last_epoch = history[-1]
    
    if "expert_usage_by_domain" not in last_epoch or last_epoch["expert_usage_by_domain"] is None:
        print("Error: expert_usage_by_domain not found in the log.")
        return
        
    usage_by_domain = last_epoch["expert_usage_by_domain"]
    
    # Domains mapping from prepare script
    domains = ["Story", "Math", "Code"]
    num_experts = len(usage_by_domain["0"])
    
    # Build matrix: shape (num_domains, num_experts)
    matrix = np.zeros((len(domains), num_experts))
    for d_idx in range(len(domains)):
        usage = usage_by_domain[str(d_idx)]
        matrix[d_idx] = usage
        
    # Normalize by row (domain) to show which experts handle what percentage of the domain
    # Adding a small epsilon to avoid division by zero
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = matrix / (row_sums + 1e-9)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix_normalized, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[f"Expert {i}" for i in range(num_experts)],
                yticklabels=domains)
    plt.title("Expert Specialization per Domain")
    plt.xlabel("Experts")
    plt.ylabel("Data Domains")
    
    plt.tight_layout()
    plt.savefig(args.output_image, dpi=300)
    print(f"Saved specialization heatmap to {args.output_image}")

if __name__ == "__main__":
    main()
