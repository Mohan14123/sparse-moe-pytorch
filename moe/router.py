import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    """
    Top-k routing mechanism with optional noisy routing and capacity constraints.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int, 
                 capacity_factor: float = 1.0, noisy_routing: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.noisy_routing = noisy_routing
        
        # Core gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Optional noise network for exploration (Shazeer et al.)
        if noisy_routing:
            self.noise_linear = nn.Linear(d_model, num_experts, bias=False)
        else:
            self.noise_linear = None
            
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Token embeddings, flattened to [B*S, d_model]
        Returns:
            dispatch_mask: [B*S, num_experts] boolean mask indicating where tokens go
            combine_weights: [B*S, num_experts] routing probabilities for selected experts
            aux_metrics: Dictionary for computing load-balancing and entropy penalties
        """
        batch_seq_len = x.size(0)
        
        # Base routing logits
        logits = self.gate(x)
        
        # Add noise during training for exploration
        if self.noisy_routing and self.training:
            noise_logits = self.noise_linear(x)
            # Parametrized standard deviation via softplus
            noise_std = F.softplus(noise_logits)
            # Sample noise from standard normal, scale by learned std
            noise = torch.randn_like(logits) * noise_std
            logits = logits + noise
            
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=1)
        
        # Softmax over only the top-k selected experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # Full routing probabilities (used for losses, not for routing)
        routing_probs = F.softmax(logits, dim=-1)
        
        # Create combine weights (scatter top-k weights to full E-dimensional space)
        # Using logits.dtype ensures compatibility with AMP (e.g. float16/bfloat16 vs float32)
        combine_weights = torch.zeros_like(logits, dtype=logits.dtype)
        combine_weights.scatter_(1, top_k_indices, top_k_weights.to(logits.dtype))
        
        # Initial dispatch mask based on non-zero combine weights
        dispatch_mask = combine_weights > 0.0
        
        # Apply capacity constraints (hard dropping)
        expert_capacity = int((batch_seq_len * self.top_k / self.num_experts) * self.capacity_factor)
        dropped_tokens = 0
        
        if self.capacity_factor > 0:
            final_dispatch_mask = torch.zeros_like(dispatch_mask)
            
            for i in range(self.num_experts):
                # Find all tokens routed to expert i
                expert_indices = dispatch_mask[:, i].nonzero(as_tuple=True)[0]
                
                # If capacity exceeded, truncate (drop remaining tokens)
                if len(expert_indices) > expert_capacity:
                    # Tokens are usually dropped uniformly or based on routing prob.
                    # Here we simply truncate up to the capacity.
                    dropped_tokens += (len(expert_indices) - expert_capacity)
                    expert_indices = expert_indices[:expert_capacity]
                    
                final_dispatch_mask[expert_indices, i] = True
            
            # Zero out combine weights for dropped tokens
            combine_weights = combine_weights * final_dispatch_mask
            dispatch_mask = final_dispatch_mask
            
        # Metrics for auxiliary loss
        drop_rate = dropped_tokens / (batch_seq_len * self.top_k) if batch_seq_len > 0 else 0.0
        
        # Fraction of tokens dispatched to each expert (for load balancing)
        f_i = dispatch_mask.float().sum(dim=0) / batch_seq_len
        
        # Mean probability assigned to each expert (for load balancing)
        P_i = routing_probs.mean(dim=0)
        
        aux_metrics = {
            "f_i": f_i,
            "P_i": P_i,
            "routing_probs": routing_probs,
            "drop_rate": drop_rate
        }
        
        return dispatch_mask, combine_weights, aux_metrics
