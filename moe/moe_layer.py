import torch
import torch.nn as nn
from .experts import ExpertGroup
from .router import TopKRouter
from .dispatcher import Dispatcher

class MoELayer(nn.Module):
    """
    A full Mixture-of-Experts layer comprising the routing mechanism,
    the expert networks, and the dispatch/recombination logic.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int, 
                 capacity_factor: float = 1.0, noisy_routing: bool = True):
        super().__init__()
        self.d_model = d_model
        
        self.router = TopKRouter(
            d_model=d_model, 
            num_experts=num_experts, 
            top_k=top_k, 
            capacity_factor=capacity_factor,
            noisy_routing=noisy_routing
        )
        
        self.experts = ExpertGroup(
            num_experts=num_experts, 
            d_model=d_model, 
            d_ff=d_ff
        )
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            output: Recombined tensor of shape [batch_size, seq_len, d_model]
            aux_metrics: Dictionary containing metrics for auxiliary losses
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten sequence and batch dimensions for routing and expert computation
        x_flat = x.view(-1, d_model)
        
        # 1. Route tokens to determine assignments and routing probabilities
        dispatch_mask, combine_weights, aux_metrics = self.router(x_flat)
        
        # 2. Dispatch tokens to assigned experts and compute outputs
        output_flat = Dispatcher.dispatch_and_compute(
            x=x_flat,
            dispatch_mask=dispatch_mask,
            combine_weights=combine_weights,
            expert_group=self.experts
        )
        
        # 3. Reshape back to original dimensions
        output = output_flat.view(batch_size, seq_len, d_model)
        
        return output, aux_metrics
