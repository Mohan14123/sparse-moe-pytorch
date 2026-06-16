import torch
import torch.nn as nn

class Dispatcher:
    """
    Handles routing tokens to experts and recombining their outputs.
    Implemented as a stateless class for pedagogical clarity.
    """
    @staticmethod
    def dispatch_and_compute(x: torch.Tensor, 
                             dispatch_mask: torch.Tensor, 
                             combine_weights: torch.Tensor, 
                             expert_group: nn.Module) -> torch.Tensor:
        """
        Args:
            x: Input token embeddings [B*S, d_model]
            dispatch_mask: Boolean mask indicating token assignments [B*S, num_experts]
            combine_weights: Routing probabilities for scaling [B*S, num_experts]
            expert_group: The ExpertGroup module containing the actual FFNs
            
        Returns:
            Recombined output embeddings of shape [B*S, d_model]
        """
        _, num_experts = dispatch_mask.shape
        
        # Recombined output tensor
        output = torch.zeros_like(x)
        
        for i in range(num_experts):
            # 1. Gather: Find indices of tokens assigned to this expert
            expert_indices = dispatch_mask[:, i].nonzero(as_tuple=True)[0]
            
            if len(expert_indices) > 0:
                # Get the actual token embeddings
                expert_inputs = x[expert_indices]
                
                # 2. Compute: Pass tokens through the specific expert
                expert_outputs = expert_group(expert_inputs, i)
                
                # 3. Scale: Multiply by routing probabilities
                expert_weights = combine_weights[expert_indices, i].unsqueeze(1)
                weighted_outputs = expert_outputs * expert_weights
                
                # 4. Scatter: Add back to the main output tensor
                output[expert_indices] += weighted_outputs
                
        return output
