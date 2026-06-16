import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    A standard 2-layer Feed Forward Network (FFN) with GELU activation.
    This acts as a single expert in the MoE layer.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single expert.
        Args:
            x: Input tensor of shape [batch_size, d_model]
        Returns:
            Output tensor of shape [batch_size, d_model]
        """
        return self.w2(F.gelu(self.w1(x)))

class ExpertGroup(nn.Module):
    """
    A collection of experts managed as a single PyTorch module.
    """
    def __init__(self, num_experts: int, d_model: int, d_ff: int):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Forward pass for a specific expert.
        Args:
            x: Input tokens assigned to the expert [capacity, d_model]
            expert_idx: Index of the expert to route to
        Returns:
            Computed output tokens [capacity, d_model]
        """
        return self.experts[expert_idx](x)
