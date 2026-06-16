import torch

def compute_routing_entropy(routing_probs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes the entropy of the routing probabilities.
    Minimizing this value (by subtracting it from the loss) encourages more confident routing.
    
    Args:
        routing_probs: Routing probabilities of shape [B*S, num_experts]
        epsilon: Small value to prevent log(0)
        
    Returns:
        Scalar entropy value
    """
    # \mathcal{H} = -\frac{1}{T}\sum_{t=1}^T \sum_{i=1}^E p_{t,i} \log(p_{t,i} + \epsilon)
    entropy = -torch.sum(routing_probs * torch.log(routing_probs + epsilon), dim=-1)
    return entropy.mean()

def compute_auxiliary_loss(f_i: torch.Tensor, P_i: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Computes the load-balancing auxiliary loss to prevent expert collapse.
    Based on the Switch Transformer formulation.
    
    Args:
        f_i: Fraction of tokens routed to each expert [num_experts]
        P_i: Mean routing probability for each expert [num_experts]
        alpha: Scaling factor for the loss
        
    Returns:
        Scalar loss value
    """
    # \mathcal{L}_{aux} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i
    num_experts = f_i.size(0)
    loss = alpha * num_experts * torch.sum(f_i * P_i)
    return loss
