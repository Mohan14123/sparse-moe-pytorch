import torch
import pytest
from moe.experts import Expert, ExpertGroup
from moe.router import TopKRouter
from moe.dispatcher import Dispatcher
from moe.losses import compute_routing_entropy, compute_auxiliary_loss
from moe.moe_layer import MoELayer

def test_expert():
    batch_size, d_model, d_ff = 4, 16, 32
    expert = Expert(d_model, d_ff)
    x = torch.randn(batch_size, d_model)
    out = expert(x)
    assert out.shape == (batch_size, d_model)

def test_expert_group():
    num_experts, capacity, d_model, d_ff = 3, 4, 16, 32
    group = ExpertGroup(num_experts, d_model, d_ff)
    x = torch.randn(capacity, d_model)
    out = group(x, expert_idx=1)
    assert out.shape == (capacity, d_model)

def test_router_shapes_and_capacity():
    batch_size, seq_len, d_model = 2, 10, 16
    num_experts, top_k = 4, 2
    router = TopKRouter(d_model, num_experts, top_k, capacity_factor=1.0)
    
    x = torch.randn(batch_size * seq_len, d_model)
    dispatch_mask, combine_weights, aux_metrics = router(x)
    
    assert dispatch_mask.shape == (batch_size * seq_len, num_experts)
    assert combine_weights.shape == (batch_size * seq_len, num_experts)
    
    # Check top-k sparsity (at most top_k non-zeros per token)
    assert (combine_weights > 0).sum(dim=1).max().item() <= top_k
    
    # Check capacity limit
    expected_capacity = int((batch_size * seq_len * top_k / num_experts) * 1.0)
    assert dispatch_mask.sum(dim=0).max().item() <= expected_capacity

def test_dispatcher():
    batch_seq_len, d_model, num_experts, d_ff = 8, 16, 4, 32
    x = torch.randn(batch_seq_len, d_model)
    dispatch_mask = torch.zeros(batch_seq_len, num_experts, dtype=torch.bool)
    dispatch_mask[:4, 0] = True
    dispatch_mask[4:, 1] = True
    
    combine_weights = torch.zeros(batch_seq_len, num_experts)
    combine_weights[:4, 0] = 1.0
    combine_weights[4:, 1] = 1.0
    
    experts = ExpertGroup(num_experts, d_model, d_ff)
    out = Dispatcher.dispatch_and_compute(x, dispatch_mask, combine_weights, experts)
    
    assert out.shape == (batch_seq_len, d_model)
    # Recombined output shouldn't be exactly original input or exactly zero for assigned tokens
    assert not torch.allclose(out[:4], x[:4])

def test_moe_layer():
    batch_size, seq_len, d_model, d_ff = 2, 8, 16, 32
    num_experts, top_k = 4, 2
    layer = MoELayer(d_model, d_ff, num_experts, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    
    out, aux = layer(x)
    assert out.shape == (batch_size, seq_len, d_model)
    assert "f_i" in aux and "P_i" in aux
    assert "routing_probs" in aux and "drop_rate" in aux

def test_losses_differentiable():
    batch_seq_len, num_experts = 10, 4
    logits = torch.randn(batch_seq_len, num_experts, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    
    f_i = torch.ones(num_experts) / num_experts # dummy fraction
    P_i = probs.mean(dim=0)
    
    L_aux = compute_auxiliary_loss(f_i, P_i)
    H = compute_routing_entropy(probs)
    
    loss = L_aux - 0.1 * H
    loss.backward()
    
    assert logits.grad is not None
