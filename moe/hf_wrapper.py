"""
HuggingFace-compatible wrapper for the Sparse MoE model.

Enables loading and saving via the standard HuggingFace API:
    from moe.hf_wrapper import SparseMoEConfig, SparseMoEForCausalLM

    model = SparseMoEForCausalLM.from_pretrained("./outputs/hf_model")
    config = SparseMoEConfig.from_pretrained("./outputs/hf_model")
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .moe_layer import MoELayer


class SparseMoEConfig(PretrainedConfig):
    """
    Configuration class for the Sparse Mixture-of-Experts model.
    Stores all architectural hyperparameters needed to reconstruct the model.
    """
    model_type = "sparse_moe"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        d_ff: int = 512,
        num_experts: int = 4,
        top_k: int = 2,
        capacity_factor: float = 1.5,
        noisy_routing: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_routing = noisy_routing


class SparseMoEForCausalLM(PreTrainedModel):
    """
    HuggingFace-compatible wrapper around the Sparse MoE architecture.
    Supports `from_pretrained()`, `save_pretrained()`, and standard
    forward pass returning `CausalLMOutputWithPast`.
    """
    config_class = SparseMoEConfig

    def __init__(self, config: SparseMoEConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.moe = MoELayer(
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_experts=config.num_experts,
            top_k=config.top_k,
            capacity_factor=config.capacity_factor,
            noisy_routing=config.noisy_routing,
        )
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        """
        Forward pass compatible with HuggingFace's CausalLM interface.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            labels: Optional target IDs for computing loss [batch_size, seq_len]

        Returns:
            CausalLMOutputWithPast with loss (if labels provided) and logits.
        """
        emb = self.embedding(input_ids)
        moe_out, _ = self.moe(emb)
        logits = self.fc_out(moe_out)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_text(self, idx, max_new_tokens, temperature=1.0):
        """
        Autoregressive text generation (matches the original MoEModel.generate API).
        """
        self.eval()
        for _ in range(max_new_tokens):
            outputs = self(input_ids=idx)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
