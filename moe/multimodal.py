"""
Multi-Modal Mixture-of-Experts Components.

Extends the existing MoE architecture to handle text, image, and multi-modal
inputs across 10 tasks using a shared sparse routing mechanism.

Tasks:
    0-4: Text tasks (Code, Maths, Science, Law, Story)
    5-7: Vision tasks (ImgClassify, ImgReconstruct, ImgEdge)
    8-9: Multi-modal tasks (ImgCaption, VQA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_layer import MoELayer


# ---------------------------------------------------------------------------
# Patch Encoder
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """
    Converts 32x32 RGB images into a sequence of patch embeddings compatible
    with the MoE layer.

    Splits each image into non-overlapping patches, linearly projects them to
    ``d_model`` dimensions, and adds learned positional embeddings.
    """

    def __init__(self, d_model: int, patch_size: int = 8, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.in_channels = in_channels

        # Number of patches along each spatial dimension (assumes 32x32 images)
        self.num_patches_h = 32 // patch_size  # 4
        self.num_patches_w = 32 // patch_size  # 4
        self.num_patches = self.num_patches_h * self.num_patches_w  # 16

        patch_dim = patch_size * patch_size * in_channels  # 192 for 8x8x3
        self.projection = nn.Linear(patch_dim, d_model)
        self.position_embeddings = nn.Parameter(
            torch.randn(self.num_patches, d_model) * 0.02
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, 32, 32] input images.

        Returns:
            Patch embeddings of shape [B, num_patches, d_model].
        """
        B, C, H, W = images.shape
        p = self.patch_size

        # Reshape into patches: [B, C, H//p, p, W//p, p]
        x = images.reshape(B, C, H // p, p, W // p, p)
        # Rearrange to [B, H//p, W//p, C, p, p] then flatten patch dims
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)

        # Linear projection + positional embeddings
        x = self.projection(x) + self.position_embeddings

        return x


# ---------------------------------------------------------------------------
# Task-Specific Output Heads
# ---------------------------------------------------------------------------

class TextLMHead(nn.Module):
    """
    Next-token prediction head for text tasks (0-4) and caption generation
    (task 8) / VQA text output (task 9).
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]

        Returns:
            Logits of shape [B, seq_len, vocab_size].
        """
        return self.linear(x)


class ClassificationHead(nn.Module):
    """
    Classification head used for image classification (task 5) and
    VQA yes/no prediction (task 9).

    Pools over the sequence dimension, applies layer normalisation, and
    projects through a two-layer MLP with ReLU activation.
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]

        Returns:
            Logits of shape [B, num_classes].
        """
        # Mean-pool over the sequence dimension
        pooled = x.mean(dim=1)  # [B, d_model]
        out = self.norm(pooled)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class ReconstructionHead(nn.Module):
    """
    Image reconstruction head (task 6).

    Projects each patch embedding back to pixel space and reshapes to the
    original image dimensions.
    """

    def __init__(self, d_model: int, patch_size: int = 8, out_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches_h = 32 // patch_size
        self.num_patches_w = 32 // patch_size

        patch_dim = patch_size * patch_size * out_channels
        self.projection = nn.Linear(d_model, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_patches, d_model]

        Returns:
            Reconstructed image of shape [B, out_channels, 32, 32].
        """
        B = x.shape[0]
        p = self.patch_size

        # Project back to patch pixel space
        patches = self.projection(x)  # [B, num_patches, p*p*C]
        patches = patches.reshape(
            B, self.num_patches_h, self.num_patches_w,
            self.out_channels, p, p
        )
        # Rearrange to [B, C, H, W]
        out = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.out_channels, self.num_patches_h * p, self.num_patches_w * p
        )
        return out


class EdgeDetectionHead(nn.Module):
    """
    Edge detection head (task 7).

    Projects each patch embedding to a single-channel patch and applies
    sigmoid for binary edge output.
    """

    def __init__(self, d_model: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = 32 // patch_size
        self.num_patches_w = 32 // patch_size

        patch_dim = patch_size * patch_size * 1
        self.projection = nn.Linear(d_model, patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_patches, d_model]

        Returns:
            Edge map of shape [B, 1, 32, 32] with values in [0, 1].
        """
        B = x.shape[0]
        p = self.patch_size

        patches = self.projection(x)  # [B, num_patches, p*p]
        patches = patches.reshape(
            B, self.num_patches_h, self.num_patches_w, 1, p, p
        )
        out = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            B, 1, self.num_patches_h * p, self.num_patches_w * p
        )
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# Multi-Modal Multi-Task MoE Model
# ---------------------------------------------------------------------------

class MultiModalMoEModel(nn.Module):
    """
    Multi-Modal Multi-Task MoE Model with 8 experts supporting 10 tasks.

    All task types (text, image, multi-modal) are processed through a
    **single** MoE forward pass so the router sees tokens from every
    modality and can learn to specialise its experts accordingly.
    """

    TASK_NAMES = [
        'Code', 'Maths', 'Science', 'Law', 'Story',      # Text: 0-4
        'ImgClassify', 'ImgReconstruct', 'ImgEdge',       # Vision: 5-7
        'ImgCaption', 'VQA',                               # Multi-modal: 8-9
    ]
    NUM_TASKS = 10

    TEXT_TASKS = {0, 1, 2, 3, 4}
    IMAGE_TASKS = {5, 6, 7}
    MULTIMODAL_TASKS = {8, 9}

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        num_shape_classes: int = 5,
        patch_size: int = 8,
        capacity_factor: float = 1.5,
        noisy_routing: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.patch_size = patch_size

        # --- Encoders ---
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.patch_encoder = PatchEncoder(d_model, patch_size)

        # --- Shared MoE backbone ---
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            noisy_routing=noisy_routing,
        )

        # --- Task heads ---
        self.text_lm_head = TextLMHead(d_model, vocab_size)
        self.classify_head = ClassificationHead(d_model, num_shape_classes)
        self.reconstruct_head = ReconstructionHead(d_model, patch_size, 3)
        self.edge_head = EdgeDetectionHead(d_model, patch_size)
        self.vqa_head = ClassificationHead(d_model, 2)  # yes / no

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, task_ids, text_input=None, images=None):
        """
        Args:
            task_ids: [B] tensor of task IDs (0-9).
            text_input: [B, seq_len] token IDs (required for text &
                        multi-modal tasks; may be ``None`` for pure-image
                        batches).
            images: [B, 3, 32, 32] images (required for image & multi-modal
                    tasks; may be ``None`` for pure-text batches).

        Returns:
            results: dict with possible keys:
                ``'text_logits'``, ``'class_logits'``, ``'recon'``,
                ``'edges'``, ``'vqa_logits'`` — each mapping to a tensor
                whose batch dimension corresponds to the relevant subset of
                samples.
            aux_metrics: dict of auxiliary metrics from the MoE layer.
        """
        device = task_ids.device
        B = task_ids.shape[0]

        # ----- 1. Group samples by modality type -------------------------
        text_mask = torch.zeros(B, dtype=torch.bool, device=device)
        image_mask = torch.zeros(B, dtype=torch.bool, device=device)
        mm_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for t in self.TEXT_TASKS:
            text_mask |= (task_ids == t)
        for t in self.IMAGE_TASKS:
            image_mask |= (task_ids == t)
        for t in self.MULTIMODAL_TASKS:
            mm_mask |= (task_ids == t)

        text_indices = text_mask.nonzero(as_tuple=False).squeeze(-1)
        image_indices = image_mask.nonzero(as_tuple=False).squeeze(-1)
        mm_indices = mm_mask.nonzero(as_tuple=False).squeeze(-1)

        # ----- 2. Build embeddings per group ------------------------------
        embeddings_list = []   # list of [Bi, Si, d_model] tensors
        group_info = []        # (group_name, batch_size, seq_len)

        # Text-only tasks
        if text_indices.numel() > 0:
            text_emb = self.text_embedding(text_input[text_indices])  # [Bt, S, D]
            embeddings_list.append(text_emb)
            group_info.append(('text', text_emb.shape[0], text_emb.shape[1]))

        # Image-only tasks
        if image_indices.numel() > 0:
            img_emb = self.patch_encoder(images[image_indices])  # [Bi, 16, D]
            embeddings_list.append(img_emb)
            group_info.append(('image', img_emb.shape[0], img_emb.shape[1]))

        # Multi-modal tasks (image patches + text tokens concatenated)
        if mm_indices.numel() > 0:
            mm_img_emb = self.patch_encoder(images[mm_indices])        # [Bm, 16, D]
            mm_txt_emb = self.text_embedding(text_input[mm_indices])   # [Bm, S, D]
            mm_emb = torch.cat([mm_img_emb, mm_txt_emb], dim=1)       # [Bm, 16+S, D]
            embeddings_list.append(mm_emb)
            group_info.append(('mm', mm_emb.shape[0], mm_emb.shape[1]))

        # ----- 3. Pad sequences to the same length & concatenate ----------
        max_seq = max(e.shape[1] for e in embeddings_list)
        padded = []
        for emb in embeddings_list:
            if emb.shape[1] < max_seq:
                pad = torch.zeros(
                    emb.shape[0], max_seq - emb.shape[1], self.d_model,
                    device=device, dtype=emb.dtype,
                )
                emb = torch.cat([emb, pad], dim=1)
            padded.append(emb)

        combined = torch.cat(padded, dim=0)  # [B_total, max_seq, D]

        # ----- 4. Single MoE forward pass ---------------------------------
        moe_out, aux_metrics = self.moe(combined)  # [B_total, max_seq, D]

        # ----- 5. Split output back by group ------------------------------
        results = {}
        offset = 0
        for group_name, group_bs, group_seq in group_info:
            group_out = moe_out[offset: offset + group_bs, :group_seq, :]
            offset += group_bs

            if group_name == 'text':
                results['text_logits'] = self.text_lm_head(group_out)

            elif group_name == 'image':
                # Dispatch to task-specific heads per sample
                img_task_ids = task_ids[image_indices]
                self._dispatch_image_heads(
                    group_out, img_task_ids, results,
                )

            elif group_name == 'mm':
                mm_task_ids = task_ids[mm_indices]
                num_img_patches = self.patch_encoder.num_patches
                self._dispatch_mm_heads(
                    group_out, mm_task_ids, num_img_patches, results,
                )

        return results, aux_metrics

    # ------------------------------------------------------------------
    # Internal head dispatching helpers
    # ------------------------------------------------------------------

    def _dispatch_image_heads(self, out, task_ids_sub, results):
        """Route image-task outputs to the correct head."""
        # Task 5 — classification
        cls_mask = (task_ids_sub == 5)
        if cls_mask.any():
            results['class_logits'] = self.classify_head(out[cls_mask])

        # Task 6 — reconstruction
        rec_mask = (task_ids_sub == 6)
        if rec_mask.any():
            results['recon'] = self.reconstruct_head(out[rec_mask])

        # Task 7 — edge detection
        edge_mask = (task_ids_sub == 7)
        if edge_mask.any():
            results['edges'] = self.edge_head(out[edge_mask])

    def _dispatch_mm_heads(self, out, task_ids_sub, num_img_patches, results):
        """Route multi-modal task outputs to the correct head."""
        img_part = out[:, :num_img_patches, :]
        txt_part = out[:, num_img_patches:, :]

        # Task 8 — image captioning (text LM head on the text portion)
        cap_mask = (task_ids_sub == 8)
        if cap_mask.any():
            results['text_logits'] = self.text_lm_head(txt_part[cap_mask])

        # Task 9 — VQA (classification head on image+text, text logits for
        # generative answer, and yes/no classification head)
        vqa_mask = (task_ids_sub == 9)
        if vqa_mask.any():
            results['vqa_logits'] = self.vqa_head(out[vqa_mask])
