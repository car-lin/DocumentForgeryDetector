"""ViT RGB + EfficientNet forensic + fusion + two-stage heads (single file)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class UnifiedForgeryModel(nn.Module):
    def __init__(self):
        # Initialize the parent PyTorch module class.
        super().__init__()

        # RGB branch:
        # Uses a pretrained Vision Transformer to extract high-level visual features
        # from the original document image. num_classes=0 removes the classifier head
        # so the model returns feature embeddings instead of final class predictions.
        self.rgb_branch = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)

        # Forensic branch:
        # Uses a pretrained EfficientNet to extract features from the stacked forensic maps
        # (ELA, SRM, FFT). in_chans=3 matches the 3-channel forensic tensor.
        # num_classes=0 again removes the classification head and returns feature embeddings.
        self.forensic_branch = timm.create_model("efficientnet_b0", pretrained=True, in_chans=3, num_classes=0)

        # Fusion block:
        # Combines features from both branches into a shared representation.
        # ViT contributes 768 features and EfficientNet contributes 1280 features,
        # so they are concatenated into a single vector of size 2048 and projected to 512.
        # ReLU adds non-linearity and Dropout helps reduce overfitting.
        self.fusion = nn.Sequential(
            nn.Linear(768 + 1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Stage 1 head:
        # Binary classifier for real vs forged.
        self.stage1_head = nn.Linear(512, 2)

        # Stage 2 head:
        # Binary classifier for edited vs AI-generated,
        # applied after the image is considered forged.
        self.stage2_head = nn.Linear(512, 2)

    def forward(self, rgb: torch.Tensor, forensic: torch.Tensor):
        # Extract visual features from the RGB document image.
        rgb_feat = self.rgb_branch(rgb)

        # Extract forensic features from the 3-channel forensic input.
        forensic_feat = self.forensic_branch(forensic)

        # Concatenate both feature vectors and pass them through the fusion layer
        # to create one joint representation for downstream classification.
        fused = self.fusion(torch.cat([rgb_feat, forensic_feat], dim=1))

        # Return both intermediate features and final logits:
        # - fused: shared representation after fusion
        # - stage1_logits: prediction scores for real vs forged
        # - stage2_logits: prediction scores for edited vs AI-generated
        # - rgb_feat / forensic_feat: branch-specific features for analysis or debugging
        return {
            "fused": fused,
            "stage1_logits": self.stage1_head(fused),
            "stage2_logits": self.stage2_head(fused),
            "rgb_feat": rgb_feat,
            "forensic_feat": forensic_feat,
        }