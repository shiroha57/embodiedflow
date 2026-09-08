"""Minimal ChunkedActionTransformer.

Honest naming: this is NOT the ACT CVAE. It is a deterministic
encoder-only transformer that maps (rgb, robot_state) to an action chunk
per the contract-v0.1 action definition. No VAE, no language conditioning
(declared disabled in ModelConfig and exported to config.yaml / manifest).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from embodiedflow.dataplane.config import ModelConfig


class _CNNStem(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = cfg.rgb_shape[0]
        for out_channels, kernel, stride, padding in cfg.cnn_stem:
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb)


class ChunkedActionTransformerMinimal(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.instruction_conditioning != "disabled":
            raise ValueError(
                f"instruction_conditioning={cfg.instruction_conditioning!r} is not "
                "implemented; this model has no language pathway. Set 'disabled'."
            )
        self.stem = _CNNStem(cfg)
        feature_h = cfg.rgb_shape[1]
        feature_w = cfg.rgb_shape[2]
        for _, kernel, stride, padding in cfg.cnn_stem:
            feature_h = (feature_h + 2 * padding - kernel) // stride + 1
            feature_w = (feature_w + 2 * padding - kernel) // stride + 1
        if feature_h <= 0 or feature_w <= 0:
            raise ValueError("CNN stem over-shrinks the input; adjust cnn_stem")
        self.pool_size = cfg.pool_size
        self.num_image_tokens = (
            cfg.pool_size * cfg.pool_size
            if cfg.pool_size is not None
            else feature_h * feature_w
        )

        self.image_proj = nn.Linear(self.stem.out_channels, cfg.d_model)
        self.state_proj = nn.Linear(cfg.state_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_image_tokens, cfg.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, cfg.chunk_len * cfg.action_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        robot_state: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """rgb [B,C,H,W], robot_state [B,S] -> action chunk [B,L,A]."""
        del padding_mask  # loss masking happens in the caller
        features = self.stem(rgb)
        if self.pool_size is not None:
            features = nn.functional.adaptive_avg_pool2d(
                features, (self.pool_size, self.pool_size)
            )
        image_tokens = features.flatten(2).transpose(1, 2)  # [B, N, C]
        image_tokens = self.image_proj(image_tokens)
        state_token = self.state_proj(robot_state).unsqueeze(1)  # [B, 1, D]
        tokens = torch.cat([state_token, image_tokens], dim=1)
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0]  # state token output
        return self.head(pooled).view(-1, self.cfg.chunk_len, self.cfg.action_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def masked_chunk_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid (mask==1) chunk entries."""
    valid = mask.sum()
    if valid.item() == 0:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    squared = (prediction - target).pow(2) * mask.unsqueeze(-1)
    return squared.sum() / valid


def cosine_schedule(warmup_steps: int, total_steps: int) -> callable:
    """LambdaLR multiplier: linear warmup, then cosine decay to ~0."""

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return schedule
