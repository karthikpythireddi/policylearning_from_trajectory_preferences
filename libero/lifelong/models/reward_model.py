"""
reward_model.py

Trajectory reward model for RLHF fine-tuning on LIBERO.

Architecture:
  Reuses the BCTransformerPolicy's encoder (spatial + temporal transformer)
  and adds a lightweight scalar head on top:

    spatial_encode(obs, task_emb)  →  (B, T, num_modalities, E)
    temporal_encode(...)           →  (B, T, E)
    mean_pool over T               →  (B, E)
    MLP reward head                →  (B, 1)  scalar reward

The encoder can be initialized from a BC checkpoint, giving the reward model
a strong visual and proprioceptive representation from the start.

Training objective: Bradley-Terry preference model
  L = -log σ( R(τ_w) - R(τ_l) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy


class RewardModel(nn.Module):
    """
    Scalar reward model built on top of the BCTransformerPolicy encoder.

    Args:
        cfg:        Hydra config (same as used for BCTransformerPolicy)
        shape_meta: Dataset shape metadata
        hidden_size: Hidden size of the scalar reward MLP head
    """

    def __init__(self, cfg, shape_meta, hidden_size=128):
        super().__init__()

        # Reuse the full BCTransformerPolicy as encoder backbone
        # We only use spatial_encode + temporal_encode from it
        self.encoder = BCTransformerPolicy(cfg, shape_meta)
        embed_size   = cfg.policy.embed_size  # typically 64

        # Lightweight scalar head: E → hidden → 1
        self.reward_head = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, data):
        """
        Compute scalar reward for a batch of trajectory windows.

        Args:
            data: dict with obs (B,T,...), actions (B,T,7), task_emb (B,768)
                  Already preprocessed (augmentation applied if desired).
        Returns:
            (B,) tensor of scalar rewards
        """
        # Encode observations through the transformer backbone
        x = self.encoder.spatial_encode(data)   # (B, T, num_modalities, E)
        x = self.encoder.temporal_encode(x)     # (B, T, E)

        # Mean-pool over the time dimension to get a trajectory-level feature
        x = x.mean(dim=1)                       # (B, E)

        # Project to scalar reward
        reward = self.reward_head(x).squeeze(-1)  # (B,)
        return reward

    def bradley_terry_loss(self, reward_w, reward_l):
        """
        Bradley-Terry preference loss.
        Maximizes P(τ_w ≻ τ_l) = σ(R(τ_w) - R(τ_l))

        Args:
            reward_w: (B,) rewards for winner trajectories
            reward_l: (B,) rewards for loser  trajectories
        Returns:
            scalar loss
        """
        return -F.logsigmoid(reward_w - reward_l).mean()

    def load_encoder_from_bc(self, bc_state_dict):
        """
        Initialize encoder weights from a BC policy checkpoint.
        Keys in bc_state_dict map directly to self.encoder's state dict.
        """
        self.encoder.load_state_dict(bc_state_dict, strict=True)
        print("  [RewardModel] Encoder initialized from BC checkpoint.")
