"""
RWR (Reward-Weighted Regression) for GR00T N1.6.

Assigns rewards to preference pairs:
    winner -> reward = +1.0
    loser  -> reward =  0.0

Training loss per sample:
    L_RWR = w(r) * L_flow(a, o)   where w(r) = exp(r / temperature)

Winner gets weight exp(1/T), loser gets weight exp(0/T) = 1.
The network is pushed to fit the winner trajectories more strongly.

Usage:
    python gr00t_rlhf/algos/rwr.py \\
        --model_path karthikpythireddi93/gr00t-n16-gr1-tabletop-sft \\
        --hdf5_path preference_data/gr1/all_tasks_preferences.hdf5 \\
        --output_dir outputs/rwr_groot \\
        --temperature 1.0 --n_epochs 3 --batch_size 4
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t_rlhf.datasets import GR00TPreferenceDataset, make_preference_collator


EMBODIMENT_TAG = "new_embodiment"
VIDEO_KEYS  = ["ego_view"]
STATE_KEYS  = ["left_arm", "left_hand", "left_leg", "neck",
               "right_arm", "right_hand", "right_leg", "waist"]
ACTION_KEYS = STATE_KEYS


def compute_flow_loss(model: Gr00tN1d6, batch: dict, device: str) -> torch.Tensor:
    """Forward pass; returns per-sample flow-matching loss, shape (B,)."""
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in batch.items()}
    out = model(inputs)
    action_loss = out["action_loss"]   # (B, H, D)
    action_mask = out["action_mask"]   # (B, H, D)
    return (action_loss * action_mask).sum(dim=(1, 2)) / (action_mask.sum(dim=(1, 2)) + 1e-6)


def rwr_loss(loss_w: torch.Tensor, loss_l: torch.Tensor,
             temperature: float) -> torch.Tensor:
    """
    Reward-weighted regression loss.
    winner reward = 1.0, loser reward = 0.0
    weight = exp(reward / temperature) (normalized within each pair)
    """
    w_winner = torch.exp(torch.tensor(1.0 / temperature))
    w_loser  = torch.exp(torch.tensor(0.0 / temperature))
    # Normalize weights
    w_total  = w_winner + w_loser
    w_w = w_winner / w_total
    w_l = w_loser  / w_total
    return (w_w * loss_w + w_l * loss_l).mean()


def train_rwr(
    model_path: str,
    hdf5_path: str,
    output_dir: str,
    temperature: float = 1.0,
    n_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-5,
    n_windows_per_pair: int = 5,
    device: str = "cuda",
    use_wandb: bool = False,
    wandb_project: str = "gr00t-rwr",
):
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            temperature=temperature, n_epochs=n_epochs,
            batch_size=batch_size, lr=lr, model_path=model_path,
        ))

    print(f"[RWR] Loading policy from {model_path}")
    policy = Gr00tN1d6.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    policy.train()

    dataset  = GR00TPreferenceDataset(hdf5_path, n_windows_per_pair=n_windows_per_pair)
    collator = make_preference_collator(EMBODIMENT_TAG, ACTION_KEYS, STATE_KEYS, VIDEO_KEYS)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collator, num_workers=0, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=lr
    )

    step = 0
    for epoch in range(n_epochs):
        for batch in loader:
            loss_w = compute_flow_loss(policy, batch["winner"], device)
            loss_l = compute_flow_loss(policy, batch["loser"],  device)
            loss   = rwr_loss(loss_w, loss_l, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            step += 1
            print(f"[RWR] epoch={epoch} step={step} loss={loss.item():.4f} "
                  f"loss_w={loss_w.mean().item():.4f} loss_l={loss_l.mean().item():.4f}")
            if use_wandb:
                import wandb
                wandb.log({"rwr_loss": loss.item(),
                           "loss_winner": loss_w.mean().item(),
                           "loss_loser": loss_l.mean().item()}, step=step)

        ckpt = os.path.join(output_dir, f"checkpoint-epoch{epoch}")
        policy.save_pretrained(ckpt)
        print(f"[RWR] Checkpoint: {ckpt}")

    policy.save_pretrained(output_dir)
    print(f"[RWR] Done. Model at {output_dir}")
    if use_wandb:
        import wandb; wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",         required=True)
    parser.add_argument("--hdf5_path",          required=True)
    parser.add_argument("--output_dir",         default="outputs/rwr_groot")
    parser.add_argument("--temperature",        type=float, default=1.0)
    parser.add_argument("--n_epochs",           type=int,   default=3)
    parser.add_argument("--batch_size",         type=int,   default=4)
    parser.add_argument("--lr",                 type=float, default=1e-5)
    parser.add_argument("--n_windows_per_pair", type=int,   default=5)
    parser.add_argument("--use_wandb",          action="store_true")
    parser.add_argument("--wandb_project",      default="gr00t-rwr")
    args = parser.parse_args()
    train_rwr(**{k: v for k, v in vars(args).items()})
