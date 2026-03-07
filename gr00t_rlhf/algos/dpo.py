"""
DPO (Direct Preference Optimization) for GR00T N1.6 flow-matching policy.

The GR00T action head uses flow matching. The per-sample loss:
    L(a, o) = MSE(pred_velocity, target_velocity)
is a proxy for -log pi(a|o). DPO loss:
    L_DPO = -log sigma(beta * (L_loser - L_winner - (L_ref_loser - L_ref_winner)))

Usage:
    python gr00t_rlhf/algos/dpo.py \\
        --model_path karthikpythireddi93/gr00t-n16-gr1-tabletop-sft \\
        --hdf5_path preference_data/gr1/all_tasks_preferences.hdf5 \\
        --output_dir outputs/dpo_groot \\
        --beta 0.1 --n_epochs 3 --batch_size 2
"""

import argparse
import copy
import os

import torch
import torch.nn.functional as F
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


def dpo_loss(policy_w, policy_l, ref_w, ref_l, beta: float) -> torch.Tensor:
    """
    DPO loss from per-sample flow-matching losses (lower = higher log prob).
    logit = beta * (policy_l - policy_w - (ref_l - ref_w))
    """
    logit = beta * (policy_l - policy_w - (ref_l - ref_w))
    return -F.logsigmoid(logit).mean()


def train_dpo(
    model_path: str,
    hdf5_path: str,
    output_dir: str,
    beta: float = 0.1,
    n_epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-5,
    n_windows_per_pair: int = 5,
    device: str = "cuda",
    use_wandb: bool = False,
    wandb_project: str = "gr00t-dpo",
):
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            beta=beta, n_epochs=n_epochs, batch_size=batch_size,
            lr=lr, model_path=model_path,
        ))

    print(f"[DPO] Loading policy from {model_path}")
    policy = Gr00tN1d6.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    policy.train()

    print("[DPO] Creating frozen reference policy")
    ref_policy = copy.deepcopy(policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

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
            policy_w = compute_flow_loss(policy, batch["winner"], device)
            policy_l = compute_flow_loss(policy, batch["loser"],  device)
            with torch.no_grad():
                ref_w = compute_flow_loss(ref_policy, batch["winner"], device)
                ref_l = compute_flow_loss(ref_policy, batch["loser"],  device)

            loss = dpo_loss(policy_w, policy_l, ref_w, ref_l, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            step += 1
            reward_acc = (policy_l > policy_w).float().mean().item()
            print(f"[DPO] epoch={epoch} step={step} loss={loss.item():.4f} "
                  f"reward_acc={reward_acc:.3f}")
            if use_wandb:
                import wandb
                wandb.log({"dpo_loss": loss.item(), "reward_acc": reward_acc}, step=step)

        ckpt = os.path.join(output_dir, f"checkpoint-epoch{epoch}")
        policy.save_pretrained(ckpt)
        print(f"[DPO] Checkpoint: {ckpt}")

    policy.save_pretrained(output_dir)
    print(f"[DPO] Done. Model at {output_dir}")
    if use_wandb:
        import wandb; wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",         required=True)
    parser.add_argument("--hdf5_path",          required=True)
    parser.add_argument("--output_dir",         default="outputs/dpo_groot")
    parser.add_argument("--beta",               type=float, default=0.1)
    parser.add_argument("--n_epochs",           type=int,   default=3)
    parser.add_argument("--batch_size",         type=int,   default=2)
    parser.add_argument("--lr",                 type=float, default=1e-5)
    parser.add_argument("--n_windows_per_pair", type=int,   default=5)
    parser.add_argument("--use_wandb",          action="store_true")
    parser.add_argument("--wandb_project",      default="gr00t-dpo")
    args = parser.parse_args()
    train_dpo(**{k: v for k, v in vars(args).items()})
