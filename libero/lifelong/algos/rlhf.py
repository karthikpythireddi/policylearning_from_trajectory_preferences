"""
rlhf.py

Reinforcement Learning from Human Feedback (RLHF) via Reward-Weighted Regression.

Two-phase training per task:

  Phase 1 — Reward Model Training
    Train a trajectory reward model on preference pairs using Bradley-Terry loss:
      L_RM = -log σ( R(τ_w) - R(τ_l) )
    The reward model encoder is initialized from the BC checkpoint.

  Phase 2 — Reward-Weighted Regression (RWR)
    Use the trained reward model to score windows from the BC training dataset.
    Fine-tune the policy with weighted BC loss:
      L_RWR = -E_{τ~demos}[ w(τ) * log π(a|τ) ]
    where w(τ) = softmax(R(τ) / temperature) over the batch

    This upweights demonstrations that the reward model scores highly and
    downweights low-reward ones, without requiring new rollouts.

Reference:
  Peters & Schaal, "Reinforcement Learning by Reward-Weighted Regression for
  Operational Space Control", ICML 2007.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets_preference import make_preference_dataloader
from libero.lifelong.metric import evaluate_one_task_success
from libero.lifelong.models.reward_model import RewardModel
from libero.lifelong.utils import safe_device, torch_load_model, torch_save_model


class RLHF(Sequential):
    """
    RLHF via Reward-Weighted Regression.

    Config keys read from cfg.lifelong:
      rm_epochs             (int,   default 20)    reward model training epochs
      rm_lr                 (float, default 1e-4)  reward model learning rate
      rwr_temperature       (float, default 1.0)   softmax temperature for weights
      bc_checkpoint_dir     (str,   required)       dir with task{i}_model.pth
      preference_data_dir   (str,   required)       dir with task{i}_preferences.hdf5
      n_windows_per_pair    (int,   default 5)      windows per preference pair
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.rwr_temperature = getattr(cfg.lifelong, "rwr_temperature", 1.0)
        self.rm_epochs       = getattr(cfg.lifelong, "rm_epochs", 20)
        self.rm_lr           = getattr(cfg.lifelong, "rm_lr", 1e-4)

    # ------------------------------------------------------------------
    # Phase 1: Reward Model Training
    # ------------------------------------------------------------------

    def _train_reward_model(self, task_id, benchmark, hdf5_path, bc_state_dict):
        """
        Train a reward model on preference pairs for one task.
        Returns the trained RewardModel (on device, eval mode).
        """
        print(f"\n  [Phase 1] Training reward model for task {task_id}...")

        # Build reward model and initialize encoder from BC checkpoint
        reward_model = RewardModel(self.cfg, self.cfg.shape_meta)
        reward_model.load_encoder_from_bc(bc_state_dict)
        reward_model = safe_device(reward_model, self.cfg.device)

        optimizer = torch.optim.AdamW(
            reward_model.parameters(), lr=self.rm_lr, weight_decay=1e-4
        )

        task_emb   = benchmark.get_task_emb(task_id)
        n_windows  = getattr(self.cfg.lifelong, "n_windows_per_pair", 5)
        dataloader, _ = make_preference_dataloader(
            hdf5_path          = hdf5_path,
            task_emb           = task_emb,
            seq_len            = self.cfg.data.seq_len,
            n_windows_per_pair = n_windows,
            batch_size         = self.cfg.train.batch_size,
            shuffle            = True,
        )

        for epoch in range(1, self.rm_epochs + 1):
            reward_model.train()
            epoch_loss = 0.0

            for batch in dataloader:
                device = self.cfg.device
                winner_data = {
                    "obs":      {k: v.to(device) for k, v in batch["winner"]["obs"].items()},
                    "actions":  batch["winner"]["actions"].to(device),
                    "task_emb": batch["task_emb"].to(device),
                }
                loser_data = {
                    "obs":      {k: v.to(device) for k, v in batch["loser"]["obs"].items()},
                    "actions":  batch["loser"]["actions"].to(device),
                    "task_emb": batch["task_emb"].to(device),
                }

                optimizer.zero_grad()
                reward_w = reward_model(winner_data)   # (B,)
                reward_l = reward_model(loser_data)    # (B,)
                loss = reward_model.bradley_terry_loss(reward_w, reward_l)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= max(len(dataloader), 1)
            if epoch % 5 == 0 or epoch == 1:
                # Compute preference accuracy: how often R(τ_w) > R(τ_l)
                reward_model.eval()
                correct = 0
                total   = 0
                with torch.no_grad():
                    for batch in dataloader:
                        device = self.cfg.device
                        wd = {"obs": {k: v.to(device) for k, v in batch["winner"]["obs"].items()},
                              "actions": batch["winner"]["actions"].to(device),
                              "task_emb": batch["task_emb"].to(device)}
                        ld = {"obs": {k: v.to(device) for k, v in batch["loser"]["obs"].items()},
                              "actions": batch["loser"]["actions"].to(device),
                              "task_emb": batch["task_emb"].to(device)}
                        rw = reward_model(wd)
                        rl = reward_model(ld)
                        correct += (rw > rl).sum().item()
                        total   += len(rw)
                acc = correct / max(total, 1)
                print(f"    RM Epoch {epoch:3d} | loss: {epoch_loss:.4f} | "
                      f"preference acc: {acc:.3f}")

        reward_model.eval()
        print(f"  [Phase 1] Reward model training done.")
        return reward_model

    # ------------------------------------------------------------------
    # Phase 2: Reward-Weighted Regression
    # ------------------------------------------------------------------

    def _compute_rwr_loss(self, data, reward_model):
        """
        Compute RWR loss for a batch of BC demonstrations.

        For each window in the batch:
          1. Compute reward R(τ) from the frozen reward model
          2. Compute weights w = softmax(R / temperature)
          3. Loss = -mean(w * log π(a|τ))

        Returns scalar loss.
        """
        with torch.no_grad():
            rewards = reward_model(data)                          # (B,)
            weights = F.softmax(rewards / self.rwr_temperature, dim=0)  # (B,)

        dist      = self.policy.forward(data)                    # GMM
        log_probs = dist.log_prob(data["actions"])               # (B, T)
        log_probs = log_probs.mean(dim=-1)                       # (B,)

        loss = -(weights * log_probs).sum()                      # weighted BC
        return loss

    # ------------------------------------------------------------------
    # Task training loop
    # ------------------------------------------------------------------

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        """
        Two-phase RLHF training for one task.
        `dataset` is the standard BC SequenceVLDataset used in Phase 2.
        """
        self.start_task(task_id)

        # ---- Load BC checkpoint ----
        bc_dir    = self.cfg.lifelong.bc_checkpoint_dir
        ckpt_path = os.path.join(bc_dir, f"task{task_id}_model.pth")

        if not os.path.exists(ckpt_path):
            print(f"  [RLHF skip] BC checkpoint not found: {ckpt_path}")
            return 0.0, 0.0

        bc_state_dict, _, _ = torch_load_model(ckpt_path)
        self.policy.load_state_dict(bc_state_dict)
        self.policy = safe_device(self.policy, self.cfg.device)

        # ---- Check preference data ----
        pref_dir  = self.cfg.lifelong.preference_data_dir
        hdf5_path = os.path.join(
            pref_dir, self.cfg.benchmark_name, f"task{task_id}_preferences.hdf5"
        )

        if not os.path.exists(hdf5_path):
            print(f"  [RLHF skip] Preference data not found: {hdf5_path}")
            return 0.0, 0.0

        # ---- Phase 1: Train reward model ----
        reward_model = self._train_reward_model(
            task_id, benchmark, hdf5_path, bc_state_dict
        )
        # Freeze reward model for Phase 2
        for p in reward_model.parameters():
            p.requires_grad_(False)
        reward_model.eval()

        # ---- Phase 2: RWR fine-tuning on BC demonstrations ----
        print(f"\n  [Phase 2] RWR fine-tuning for task {task_id}...")

        train_dataloader = DataLoader(
            dataset,
            batch_size   = self.cfg.train.batch_size,
            num_workers  = 0,
            sampler      = RandomSampler(dataset),
        )

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )
        task     = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        prev_success_rate = -1.0
        idx_at_best_succ  = 0
        cumulated_counter = 0.0
        losses    = []
        successes = []

        for epoch in range(0, self.cfg.train.n_epochs + 1):
            t0 = time.time()

            if epoch > 0:
                self.policy.train()
                epoch_loss = 0.0

                for data in train_dataloader:
                    data = self.map_tensor_to_device(data)
                    data = self.policy.preprocess_input(data, train_mode=True)

                    self.optimizer.zero_grad()
                    loss = self._compute_rwr_loss(data, reward_model)
                    (self.loss_scale * loss).backward()

                    if self.cfg.train.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.cfg.train.grad_clip
                        )
                    self.optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= max(len(train_dataloader), 1)
            else:
                epoch_loss = 0.0

            t1 = time.time()
            print(
                f"[info] Epoch: {epoch:3d} | RWR loss: {epoch_loss:8.4f} | "
                f"time: {(t1-t0)/60:4.2f}"
            )

            if epoch % self.cfg.eval.eval_every == 0:
                losses.append(epoch_loss)
                success_rate = evaluate_one_task_success(
                    cfg      = self.cfg,
                    algo     = self,
                    task     = task,
                    task_emb = task_emb,
                    task_id  = task_id,
                    sim_states = None,
                    task_str = "",
                )
                successes.append(success_rate)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ  = len(losses) - 1

                cumulated_counter += 1.0
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} | "
                    f"best succ: {prev_success_rate}"
                )

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        losses    = np.array(losses)
        successes = np.array(successes)
        losses[idx_at_best_succ:]    = losses[idx_at_best_succ]
        successes[idx_at_best_succ:] = successes[idx_at_best_succ]

        del reward_model
        torch.cuda.empty_cache()

        return (
            successes.sum() / max(cumulated_counter, 1),
            losses.sum()    / max(cumulated_counter, 1),
        )
