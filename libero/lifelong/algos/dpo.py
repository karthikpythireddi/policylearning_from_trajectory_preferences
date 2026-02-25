"""
dpo.py

Direct Policy Optimization (DPO) for robotic manipulation.

Given preference pairs (winner τ_w, loser τ_l) generated from the same policy,
DPO fine-tunes the policy directly without training a reward model:

  L_DPO = -log σ( β * [ (log π(τ_w) - log π_ref(τ_w))
                       - (log π(τ_l) - log π_ref(τ_l)) ] )

where:
  π     = current (trainable) policy
  π_ref = frozen BC checkpoint (reference policy)
  β     = KL penalty coefficient (controls how far π drifts from π_ref)

Both π and π_ref start from the same BC checkpoint.
Log probs are computed as the mean per-timestep log prob over a seq_len window.

Reference:
  Rafailov et al., "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model", NeurIPS 2023.
"""

import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets_preference import make_preference_dataloader
from libero.lifelong.metric import evaluate_one_task_success
from libero.lifelong.utils import safe_device, torch_load_model, torch_save_model


class DPO(Sequential):
    """
    DPO fine-tuning on top of a pretrained BC policy.

    Config keys read from cfg.lifelong:
      dpo_beta              (float, default 0.1)  KL penalty coefficient
      bc_checkpoint_dir     (str, required)        dir with task{i}_model.pth
      preference_data_dir   (str, required)        dir with task{i}_preferences.hdf5
      n_windows_per_pair    (int, default 5)       windows sampled per preference pair
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.beta         = getattr(cfg.lifelong, "dpo_beta", 0.1)
        self.ref_policy   = None  # set in learn_one_task

    # ------------------------------------------------------------------
    # Core DPO computation
    # ------------------------------------------------------------------

    def _compute_logprob(self, policy, data):
        """
        Compute mean per-timestep log prob of actions under the policy.

        Args:
            policy: BCTransformerPolicy
            data:   dict with obs (B,T,...), actions (B,T,7), task_emb (B,768)
                    Already on the correct device and preprocessed.
        Returns:
            (B,) tensor — mean log prob over the T timesteps
        """
        dist = policy.forward(data)                    # GMM distribution
        log_probs = dist.log_prob(data["actions"])     # (B, T)
        return log_probs.mean(dim=-1)                  # (B,)

    def _dpo_loss(self, winner_data, loser_data):
        """
        Compute DPO loss for one batch of preference pairs.

        Returns:
            loss   (scalar tensor) — DPO loss
            margin (float)         — mean implicit reward margin (for logging)
        """
        # Apply augmentation once — both policies see identical augmented inputs
        # preprocess_input modifies obs images in-place and returns the same dict
        winner_data = self.policy.preprocess_input(winner_data, train_mode=True)
        loser_data  = self.policy.preprocess_input(loser_data,  train_mode=True)

        # Current policy log probs (with gradient)
        self.policy.train()
        log_pi_w = self._compute_logprob(self.policy, winner_data)  # (B,)
        log_pi_l = self._compute_logprob(self.policy, loser_data)   # (B,)

        # Reference policy log probs (no gradient)
        with torch.no_grad():
            self.ref_policy.eval()
            log_ref_w = self._compute_logprob(self.ref_policy, winner_data)  # (B,)
            log_ref_l = self._compute_logprob(self.ref_policy, loser_data)   # (B,)

        # DPO loss: -log σ(β * ((log π_w - log ref_w) - (log π_l - log ref_l)))
        logits = self.beta * ((log_pi_w - log_ref_w) - (log_pi_l - log_ref_l))
        loss   = -F.logsigmoid(logits).mean()

        # Implicit reward margin (positive means policy correctly prefers winner)
        margin = (log_pi_w - log_ref_w - log_pi_l + log_ref_l).detach().mean().item()

        return loss, margin

    def _prep_batch(self, traj_dict, task_emb_batch):
        """Move a winner/loser batch from DataLoader to the training device."""
        device = self.cfg.device
        return {
            "obs":      {k: v.to(device) for k, v in traj_dict["obs"].items()},
            "actions":  traj_dict["actions"].to(device),
            "task_emb": task_emb_batch.to(device),
        }

    # ------------------------------------------------------------------
    # Task training loop
    # ------------------------------------------------------------------

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        """
        Fine-tune the BC policy for one task using DPO.

        `dataset` is the standard BC SequenceVLDataset — we don't use it here.
        Instead we load the preference HDF5 file for this task.
        """
        self.start_task(task_id)

        # ---- Load BC checkpoint as starting point for both π and π_ref ----
        bc_dir   = self.cfg.lifelong.bc_checkpoint_dir
        ckpt_path = os.path.join(bc_dir, f"task{task_id}_model.pth")

        if not os.path.exists(ckpt_path):
            print(f"  [DPO skip] BC checkpoint not found: {ckpt_path}")
            return 0.0, 0.0

        state_dict, _, _ = torch_load_model(ckpt_path)
        self.policy.load_state_dict(state_dict)
        self.policy = safe_device(self.policy, self.cfg.device)

        # Frozen reference policy
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        # ---- Load preference dataset ----
        pref_dir  = self.cfg.lifelong.preference_data_dir
        hdf5_path = os.path.join(
            pref_dir, self.cfg.benchmark_name, f"task{task_id}_preferences.hdf5"
        )

        if not os.path.exists(hdf5_path):
            print(f"  [DPO skip] Preference data not found: {hdf5_path}")
            return 0.0, 0.0

        task_emb = benchmark.get_task_emb(task_id)
        n_windows = getattr(self.cfg.lifelong, "n_windows_per_pair", 5)

        dataloader, pref_dataset = make_preference_dataloader(
            hdf5_path    = hdf5_path,
            task_emb     = task_emb,
            seq_len      = self.cfg.data.seq_len,
            n_windows_per_pair = n_windows,
            batch_size   = self.cfg.train.batch_size,
            shuffle      = True,
        )

        print(f"  Preference dataset: {len(pref_dataset)} windows "
              f"from {hdf5_path.split('/')[-1]}")

        # ---- Training loop ----
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
                epoch_loss   = 0.0
                epoch_margin = 0.0

                for batch in dataloader:
                    winner_data = self._prep_batch(batch["winner"], batch["task_emb"])
                    loser_data  = self._prep_batch(batch["loser"],  batch["task_emb"])

                    self.optimizer.zero_grad()
                    loss, margin = self._dpo_loss(winner_data, loser_data)
                    (self.loss_scale * loss).backward()

                    if self.cfg.train.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.cfg.train.grad_clip
                        )

                    self.optimizer.step()
                    epoch_loss   += loss.item()
                    epoch_margin += margin

                epoch_loss   /= max(len(dataloader), 1)
                epoch_margin /= max(len(dataloader), 1)
            else:
                # Epoch 0: evaluate before any DPO updates (BC baseline perf)
                epoch_loss, epoch_margin = 0.0, 0.0

            t1 = time.time()
            print(
                f"[info] Epoch: {epoch:3d} | DPO loss: {epoch_loss:7.4f} | "
                f"reward margin: {epoch_margin:+.4f} | time: {(t1-t0)/60:4.2f}"
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

        # Clean up reference policy to free GPU memory
        del self.ref_policy
        self.ref_policy = None
        torch.cuda.empty_cache()

        return (
            successes.sum() / max(cumulated_counter, 1),
            losses.sum()    / max(cumulated_counter, 1),
        )
