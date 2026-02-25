"""
ppo.py

Proximal Policy Optimization (PPO) for RLHF on LIBERO.

Training uses a learned reward model to provide dense per-timestep rewards.
The policy is updated using the PPO-Clip objective.

Pipeline per task:
  Phase 1 — Train reward model on preference pairs (Bradley-Terry loss)
  Phase 2 — PPO fine-tuning:
    for each iteration:
      1. Collect N rollouts using the current policy
      2. Compute per-step rewards using the reward model (sliding window)
      3. Compute GAE advantages using the value network
      4. Run K inner epochs of PPO updates on mini-batches
      5. Evaluate and save best checkpoint

References:
  Schulman et al., "Proximal Policy Optimization Algorithms", 2017.
  Ouyang et al., "Training language models to follow instructions with
    human feedback (InstructGPT)", NeurIPS 2022.
"""

import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv
from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.rlhf import RLHF
from libero.lifelong.datasets_preference import make_preference_dataloader
from libero.lifelong.metric import evaluate_one_task_success, raw_obs_to_tensor_obs
from libero.lifelong.models.reward_model import RewardModel
from libero.lifelong.utils import safe_device, torch_load_model, torch_save_model


# ---------------------------------------------------------------------------
# Value Network
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    State value estimator V(s_t) for PPO.
    Same architecture as RewardModel: BCTransformerPolicy encoder + scalar head.
    Initialized from the BC checkpoint encoder weights.
    """

    def __init__(self, cfg, shape_meta, hidden_size=128):
        super().__init__()
        from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
        self.encoder    = BCTransformerPolicy(cfg, shape_meta)
        embed_size      = cfg.policy.embed_size
        self.value_head = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, data):
        """
        Args:
            data: dict with obs (B,T,...), task_emb (B,768)
        Returns:
            (B,) scalar value estimates
        """
        x = self.encoder.spatial_encode(data)   # (B, T, num_mod, E)
        x = self.encoder.temporal_encode(x)     # (B, T, E)
        x = x.mean(dim=1)                       # (B, E) — mean pool over time
        return self.value_head(x).squeeze(-1)   # (B,)

    def load_encoder_from_bc(self, bc_state_dict):
        self.encoder.load_state_dict(bc_state_dict, strict=True)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one episode's worth of data for PPO updates.
    Images are stored as uint8 to save memory; converted to float32 during sampling.
    """

    def __init__(self):
        self.obs       = {}        # {key: list of numpy arrays (H,W,C) or (dim,)}
        self.actions   = []        # list of (7,) numpy arrays
        self.log_probs = []        # list of scalar floats (old log prob)
        self.rewards   = []        # list of scalar floats
        self.values    = []        # list of scalar floats
        self.dones     = []        # list of bools

    def clear(self):
        self.__init__()

    def add(self, obs_step, action, log_prob, reward, value, done):
        """Add one timestep."""
        for k, v in obs_step.items():
            if k not in self.obs:
                self.obs[k] = []
            self.obs[k].append(v)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def length(self):
        return len(self.actions)


# ---------------------------------------------------------------------------
# PPO Algorithm
# ---------------------------------------------------------------------------

class PPO(Sequential):
    """
    PPO with RLHF reward model for robotic manipulation on LIBERO.

    Config keys read from cfg.lifelong:
      rm_epochs           (int,   default 20)    reward model training epochs
      rm_lr               (float, default 1e-4)  reward model LR
      ppo_iters           (int,   default 30)    outer PPO iterations
      n_rollouts_per_iter (int,   default 4)     rollouts collected per iteration
      ppo_epochs          (int,   default 4)     inner PPO update epochs per iter
      ppo_mini_batch      (int,   default 64)    mini-batch size for PPO updates
      clip_eps            (float, default 0.2)   PPO clipping coefficient
      gamma               (float, default 0.99)  discount factor
      gae_lambda          (float, default 0.95)  GAE lambda
      entropy_coef        (float, default 0.01)  entropy bonus coefficient
      value_coef          (float, default 0.5)   value loss coefficient
      max_steps           (int,   default 300)   max steps per rollout
      bc_checkpoint_dir   (str,   required)
      preference_data_dir (str,   required)
      n_windows_per_pair  (int,   default 5)
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        lc = cfg.lifelong
        self.ppo_iters           = getattr(lc, "ppo_iters",           30)
        self.n_rollouts_per_iter = getattr(lc, "n_rollouts_per_iter",  4)
        self.ppo_epochs          = getattr(lc, "ppo_epochs",           4)
        self.ppo_mini_batch      = getattr(lc, "ppo_mini_batch",      64)
        self.clip_eps            = getattr(lc, "clip_eps",           0.2)
        self.gamma               = getattr(lc, "gamma",              0.99)
        self.gae_lambda          = getattr(lc, "gae_lambda",         0.95)
        self.entropy_coef        = getattr(lc, "entropy_coef",       0.01)
        self.value_coef          = getattr(lc, "value_coef",          0.5)
        self.rm_epochs           = getattr(lc, "rm_epochs",           20)
        self.rm_lr               = getattr(lc, "rm_lr",             1e-4)
        self.ppo_max_steps       = getattr(lc, "max_steps",          300)

    # ------------------------------------------------------------------
    # Reward model training (reuse RLHF Phase 1)
    # ------------------------------------------------------------------

    def _train_reward_model(self, task_id, benchmark, hdf5_path, bc_state_dict):
        """Train reward model on preference pairs. Returns trained model."""
        print(f"\n  [Phase 1] Training reward model for task {task_id}...")
        reward_model = RewardModel(self.cfg, self.cfg.shape_meta)
        reward_model.load_encoder_from_bc(bc_state_dict)
        reward_model = safe_device(reward_model, self.cfg.device)

        optimizer = torch.optim.AdamW(
            reward_model.parameters(), lr=self.rm_lr, weight_decay=1e-4
        )
        task_emb   = benchmark.get_task_emb(task_id)
        n_windows  = getattr(self.cfg.lifelong, "n_windows_per_pair", 5)
        dataloader, _ = make_preference_dataloader(
            hdf5_path=hdf5_path, task_emb=task_emb,
            seq_len=self.cfg.data.seq_len, n_windows_per_pair=n_windows,
            batch_size=self.cfg.train.batch_size,
        )

        for epoch in range(1, self.rm_epochs + 1):
            reward_model.train()
            epoch_loss = 0.0
            for batch in dataloader:
                dev = self.cfg.device
                wd = {"obs": {k: v.to(dev) for k, v in batch["winner"]["obs"].items()},
                      "actions": batch["winner"]["actions"].to(dev),
                      "task_emb": batch["task_emb"].to(dev)}
                ld = {"obs": {k: v.to(dev) for k, v in batch["loser"]["obs"].items()},
                      "actions": batch["loser"]["actions"].to(dev),
                      "task_emb": batch["task_emb"].to(dev)}
                optimizer.zero_grad()
                loss = reward_model.bradley_terry_loss(reward_model(wd), reward_model(ld))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 5 == 0 or epoch == 1:
                print(f"    RM Epoch {epoch:3d} | loss: {epoch_loss/max(len(dataloader),1):.4f}")

        reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad_(False)
        print("  [Phase 1] Done.")
        return reward_model

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, policy, value_net, reward_model,
                          env, init_state, task_emb, cfg):
        """
        Run one episode and fill a RolloutBuffer with PPO data.
        Per-step reward = reward_model(sliding window of last seq_len obs).
        """
        seq_len         = cfg.data.seq_len
        obs_key_mapping = cfg.data.obs_key_mapping   # internal → env key
        device          = cfg.device

        env.reset()
        obs = env.set_init_state([init_state])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros((1, 7)))

        policy.reset()
        policy.eval()

        buffer      = RolloutBuffer()
        obs_history = {k: [] for k in obs_key_mapping}  # rolling obs for reward/value

        with torch.no_grad():
            for step in range(self.ppo_max_steps):
                # Record raw obs (uint8 for images, float64 for proprio)
                obs_step = {}
                for internal_key, env_key in obs_key_mapping.items():
                    if env_key in obs[0]:
                        obs_step[internal_key] = obs[0][env_key]
                        obs_history[internal_key].append(obs[0][env_key])

                # Build tensor obs for policy inference (uses latent_queue)
                tensor_obs = raw_obs_to_tensor_obs(obs, task_emb.unsqueeze(0), cfg)
                dist   = policy.policy_head(
                    policy.temporal_encode(
                        policy.spatial_encode(
                            policy.preprocess_input(tensor_obs, train_mode=False)
                        )
                    )[:, -1]
                )
                action   = dist.sample()                         # (1, 7)
                log_prob = dist.log_prob(action).item()          # scalar
                action_np = action.squeeze(0).cpu().numpy()      # (7,)

                # Reward model is on CPU; value net is on GPU
                window_data = self._build_window(
                    obs_history, action_np, task_emb, seq_len, device
                )
                with torch.no_grad():
                    reward = reward_model(window_data).item()
                    value  = value_net(window_data).item()

                obs, _, done, _ = env.step(action.cpu().numpy())

                buffer.add(obs_step, action_np, log_prob, reward, value, done[0])

                if done[0]:
                    break

        return buffer

    def _build_window(self, obs_history, last_action, task_emb, seq_len, device):
        """Build a seq_len window tensor dict from rolling obs history."""
        window = {}
        for key, hist in obs_history.items():
            arr = np.array(hist[-seq_len:])           # (min(T, seq_len), ...)
            # Pad at front if we don't have seq_len steps yet
            if len(arr) < seq_len:
                pad_shape = (seq_len - len(arr),) + arr.shape[1:]
                arr = np.concatenate([np.zeros(pad_shape, dtype=arr.dtype), arr], axis=0)
            # Images: uint8 (T, H, W, C) → float32 (1, T, C, H, W)
            if arr.ndim == 4:
                arr = arr.astype(np.float32) / 255.0
                arr = arr.transpose(0, 3, 1, 2)
            else:
                arr = arr.astype(np.float32)
            window[key] = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, T, ...)

        return {
            "obs":      window,
            "actions":  torch.zeros(1, seq_len, 7, device=device),  # unused by encoder
            "task_emb": task_emb.unsqueeze(0).to(device),           # (1, 768)
        }

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self, buffer):
        """Compute GAE advantages and discounted returns."""
        T        = buffer.length()
        rewards  = np.array(buffer.rewards)
        values   = np.array(buffer.values)
        dones    = np.array(buffer.dones, dtype=np.float32)

        # Normalize rewards to zero mean, unit variance so the reward model's
        # uncalibrated scale doesn't cause exploding advantages and value loss.
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            next_val   = 0.0 if dones[t] else (values[t + 1] if t + 1 < T else 0.0)
            delta      = rewards[t] + self.gamma * next_val - values[t]
            gae        = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO mini-batch update
    # ------------------------------------------------------------------

    def _ppo_update(self, buffers, advantages_list, returns_list,
                     task_emb, value_net, value_optimizer):
        """
        Run K inner epochs of PPO updates over all collected rollout data.
        """
        seq_len = self.cfg.data.seq_len
        device  = self.cfg.device

        # Flatten all rollout data into lists of windows
        all_windows   = []   # each: (seq_len, ...) obs window
        all_actions   = []   # (7,)
        all_old_lp    = []   # scalar
        all_advantages= []   # scalar
        all_returns   = []   # scalar

        for buf, advs, rets in zip(buffers, advantages_list, returns_list):
            T = buf.length()
            for t in range(T):
                # Build obs window [t-seq_len+1 : t+1]
                win_obs = {}
                for key in buf.obs:
                    hist = buf.obs[key]
                    start = max(0, t - seq_len + 1)
                    arr = np.array(hist[start:t + 1])
                    if len(arr) < seq_len:
                        pad = (seq_len - len(arr),) + arr.shape[1:]
                        arr = np.concatenate([np.zeros(pad, dtype=arr.dtype), arr], 0)
                    if arr.ndim == 4:  # images
                        arr = arr.astype(np.float32) / 255.0
                        arr = arr.transpose(0, 3, 1, 2)
                    else:
                        arr = arr.astype(np.float32)
                    win_obs[key] = arr  # (seq_len, ...)

                all_windows.append(win_obs)
                all_actions.append(buf.actions[t])
                all_old_lp.append(buf.log_probs[t])
                all_advantages.append(advs[t])
                all_returns.append(rets[t])

        N = len(all_windows)
        all_old_lp    = np.array(all_old_lp,     dtype=np.float32)
        all_advantages= np.array(all_advantages, dtype=np.float32)
        all_returns   = np.array(all_returns,    dtype=np.float32)

        # Normalize advantages
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(N)

            for start in range(0, N, self.ppo_mini_batch):
                idx = indices[start:start + self.ppo_mini_batch]
                if len(idx) == 0:
                    continue

                # Build batch tensors
                obs_batch   = self._collate_windows([all_windows[i] for i in idx], device)
                act_batch   = torch.tensor(
                    np.array([all_actions[i] for i in idx]), dtype=torch.float32
                ).to(device)
                old_lp_b    = torch.tensor(all_old_lp[idx],    device=device)
                adv_b       = torch.tensor(all_advantages[idx], device=device)
                ret_b       = torch.tensor(all_returns[idx],    device=device)
                temb_b      = task_emb.unsqueeze(0).expand(len(idx), -1).to(device)

                data = {
                    "obs":      obs_batch,
                    "actions":  act_batch.unsqueeze(1).expand(-1, seq_len, -1),
                    "task_emb": temb_b,
                }

                # Policy forward pass
                self.policy.train()
                x    = self.policy.spatial_encode(data)
                x    = self.policy.temporal_encode(x)
                dist = self.policy.policy_head(x[:, -1])   # dist at last step

                new_log_probs = dist.log_prob(act_batch)   # (B,)
                # GMM doesn't implement analytic entropy; use sample-based approx
                try:
                    entropy = dist.entropy()
                    if entropy.dim() > 0:
                        entropy = entropy.mean()
                except NotImplementedError:
                    with torch.no_grad():
                        sampled = dist.sample()            # (B, action_dim)
                    entropy = -dist.log_prob(sampled).mean()

                # PPO clipped surrogate loss
                ratio  = torch.exp(new_log_probs - old_lp_b)
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_net.train()
                values_pred = value_net(data)               # (B,)
                value_loss  = F.mse_loss(values_pred, ret_b)

                # Total loss
                loss = (policy_loss
                        + self.value_coef  * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),    self.cfg.train.grad_clip)
                nn.utils.clip_grad_norm_(value_net.parameters(),      self.cfg.train.grad_clip)
                self.optimizer.step()
                value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()

        return total_policy_loss, total_value_loss

    def _collate_windows(self, windows, device):
        """Stack list of obs window dicts into a batched dict."""
        keys   = windows[0].keys()
        batch  = {}
        for key in keys:
            arr = np.stack([w[key] for w in windows], axis=0)   # (B, T, ...)
            batch[key] = torch.from_numpy(arr).to(device)
        return batch

    # ------------------------------------------------------------------
    # Main task training loop
    # ------------------------------------------------------------------

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        """Two-phase PPO training: reward model → PPO fine-tuning."""
        self.start_task(task_id)

        # Load BC checkpoint
        bc_dir    = self.cfg.lifelong.bc_checkpoint_dir
        ckpt_path = os.path.join(bc_dir, f"task{task_id}_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  [PPO skip] Checkpoint not found: {ckpt_path}")
            return 0.0, 0.0

        bc_state_dict, _, _ = torch_load_model(ckpt_path)
        self.policy.load_state_dict(bc_state_dict)
        self.policy = safe_device(self.policy, self.cfg.device)

        # Preference data
        pref_dir  = self.cfg.lifelong.preference_data_dir
        hdf5_path = os.path.join(
            pref_dir, self.cfg.benchmark_name, f"task{task_id}_preferences.hdf5"
        )
        if not os.path.exists(hdf5_path):
            print(f"  [PPO skip] Preference data not found: {hdf5_path}")
            return 0.0, 0.0

        # ---- Phase 1: Train reward model ----
        reward_model = self._train_reward_model(
            task_id, benchmark, hdf5_path, bc_state_dict
        )
        torch.cuda.empty_cache()

        # ---- Phase 2: PPO fine-tuning ----
        print(f"\n  [Phase 2] PPO fine-tuning for task {task_id}...")

        # Value network — initialize encoder from BC checkpoint
        value_net = ValueNetwork(self.cfg, self.cfg.shape_meta)
        value_net.load_encoder_from_bc(bc_state_dict)
        value_net = safe_device(value_net, self.cfg.device)
        value_optimizer = torch.optim.AdamW(
            value_net.parameters(), lr=self.cfg.train.optimizer.kwargs.lr, weight_decay=1e-4
        )

        # Setup simulation environment
        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)
        init_states_path = os.path.join(
            self.cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)

        env_args = {
            "bddl_file_name": os.path.join(
                self.cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": self.cfg.data.img_h,
            "camera_widths":  self.cfg.data.img_w,
        }
        env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )

        prev_success_rate = -1.0
        idx_at_best_succ  = 0
        cumulated_counter = 0.0
        successes = []
        ppo_losses = []
        n_init = init_states.shape[0]

        for ppo_iter in range(self.ppo_iters):
            t0 = time.time()

            # Collect rollouts
            buffers        = []
            advantages_list = []
            returns_list    = []

            for r in range(self.n_rollouts_per_iter):
                init_idx   = (ppo_iter * self.n_rollouts_per_iter + r) % n_init
                init_state = np.array(init_states[init_idx])

                buf = self._collect_rollout(
                    self.policy, value_net, reward_model,
                    env, init_state, task_emb, self.cfg
                )
                advs, rets = self._compute_gae(buf)
                buffers.append(buf)
                advantages_list.append(advs)
                returns_list.append(rets)

            # PPO update
            pol_loss, val_loss = self._ppo_update(
                buffers, advantages_list, returns_list,
                task_emb, value_net, value_optimizer
            )

            t1 = time.time()
            mean_reward = np.mean([np.mean(b.rewards) for b in buffers])
            print(
                f"[info] PPO iter {ppo_iter+1:3d}/{self.ppo_iters} | "
                f"policy loss: {pol_loss:.4f} | value loss: {val_loss:.4f} | "
                f"mean reward: {mean_reward:.4f} | time: {(t1-t0)/60:.2f}m"
            )

            # Periodic evaluation
            eval_every = getattr(self.cfg.eval, "eval_every", 5)
            if (ppo_iter + 1) % eval_every == 0 or ppo_iter == 0:
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg, algo=self, task=task,
                    task_emb=task_emb, task_id=task_id,
                    sim_states=None, task_str="",
                )
                successes.append(success_rate)
                ppo_losses.append(pol_loss)
                cumulated_counter += 1.0

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ  = len(successes) - 1

                print(
                    f"[info] PPO iter {ppo_iter+1:3d} | succ: {success_rate:.2f} | "
                    f"best succ: {prev_success_rate:.2f}"
                )

        env.close()
        del reward_model, value_net
        torch.cuda.empty_cache()
        gc.collect()

        successes  = np.array(successes)
        ppo_losses = np.array(ppo_losses)
        successes[idx_at_best_succ:]  = successes[idx_at_best_succ]
        ppo_losses[idx_at_best_succ:] = ppo_losses[idx_at_best_succ]

        return (
            successes.sum()  / max(cumulated_counter, 1),
            ppo_losses.sum() / max(cumulated_counter, 1),
        )
