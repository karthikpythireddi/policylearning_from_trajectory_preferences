#!/usr/bin/env python3
"""
collect_preferences.py

Generates genuine preference trajectory pairs for DPO/RLHF training on LIBERO.

Strategy:
  Two rollouts are generated from the SAME policy for each init state.
  Preferences are determined by comparing outcomes:

  1. success vs failure  → success is winner  (strong preference signal)
  2. both succeed        → shorter is winner   (efficiency preference)
  3. both fail           → skip               (no reliable signal)

This is a genuine preference setup: both trajectories come from the same
policy distribution, so log pi_ref(tau_w) and log pi_ref(tau_l) are both
well-defined — unlike the approach of using human demos as winners.

Usage:
  python scripts/collect_preferences.py \\
    --cfg outputs/2026-02-23/08-37-38/.hydra/config.yaml \\
    --checkpoint_dir experiments/libero_10/SingleTask/BCTransformerPolicy_seed10000/run_003 \\
    --n_pairs 20 \\
    --output_dir preference_data

Output structure:
  preference_data/libero_10/task{i}_preferences.hdf5
    /metadata
      n_pairs, task_name, task_emb, preference_type counts
    /pair_{i}/
      winner/actions (T_w, 7)
      winner/obs/agentview_rgb     (T_w, H, W, 3)  uint8
      winner/obs/eye_in_hand_rgb   (T_w, H, W, 3)  uint8
      winner/obs/joint_states      (T_w, 7)
      winner/obs/gripper_states    (T_w, 2)
      loser/actions  (T_l, 7)
      loser/obs/...
      attrs: preference_type ("success_vs_failure" or "efficiency")
"""

import argparse
import gc
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from easydict import EasyDict
from transformers import logging

# Make sure LIBERO root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robomimic.utils.obs_utils as ObsUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv
from libero.lifelong.datasets import get_dataset
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.models import get_policy_class
from libero.lifelong.utils import get_task_embs, safe_device, torch_load_model


# ---------------------------------------------------------------------------
# Policy rollout
# ---------------------------------------------------------------------------

def rollout_policy(policy, env, init_state, task_emb, cfg, max_steps=300):
    """
    Roll out policy from a given init state.
    Records obs with internal key names. Images stored as uint8 (H, W, C).
    Returns a trajectory dict with obs, actions, success flag, and length.
    """
    env.reset()
    obs = env.set_init_state([init_state])

    # Stabilize physics with dummy actions
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros((1, 7)))

    policy.reset()
    policy.eval()

    obs_key_mapping = cfg.data.obs_key_mapping  # internal_key -> env_key

    traj = {
        "actions": [],
        "obs": {k: [] for k in obs_key_mapping.keys()},
        "success": False,
        "length": 0,
    }

    with torch.no_grad():
        for _ in range(max_steps):
            for internal_key, env_key in obs_key_mapping.items():
                if env_key in obs[0]:
                    traj["obs"][internal_key].append(obs[0][env_key])

            data = raw_obs_to_tensor_obs(obs, task_emb.unsqueeze(0), cfg)
            action = policy.get_action(data)  # (1, 7)
            traj["actions"].append(action[0])

            obs, _, done, _ = env.step(action)

            if done[0]:
                traj["success"] = True
                break

    traj["length"] = len(traj["actions"])
    traj["actions"] = np.array(traj["actions"])  # (T, 7)
    for k in traj["obs"]:
        traj["obs"][k] = np.array(traj["obs"][k]) if traj["obs"][k] else np.array([])

    return traj


# ---------------------------------------------------------------------------
# Preference collection for one task
# ---------------------------------------------------------------------------

def collect_preferences_for_task(
    policy, task, task_emb, cfg,
    n_pairs=20, max_steps=300, noise_prob=0.0
):
    """
    Collect up to n_pairs of (winner, loser) preference pairs for a single task.

    Two rollouts are generated from the policy for each init state.
    Preference rules:
      - success vs failure  → success wins   (label: "success_vs_failure")
      - both succeed        → shorter wins    (label: "efficiency")
      - both fail           → skip            (no reliable signal)

    noise_prob: probability of flipping the label (simulates noisy feedback)
    """
    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path)  # (N, state_dim)

    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths":  cfg.data.img_w,
    }
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])

    pairs = []
    counts = {"success_vs_failure": 0, "efficiency": 0, "skipped_both_fail": 0}
    n_init_states = init_states.shape[0]

    for i in range(n_init_states):
        if len(pairs) >= n_pairs:
            break

        init_state = np.array(init_states[i])

        # Generate two independent rollouts from the same init state
        traj_a = rollout_policy(policy, env, init_state, task_emb, cfg, max_steps)
        traj_b = rollout_policy(policy, env, init_state, task_emb, cfg, max_steps)

        # Determine winner and loser
        if traj_a["success"] and not traj_b["success"]:
            winner, loser = traj_a, traj_b
            pref_type = "success_vs_failure"

        elif traj_b["success"] and not traj_a["success"]:
            winner, loser = traj_b, traj_a
            pref_type = "success_vs_failure"

        elif traj_a["success"] and traj_b["success"]:
            # Both succeed — prefer the more efficient (shorter) trajectory
            if traj_a["length"] <= traj_b["length"]:
                winner, loser = traj_a, traj_b
            else:
                winner, loser = traj_b, traj_a
            pref_type = "efficiency"

        else:
            # Both fail — skip, no reliable preference signal
            counts["skipped_both_fail"] += 1
            continue

        counts[pref_type] += 1

        # Optionally inject label noise
        if noise_prob > 0 and np.random.random() < noise_prob:
            pairs.append((loser, winner, pref_type))   # noisy: wrong label
        else:
            pairs.append((winner, loser, pref_type))   # correct label

    env.close()
    gc.collect()

    print(
        f"  Init states used: {min(i+1, n_init_states)} | "
        f"Pairs: {len(pairs)} "
        f"(success_vs_failure: {counts['success_vs_failure']}, "
        f"efficiency: {counts['efficiency']}, "
        f"skipped_both_fail: {counts['skipped_both_fail']})"
    )
    return pairs


# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------

def save_pairs_to_hdf5(pairs, output_path, task_emb, task_name):
    """Save preference pairs to an HDF5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_success_vs_failure = sum(1 for _, _, t in pairs if t == "success_vs_failure")
    n_efficiency = sum(1 for _, _, t in pairs if t == "efficiency")

    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_pairs"]              = len(pairs)
        meta.attrs["task_name"]            = task_name
        meta.attrs["n_success_vs_failure"] = n_success_vs_failure
        meta.attrs["n_efficiency"]         = n_efficiency
        meta.create_dataset("task_emb", data=task_emb.cpu().numpy())  # (768,)

        for i, (winner, loser, pref_type) in enumerate(pairs):
            grp = f.create_group(f"pair_{i}")
            grp.attrs["winner_length"]   = winner["length"]
            grp.attrs["loser_length"]    = loser["length"]
            grp.attrs["winner_success"]  = winner["success"]
            grp.attrs["loser_success"]   = loser["success"]
            grp.attrs["preference_type"] = pref_type

            for label, traj in [("winner", winner), ("loser", loser)]:
                t_grp = grp.create_group(label)
                t_grp.create_dataset("actions", data=traj["actions"])
                obs_grp = t_grp.create_group("obs")
                for obs_key, obs_val in traj["obs"].items():
                    if obs_val is not None and len(obs_val) > 0:
                        obs_grp.create_dataset(obs_key, data=obs_val)

    print(
        f"  Saved {len(pairs)} pairs → {output_path} "
        f"[success_vs_failure: {n_success_vs_failure}, efficiency: {n_efficiency}]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect genuine preference trajectory pairs for DPO/RLHF training."
    )
    parser.add_argument(
        "--cfg", required=True,
        help="Path to Hydra config yaml (e.g. outputs/DATE/TIME/.hydra/config.yaml)"
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Directory containing task{i}_model.pth checkpoint files"
    )
    parser.add_argument("--benchmark",   default="libero_10")
    parser.add_argument("--output_dir",  default="preference_data")
    parser.add_argument("--n_pairs",     type=int,   default=20,
                        help="Number of preference pairs to collect per task")
    parser.add_argument("--max_steps",   type=int,   default=300,
                        help="Max rollout steps per episode")
    parser.add_argument("--noise_prob",  type=float, default=0.0,
                        help="Probability of flipping preference label")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--task_ids",    type=int, nargs="+", default=None,
                        help="Specific task IDs to process (default: all)")
    args = parser.parse_args()

    # Load Hydra config
    with open(args.cfg) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.device = args.device
    cfg.bddl_folder        = cfg.bddl_folder        or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    cfg.folder             = cfg.folder             or get_libero_path("datasets")

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    logging.set_verbosity_error()

    benchmark    = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs    = get_task_embs(cfg, descriptions)

    # Get shape_meta from first available dataset
    first_demo = os.path.join(cfg.folder, benchmark.get_task_demonstration(0))
    _, shape_meta = get_dataset(
        dataset_path=first_demo,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=False,
    )
    cfg.shape_meta = shape_meta

    task_ids = args.task_ids if args.task_ids else list(range(benchmark.n_tasks))

    for task_id in task_ids:
        task     = benchmark.get_task(task_id)
        task_emb = task_embs[task_id]
        ckpt_path = os.path.join(args.checkpoint_dir, f"task{task_id}_model.pth")

        print(f"\n{'='*60}")
        print(f"[Task {task_id}] {task.language}")
        print(f"{'='*60}")

        if not os.path.exists(ckpt_path):
            print(f"  [skip] Checkpoint not found: {ckpt_path}")
            continue

        # Load policy
        policy = get_policy_class(cfg.policy.policy_type)(cfg, shape_meta)
        state_dict, _, _ = torch_load_model(ckpt_path)
        policy.load_state_dict(state_dict)
        policy = safe_device(policy, cfg.device)
        policy.eval()

        # Collect preference pairs
        pairs = collect_preferences_for_task(
            policy, task, task_emb, cfg,
            n_pairs=args.n_pairs,
            max_steps=args.max_steps,
            noise_prob=args.noise_prob,
        )

        if len(pairs) == 0:
            print(f"  [warn] No pairs collected for task {task_id} — "
                  f"all rollouts had same outcome (both fail or both succeed with equal length)")
        else:
            output_path = os.path.join(
                args.output_dir, cfg.benchmark_name,
                f"task{task_id}_preferences.hdf5"
            )
            save_pairs_to_hdf5(pairs, output_path, task_emb, task.language)

        del policy
        torch.cuda.empty_cache()
        gc.collect()

    print("\n[done] Preference data collection complete!")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
