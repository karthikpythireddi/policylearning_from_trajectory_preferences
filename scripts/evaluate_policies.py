#!/usr/bin/env python3
"""
evaluate_policies.py

Evaluates BC, DPO, RLHF, and PPO checkpoints on selected tasks and
prints a comparison table of success rates.

Optionally records one video rollout per (task, method) pair.

Usage:
  python scripts/evaluate_policies.py \
    --cfg outputs/2026-02-23/08-37-38/.hydra/config.yaml \
    --task_ids 0 3 4 7 \
    --n_eval 20 \
    --device cuda \
    --bc_dir   experiments/libero_10/SingleTask/BCTransformerPolicy_seed10000/run_003 \
    --dpo_dir  experiments/libero_10/DPO/BCTransformerPolicy_seed10000/run_001 \
    --rlhf_dir experiments/libero_10/RLHF/BCTransformerPolicy_seed10000/run_001 \
    --ppo_dir  experiments/libero_10/PPO/BCTransformerPolicy_seed10000/run_001 \
    --save_video \
    --video_dir videos
"""

import argparse
import os
import sys

import imageio
import numpy as np
import torch
import yaml
from easydict import EasyDict
from transformers import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robomimic.utils.obs_utils as ObsUtils
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.datasets import get_dataset
from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.models import get_policy_class
from libero.lifelong.utils import get_task_embs, safe_device, torch_load_model


def evaluate_task(policy, task, task_emb, cfg, n_eval=20):
    """
    Evaluate a single policy on a single task.
    Returns success rate (float in [0, 1]).
    """
    policy.eval()

    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths":  cfg.data.img_w,
    }

    # Use DummyVectorEnv (no fork) to avoid EGL_BAD_ALLOC in subprocess + GPU context
    env_num      = min(cfg.eval.num_procs, n_eval)
    eval_loop_num = (n_eval + env_num - 1) // env_num

    env_fns = [lambda: OffScreenRenderEnv(**env_args)] * env_num
    env = DummyVectorEnv(env_fns)

    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)

    successes = []

    for i in range(eval_loop_num):
        env.reset()
        indices = np.arange(i * env_num, min((i + 1) * env_num, n_eval))

        init_batch = [np.array(init_states[j % init_states.shape[0]]) for j in indices]
        obs = env.set_init_state(init_batch)

        # Stabilize
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros((len(indices), 7)))

        policy.reset()
        done_flags = [False] * len(indices)
        success_flags = [False] * len(indices)

        with torch.no_grad():
            for _ in range(cfg.eval.max_steps):
                data = raw_obs_to_tensor_obs(obs, task_emb.unsqueeze(0), cfg)
                action = policy.get_action(data)
                obs, _, done, _ = env.step(action)

                for k, (d, s) in enumerate(zip(done, success_flags)):
                    if d and not done_flags[k]:
                        success_flags[k] = True
                    if d:
                        done_flags[k] = True

                if all(done_flags):
                    break

        successes.extend(success_flags)

    env.close()
    return np.mean(successes)


def record_video(policy, task, task_emb, cfg, video_path):
    """
    Record a single rollout as an MP4 video.

    Runs one episode (first init state) and saves an agentview video.
    Frames are stacked side-by-side: agentview (left) | eye-in-hand (right).
    A thin colored border (green=success, red=failure) is added at the end.
    """
    policy.eval()

    env_args = {
        "bddl_file_name": os.path.join(
            cfg.bddl_folder, task.problem_folder, task.bddl_file
        ),
        "camera_heights": cfg.data.img_h,
        "camera_widths":  cfg.data.img_w,
    }
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])

    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)
    init_state  = np.array(init_states[0])

    env.reset()
    obs = env.set_init_state([init_state])

    # Stabilize
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros((1, 7)))

    policy.reset()
    frames  = []
    success = False

    with torch.no_grad():
        for _ in range(cfg.eval.max_steps):
            # Collect agentview and eye-in-hand frames from env obs
            agentview    = obs[0].get("agentview_image",          None)
            eye_in_hand  = obs[0].get("robot0_eye_in_hand_image", None)

            if agentview is not None:
                # MuJoCo renders upside-down — flip vertically
                av = agentview[::-1].copy()
                if eye_in_hand is not None:
                    eih = eye_in_hand[::-1].copy()
                    # Stack side-by-side
                    frame = np.concatenate([av, eih], axis=1)
                else:
                    frame = av
                frames.append(frame)

            data   = raw_obs_to_tensor_obs(obs, task_emb.unsqueeze(0), cfg)
            action = policy.get_action(data)
            obs, _, done, _ = env.step(action)

            if done[0]:
                success = True
                break

    env.close()

    if not frames:
        return

    # Add outcome flash frames at the end (green=success, red=failure)
    h, w = frames[-1].shape[:2]
    border_color = (0, 200, 0) if success else (200, 0, 0)
    flash = np.zeros((h, w, 3), dtype=np.uint8)
    b = 8  # border thickness
    flash[:b, :] = border_color
    flash[-b:, :] = border_color
    flash[:, :b] = border_color
    flash[:, -b:] = border_color
    # Blend last frame with the border overlay
    last = frames[-1].copy()
    mask = (flash > 0).any(axis=-1, keepdims=True)
    blended = np.where(mask, (last * 0.3 + flash * 0.7).astype(np.uint8), last)
    for _ in range(15):   # ~0.5 s at 30 fps
        frames.append(blended)

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=30)
    for f in frames:
        writer.append_data(f)
    writer.close()

    status = "SUCCESS" if success else "failure"
    print(f"    video saved → {video_path}  [{status}]")


def load_policy(cfg, shape_meta, ckpt_path, device):
    """Load a policy from a checkpoint file."""
    policy = get_policy_class(cfg.policy.policy_type)(cfg, shape_meta)
    state_dict, _, _ = torch_load_model(ckpt_path)
    policy.load_state_dict(state_dict)
    policy = safe_device(policy, device)
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare BC, DPO, RLHF, PPO policies."
    )
    parser.add_argument("--cfg",       required=True,  help="Hydra config yaml")
    parser.add_argument("--task_ids",  type=int, nargs="+", default=[0, 3, 4, 7])
    parser.add_argument("--n_eval",    type=int, default=20, help="Rollouts per task")
    parser.add_argument("--device",    default="cuda")
    parser.add_argument("--save_video", action="store_true",
                        help="Record one video rollout per (task, method) pair")
    parser.add_argument("--video_dir",  default="videos",
                        help="Root directory for saved videos")

    # Checkpoint directories for each method
    parser.add_argument("--bc_dir",   default=None, help="BC checkpoint dir")
    parser.add_argument("--dpo_dir",  default=None, help="DPO checkpoint dir")
    parser.add_argument("--rlhf_dir", default=None, help="RLHF checkpoint dir")
    parser.add_argument("--ppo_dir",  default=None, help="PPO checkpoint dir")

    args = parser.parse_args()

    # Load config
    with open(args.cfg) as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.device             = args.device
    cfg.bddl_folder        = cfg.bddl_folder        or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    cfg.folder             = cfg.folder             or get_libero_path("datasets")

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    logging.set_verbosity_error()

    benchmark    = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs    = get_task_embs(cfg, descriptions)

    first_demo = os.path.join(cfg.folder, benchmark.get_task_demonstration(0))
    _, shape_meta = get_dataset(
        dataset_path=first_demo,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=False,
    )
    cfg.shape_meta = shape_meta

    # Which methods to evaluate
    methods = {}
    if args.bc_dir:   methods["BC"]   = args.bc_dir
    if args.dpo_dir:  methods["DPO"]  = args.dpo_dir
    if args.rlhf_dir: methods["RLHF"] = args.rlhf_dir
    if args.ppo_dir:  methods["PPO"]  = args.ppo_dir

    if not methods:
        print("[error] Provide at least one of --bc_dir, --dpo_dir, --rlhf_dir, --ppo_dir")
        return

    # Results table: results[task_id][method] = success_rate
    results    = {tid: {} for tid in args.task_ids}
    task_names = {}

    for task_id in args.task_ids:
        task      = benchmark.get_task(task_id)
        task_emb  = task_embs[task_id]
        task_names[task_id] = task.language

        # Shorten task name for directory: take first 4 words
        short_name = "_".join(task.language.split()[:4])

        print(f"\n{'='*60}")
        print(f"[Task {task_id}] {task.language}")
        print(f"{'='*60}")

        for method, ckpt_dir in methods.items():
            ckpt_path = os.path.join(ckpt_dir, f"task{task_id}_model.pth")

            if not os.path.exists(ckpt_path):
                print(f"  [{method}] checkpoint not found — skipping")
                results[task_id][method] = None
                continue

            print(f"  [{method}] evaluating with {args.n_eval} rollouts...", end="", flush=True)
            policy       = load_policy(cfg, shape_meta, ckpt_path, args.device)
            success_rate = evaluate_task(policy, task, task_emb, cfg, n_eval=args.n_eval)
            results[task_id][method] = success_rate
            print(f" {success_rate*100:.1f}%")

            if args.save_video:
                video_path = os.path.join(
                    args.video_dir,
                    f"task{task_id}_{short_name}",
                    f"{method}.mp4",
                )
                record_video(policy, task, task_emb, cfg, video_path)

            del policy
            torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS: Success Rate Comparison")
    print("="*70)

    method_names = list(methods.keys())
    col_w = 10

    # Header
    header = f"{'Task':<5} {'Description':<35}"
    for m in method_names:
        header += f"{m:>{col_w}}"
    print(header)
    print("-" * (40 + col_w * len(method_names)))

    # Rows
    avgs = {m: [] for m in method_names}
    for task_id in args.task_ids:
        name = task_names[task_id]
        name = name[:33] + ".." if len(name) > 35 else name
        row  = f"{task_id:<5} {name:<35}"
        for m in method_names:
            val = results[task_id].get(m)
            if val is None:
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{val*100:>{col_w-1}.1f}%"
                avgs[m].append(val)
        print(row)

    # Average row
    print("-" * (40 + col_w * len(method_names)))
    avg_row = f"{'AVG':<5} {'':<35}"
    for m in method_names:
        if avgs[m]:
            avg_row += f"{np.mean(avgs[m])*100:>{col_w-1}.1f}%"
        else:
            avg_row += f"{'N/A':>{col_w}}"
    print(avg_row)
    print("="*70)

    if args.save_video:
        print(f"\nVideos saved to: {os.path.abspath(args.video_dir)}/")
        print("Structure: videos/task{id}_{name}/{method}.mp4")


if __name__ == "__main__":
    main()
