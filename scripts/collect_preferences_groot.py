#!/usr/bin/env python3
"""
collect_preferences_groot.py

Generates genuine preference trajectory pairs for DPO/RWR/RLHF training
using GR00T N1.6 as the base policy on RoboCasa GR1 tabletop tasks.

Strategy:
  Two rollouts are generated from the SAME policy for each init seed.
  Preferences are determined by comparing outcomes:

  1. success vs failure  -> success is winner  (strong preference signal)
  2. both succeed        -> shorter is winner   (efficiency preference)
  3. both fail           -> skip               (no reliable signal)

Prerequisites:
  1. Start the GR00T inference server WITH --use-sim-policy-wrapper:
       conda run -n groot python gr00t/eval/run_gr00t_server.py \\
           --model-path nvidia/GR00T-N1.6-3B \\
           --embodiment-tag gr1 \\
           --use-sim-policy-wrapper \\
           --denoising-steps 4 \\
           --port 5555

  2. Install RoboCasa GR1 sim env in gr1_sim conda env:
       conda activate gr1_sim
       pip install -e external_dependencies/robocasa-gr1-tabletop-tasks

Usage:
  conda activate gr1_sim
  python scripts/collect_preferences_groot.py \\
    --env_name "gr1_unified/PnPCounterToCab_GR1ArmsAndWaistFourierHands_Env" \\
    --n_pairs 20 \\
    --output_dir preference_data/gr1

Output structure:
  preference_data/gr1/{task_name}_preferences.hdf5
    /metadata
      n_pairs, task_name, preference_type counts
    /pair_{i}/
      winner/actions/{action_key} (T_w, D)
      winner/obs/{obs_key}        (T_w, ...)
      loser/actions/{action_key}  (T_l, D)
      loser/obs/{obs_key}         (T_l, ...)
      attrs: preference_type ("success_vs_failure" or "efficiency")
"""

import argparse
import gc
import os

import gymnasium as gym
import h5py
import numpy as np

# Set rendering backend before importing robocasa
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401 — registers gr1_unified/* envs
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy.server_client import PolicyClient


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

N_ACTION_STEPS = 8   # GR00T standard chunk size


def make_env_fn(env_name: str, max_episode_steps: int):
    """Return a factory function that builds a single wrapped env."""
    def _make():
        env = gym.make(env_name, enable_render=True)
        env = MultiStepWrapper(
            env,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=N_ACTION_STEPS,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        )
        return env
    return _make


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def rollout(vec_env, policy: PolicyClient, seed: int,
            action_noise_scale: float = 0.0) -> dict:
    """
    Roll out the GR00T policy from a given seed.

    vec_env:            gym.vector.SyncVectorEnv wrapping one MultiStepWrapper env
    policy:             PolicyClient connected to GR00T server (--use-sim-policy-wrapper)
    action_noise_scale: std of Gaussian noise added to actions (0 = clean rollout)

    Returns dict:
      obs      - {obs_key: list of np.ndarray per step}
      actions  - {action_key: list of np.ndarray per step}
      success  - bool
      length   - int (number of action chunks executed)
    """
    obs, _ = vec_env.reset(seed=[seed])
    policy.reset()

    # Collect obs/action keys on first call
    obs_keys = [k for k in obs.keys() if not k.startswith("annotation")]

    traj_obs = {k: [] for k in obs_keys}
    traj_act = {}
    success = False
    length = 0
    cumulative_reward = 0.0

    while True:
        # Record obs (squeeze B=1 dim, then T=1 dim from MultiStepWrapper)
        for k in obs_keys:
            # obs[k] shape: (B=1, T=1, ...) for video/state
            traj_obs[k].append(obs[k][0, 0])

        # Query policy — obs has B=1 batch dim as expected by Gr00tSimPolicyWrapper
        actions, _info = policy.get_action(obs)
        # actions: {action.key: (B=1, T=n_action_steps, D)}

        # Record all action steps for each key — shape per entry: (T_chunk, D)
        # actions[ak] shape: (B=1, T_chunk, D); squeeze B dim only
        if not traj_act:
            traj_act = {ak: [] for ak in actions}
        for ak in actions:
            traj_act[ak].append(actions[ak][0])  # (T_chunk, D)

        # Optionally corrupt actions with Gaussian noise before stepping
        if action_noise_scale > 0:
            noisy_actions = {}
            for ak, av in actions.items():
                noisy_actions[ak] = av + np.random.randn(*av.shape).astype(av.dtype) * action_noise_scale
        else:
            noisy_actions = actions

        obs, chunk_reward, done, _truncated, infos = vec_env.step(noisy_actions)
        length += 1
        # chunk_reward: scalar (max reward over n_action_steps from MultiStepWrapper)
        # SyncVectorEnv adds B dim: shape (1,) or scalar
        cumulative_reward += float(np.asarray(chunk_reward).flat[0])

        # Success detection — SyncVectorEnv may wrap info["success"] as an
        # object-dtype array (since MultiStepWrapper returns it as an ndarray
        # of shape (n_action_steps,), not a scalar).
        ep_success = False
        if "success" in infos:
            s = np.asarray(infos["success"])
            if s.dtype == object:
                ep_success = any(bool(np.any(v)) for v in s.flat)
            else:
                ep_success = bool(s.any())
        elif "final_info" in infos and infos["final_info"] is not None:
            fi = infos["final_info"]
            if isinstance(fi, (list, tuple)) and len(fi) > 0 and fi[0] is not None:
                ep_success = bool(fi[0].get("success", False))

        if ep_success:
            success = True
            break
        if done:
            break

    print(
        f"    rollout seed={seed}: success={success}, "
        f"chunks={length}, reward={cumulative_reward:.3f}"
    )

    # Stack lists -> arrays
    for k in traj_obs:
        traj_obs[k] = np.array(traj_obs[k]) if traj_obs[k] else np.array([])
    for ak in traj_act:
        traj_act[ak] = np.array(traj_act[ak]) if traj_act[ak] else np.array([])

    return {
        "obs": traj_obs,
        "actions": traj_act,
        "success": success,
        "length": length,
        "cumulative_reward": cumulative_reward,
    }


# ---------------------------------------------------------------------------
# Preference collection for one task
# ---------------------------------------------------------------------------

def collect_preferences(
    vec_env,
    policy: PolicyClient,
    n_pairs: int = 20,
    seed_offset: int = 0,
    noise_prob: float = 0.0,
    reward_fallback: bool = True,
    min_reward_gap: float = 0.01,
    noise_injection: bool = False,
    noise_scale: float = 0.05,
) -> list:
    """
    Collect up to n_pairs of (winner, loser, pref_type) tuples.

    Preference labeling priority:
      1. success vs failure  -> success wins  ("success_vs_failure")
      2. both succeed        -> shorter wins  ("efficiency")
      3. both fail + reward_fallback=True -> higher cumulative reward wins
         ("reward_comparison"), provided the gap > min_reward_gap
      4. both fail + noise_injection=True -> clean rollout beats noisy rollout
         ("noise_injection")  — always produces a pair, useful for sparse-reward envs
      5. both fail + no signal -> skip

    reward_fallback: use cumulative sim reward when both fail (default True)
    min_reward_gap: minimum |R_a - R_b| to create a reward_comparison pair
    noise_injection: fallback to clean-vs-noisy pairs when both rollouts fail
    noise_scale:    std of Gaussian noise added to the loser's actions
    """
    pairs = []
    counts = {
        "success_vs_failure": 0, "efficiency": 0,
        "reward_comparison": 0, "noise_injection": 0, "skipped": 0,
    }

    seed = seed_offset
    while len(pairs) < n_pairs:
        traj_a = rollout(vec_env, policy, seed=seed)
        traj_b = rollout(vec_env, policy, seed=seed + 1)
        seed += 2

        # Determine winner / loser
        if traj_a["success"] and not traj_b["success"]:
            winner, loser = traj_a, traj_b
            pref_type = "success_vs_failure"

        elif traj_b["success"] and not traj_a["success"]:
            winner, loser = traj_b, traj_a
            pref_type = "success_vs_failure"

        elif traj_a["success"] and traj_b["success"]:
            if traj_a["length"] <= traj_b["length"]:
                winner, loser = traj_a, traj_b
            else:
                winner, loser = traj_b, traj_a
            pref_type = "efficiency"

        elif reward_fallback:
            # Both fail — use cumulative simulation reward as proxy signal
            ra = traj_a["cumulative_reward"]
            rb = traj_b["cumulative_reward"]
            if abs(ra - rb) >= min_reward_gap:
                if ra >= rb:
                    winner, loser = traj_a, traj_b
                else:
                    winner, loser = traj_b, traj_a
                pref_type = "reward_comparison"
            elif noise_injection:
                # Sparse reward env: re-run seed with action noise as the loser
                print(f"    [noise_injection] re-running seed={seed-2} with noise_scale={noise_scale}")
                traj_noisy = rollout(vec_env, policy, seed=seed - 2,
                                     action_noise_scale=noise_scale)
                winner, loser = traj_a, traj_noisy
                pref_type = "noise_injection"
            else:
                counts["skipped"] += 1
                continue

        elif noise_injection:
            # Both fail, no reward fallback — use clean vs noisy
            print(f"    [noise_injection] re-running seed={seed-2} with noise_scale={noise_scale}")
            traj_noisy = rollout(vec_env, policy, seed=seed - 2,
                                 action_noise_scale=noise_scale)
            winner, loser = traj_a, traj_noisy
            pref_type = "noise_injection"

        else:
            counts["skipped"] += 1
            # Avoid infinite loops on tasks where policy always fails
            if seed - seed_offset > 10 * n_pairs:
                print("  [warn] Too many failures — stopping early.")
                break
            continue

        counts[pref_type] += 1

        # Optionally inject label noise
        if noise_prob > 0 and np.random.random() < noise_prob:
            pairs.append((loser, winner, pref_type))
        else:
            pairs.append((winner, loser, pref_type))

        extra = (
            f"w_rew={winner['cumulative_reward']:.3f} l_rew={loser['cumulative_reward']:.3f}"
            if pref_type == "reward_comparison"
            else f"w_len={winner['length']} l_len={loser['length']}"
        )
        print(
            f"  pair {len(pairs)}/{n_pairs} | "
            f"seed={seed-2}/{seed-1} | "
            f"type={pref_type} | {extra}"
        )

    print(
        f"  Total: {len(pairs)} pairs "
        f"(success_vs_failure: {counts['success_vs_failure']}, "
        f"efficiency: {counts['efficiency']}, "
        f"reward_comparison: {counts['reward_comparison']}, "
        f"noise_injection: {counts['noise_injection']}, "
        f"skipped: {counts['skipped']})"
    )
    return pairs


# ---------------------------------------------------------------------------
# Save to HDF5
# ---------------------------------------------------------------------------

def save_pairs_to_hdf5(pairs: list, output_path: str, task_name: str):
    """Save preference pairs to an HDF5 file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    n_svf = sum(1 for _, _, t in pairs if t == "success_vs_failure")
    n_eff = sum(1 for _, _, t in pairs if t == "efficiency")
    n_rew = sum(1 for _, _, t in pairs if t == "reward_comparison")
    n_noi = sum(1 for _, _, t in pairs if t == "noise_injection")

    with h5py.File(output_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["n_pairs"]              = len(pairs)
        meta.attrs["task_name"]            = task_name
        meta.attrs["n_success_vs_failure"] = n_svf
        meta.attrs["n_efficiency"]         = n_eff
        meta.attrs["n_reward_comparison"]  = n_rew
        meta.attrs["n_noise_injection"]    = n_noi

        for i, (winner, loser, pref_type) in enumerate(pairs):
            grp = f.create_group(f"pair_{i}")
            grp.attrs["winner_length"]           = winner["length"]
            grp.attrs["loser_length"]            = loser["length"]
            grp.attrs["winner_success"]          = winner["success"]
            grp.attrs["loser_success"]           = loser["success"]
            grp.attrs["winner_cumulative_reward"] = winner["cumulative_reward"]
            grp.attrs["loser_cumulative_reward"]  = loser["cumulative_reward"]
            grp.attrs["preference_type"]         = pref_type

            for label, traj in [("winner", winner), ("loser", loser)]:
                t_grp = grp.create_group(label)

                act_grp = t_grp.create_group("actions")
                for ak, av in traj["actions"].items():
                    if av is not None and len(av) > 0:
                        act_grp.create_dataset(ak, data=av)

                obs_grp = t_grp.create_group("obs")
                for ok, ov in traj["obs"].items():
                    if ov is not None and len(ov) > 0:
                        obs_grp.create_dataset(ok, data=ov)

    print(
        f"  Saved {len(pairs)} pairs -> {output_path} "
        f"[success_vs_failure: {n_svf}, efficiency: {n_eff}, "
        f"reward_comparison: {n_rew}, noise_injection: {n_noi}]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect GR00T preference pairs for DPO/RLHF training."
    )
    parser.add_argument(
        "--env_name", required=True,
        help=(
            "Gym env ID, e.g. "
            "'gr1_unified/PnPCounterToCab_GR1ArmsAndWaistFourierHands_Env'"
        ),
    )
    parser.add_argument("--host",       default="localhost")
    parser.add_argument("--port",       type=int, default=5555)
    parser.add_argument("--output_dir", default="preference_data/gr1")
    parser.add_argument("--n_pairs",    type=int, default=20)
    parser.add_argument("--max_steps",  type=int, default=600,
                        help="Max episode steps (individual sim steps; e.g. 600 = 75 GR00T queries at n_action_steps=8)")
    parser.add_argument("--noise_prob",      type=float, default=0.0)
    parser.add_argument("--seed_offset",     type=int,   default=0)
    parser.add_argument("--min_reward_gap",  type=float, default=0.01,
                        help="Min |R_w - R_l| required for a reward_comparison pair")
    parser.add_argument("--no_reward_fallback", action="store_true",
                        help="Disable reward-based labeling when both rollouts fail")
    parser.add_argument("--noise_injection", action="store_true",
                        help="When both rollouts fail (sparse reward), re-run seed with "
                             "action noise and treat clean rollout as winner")
    parser.add_argument("--noise_scale", type=float, default=0.05,
                        help="Std of Gaussian noise added to actions in noise_injection mode")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Task env : {args.env_name}")
    print(f"Server   : {args.host}:{args.port}")
    print(f"n_pairs  : {args.n_pairs}")
    print(f"{'='*60}")
    print(
        "IMPORTANT: GR00T server must be started with --use-sim-policy-wrapper\n"
        "  conda run -n groot python gr00t/eval/run_gr00t_server.py \\\n"
        "      --model-path nvidia/GR00T-N1.6-3B \\\n"
        "      --embodiment-tag gr1 \\\n"
        "      --use-sim-policy-wrapper \\\n"
        "      --denoising-steps 4 \\\n"
        "      --port 5555\n"
    )

    # Build vectorized env (SyncVectorEnv adds B=1 batch dim)
    vec_env = gym.vector.SyncVectorEnv(
        [make_env_fn(args.env_name, args.max_steps)]
    )

    # Connect to GR00T inference server
    policy = PolicyClient(host=args.host, port=args.port, strict=False)

    # Collect preference pairs
    pairs = collect_preferences(
        vec_env,
        policy,
        n_pairs=args.n_pairs,
        seed_offset=args.seed_offset,
        noise_prob=args.noise_prob,
        reward_fallback=not args.no_reward_fallback,
        min_reward_gap=args.min_reward_gap,
        noise_injection=args.noise_injection,
        noise_scale=args.noise_scale,
    )

    vec_env.close()
    gc.collect()

    if len(pairs) == 0:
        print("[warn] No pairs collected — policy may always succeed or always fail.")
        return

    task_name = args.env_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{task_name}_preferences.hdf5")
    save_pairs_to_hdf5(pairs, output_path, task_name)

    print("\n[done] Preference data collection complete!")
    print(f"Output: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
