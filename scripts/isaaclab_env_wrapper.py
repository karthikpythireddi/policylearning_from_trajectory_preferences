"""
Isaac Lab environment wrapper that matches LIBERO's OffScreenRenderEnv interface.

Usage:
    from scripts.isaaclab_env_wrapper import IsaacLabEnvWrapper
    env = IsaacLabEnvWrapper(num_envs=1)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    success = env.check_success()
    env.close()

The wrapper exposes the same obs dict keys as LIBERO so that BCTransformerPolicy,
collect_preferences.py, evaluate_policies.py, and the RLHF/DPO/PPO algos all work
without modification.
"""

import torch
import numpy as np


class IsaacLabEnvWrapper:
    """
    Wraps Isaac-Lift-Cube-Franka-Visuomotor-v0 to match LIBERO's OffScreenRenderEnv.

    Obs dict keys returned by reset() and step():
        agentview_rgb   : np.uint8  (H, W, 3)   — front-view camera
        eye_in_hand_rgb : np.uint8  (H, W, 3)   — wrist camera
        joint_states    : np.float32 (7,)        — arm joint positions
        gripper_states  : np.float32 (2,)        — finger positions
    """

    TASK_ID = "Isaac-Lift-Cube-Franka-Visuomotor-v0"

    def __init__(self, num_envs: int = 1, device: str = "cuda:0"):
        # Isaac Lab imports are deferred so this file can be imported
        # outside the isaaclab conda env without errors
        from isaaclab.envs import ManagerBasedRLEnv
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401 — registers all task IDs

        self.device = device
        env_cfg_cls = gym.spec(self.TASK_ID).kwargs["env_cfg_entry_point"]

        # Resolve string entry point to class
        module_path, cls_name = env_cfg_cls.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        env_cfg = getattr(module, cls_name)()
        env_cfg.scene.num_envs = num_envs

        self._env = ManagerBasedRLEnv(cfg=env_cfg)
        self._num_envs = num_envs

    # ------------------------------------------------------------------
    # Core interface (mirrors LIBERO's OffScreenRenderEnv)
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        obs_dict, _ = self._env.reset()
        return self._process_obs(obs_dict)

    def step(self, action: np.ndarray):
        """
        action: np.ndarray of shape (7,) or (num_envs, 7)
          [0:6] EEF pose delta, [6] gripper binary
        Returns: (obs_dict, reward, done, info)
        """
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0).expand(self._num_envs, -1)

        obs_dict, reward, terminated, truncated, info = self._env.step(action_tensor)
        done = (terminated | truncated).cpu().numpy()
        reward = reward.cpu().numpy()
        return self._process_obs(obs_dict), reward, done, info

    def check_success(self) -> np.ndarray:
        """Returns bool array of shape (num_envs,)."""
        # Isaac Lab stores episode stats in extras after each step
        # The lift task's termination condition is tracked via the reward signal;
        # here we check object height as a proxy (same as LIBERO's _check_success)
        object_pos = self._env.scene["object"].data.root_pos_w[:, 2]
        return (object_pos > 0.1).cpu().numpy()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_obs(self, obs_dict: dict) -> dict:
        """Convert Isaac Lab obs tensors to LIBERO-compatible numpy arrays."""
        policy_obs = obs_dict["policy"]
        out = {}

        # RGB images: (num_envs, H, W, 3) tensor → (H, W, 3) uint8 for env 0
        # For multi-env rollout collection keep the batch dim; callers handle it.
        out["agentview_rgb"] = self._to_uint8(policy_obs["agentview_rgb"])
        out["eye_in_hand_rgb"] = self._to_uint8(policy_obs["eye_in_hand_rgb"])

        # Proprioception
        out["joint_states"] = policy_obs["joint_states"].cpu().numpy().astype(np.float32)
        out["gripper_states"] = policy_obs["gripper_states"].cpu().numpy().astype(np.float32)

        return out

    @staticmethod
    def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
        """Convert (N, H, W, 3) float/uint tensor to uint8 numpy."""
        arr = tensor.cpu().numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
