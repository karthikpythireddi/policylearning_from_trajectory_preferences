# Preference Data Collection

## Overview

`scripts/collect_preferences.py` generates trajectory preference pairs for training DPO and RLHF policies.

### Strategy

| Role | Source | Label |
|---|---|---|
| **Winner (τ_w)** | Human demonstrations from LIBERO HDF5 files | Always preferred |
| **Loser (τ_l)** | BC policy rollouts that fail the task | Rejected |

Only failed rollouts are kept as losers — this ensures a **clear preference signal**. Ties (both succeed) are skipped.

---

## Prerequisites

1. BC policy trained on `libero_10` via `libero/lifelong/main.py`
2. Checkpoints saved at `experiments/libero_10/SingleTask/.../task{i}_model.pth`

---

## Usage

```bash
conda activate libero
cd /home/karthik/CS234/final_project/LIBERO

python scripts/collect_preferences.py \
    --cfg outputs/DATE/TIME/.hydra/config.yaml \
    --checkpoint_dir experiments/libero_10/SingleTask/SEED10000 \
    --n_pairs 20 \
    --max_steps 300 \
    --output_dir preference_data \
    --device cuda
```

### Find your checkpoint directory

```bash
find experiments/ -name "task0_model.pth"
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--cfg` | required | Path to Hydra config yaml from the BC training run |
| `--checkpoint_dir` | required | Directory containing `task{i}_model.pth` files |
| `--benchmark` | `libero_10` | LIBERO benchmark name |
| `--output_dir` | `preference_data` | Where to save preference HDF5 files |
| `--n_pairs` | `20` | Number of preference pairs to collect per task |
| `--max_steps` | `300` | Max rollout steps per episode |
| `--noise_prob` | `0.0` | Probability of flipping a label (simulates imperfect human feedback) |
| `--device` | `cuda` | Device for policy inference |
| `--task_ids` | all | Specific task IDs to process, e.g. `--task_ids 0 1 2` |

---

## Output Structure

```
preference_data/
└── libero_10/
    ├── task0_preferences.hdf5
    ├── task1_preferences.hdf5
    ├── ...
    └── task9_preferences.hdf5
```

Each HDF5 file has the following structure:

```
task{i}_preferences.hdf5
├── metadata/
│   ├── n_pairs          (int)
│   ├── task_name        (str)
│   └── task_emb         (768,)   BERT language embedding
│
├── pair_0/
│   ├── winner/
│   │   ├── actions               (T_w, 7)
│   │   └── obs/
│   │       ├── agentview_rgb     (T_w, H, W, 3)  uint8
│   │       ├── eye_in_hand_rgb   (T_w, H, W, 3)  uint8
│   │       ├── joint_states      (T_w, 7)
│   │       └── gripper_states    (T_w, 2)
│   └── loser/
│       ├── actions               (T_l, 7)
│       └── obs/  (same keys as winner)
│
├── pair_1/
│   └── ...
└── pair_{n-1}/
```

---

## Pipeline Position

```
[1] Train BC Policy (π_ref)          ← libero/lifelong/main.py
        ↓
[2] Collect Preferences              ← scripts/collect_preferences.py  (YOU ARE HERE)
        ↓
   ┌────┴────┐
   ↓         ↓
[3] DPO     RLHF                     ← libero/lifelong/algos/dpo.py
                                        libero/lifelong/algos/rlhf.py
```

---

## Notes

- **Label noise**: Use `--noise_prob 0.1` to simulate 10% noisy human feedback, as described in the proposal
- **Policy quality matters**: A policy that always succeeds produces no losers. Run collection after early training (e.g. 10-20 epochs) when the policy still fails frequently
- **Scaling**: With `--n_pairs 20` and 10 tasks, collection takes ~20-40 min on GPU depending on rollout length
