# CS234 Final Project: RLHF for Robot Manipulation

Fine-tuning BC policies with human preference feedback on the [LIBERO-10](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark.

Built on top of the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) framework.

---

## Overview

We compare three preference-based fine-tuning methods against a BC baseline on 4 LIBERO-10 tasks:

- **BC** — Behavioral Cloning baseline (BCTransformerPolicy with GMM head, trained on 50 expert demos per task)
- **DPO** — Direct Preference Optimization (β=0.1, fine-tunes BC directly on preference pairs)
- **RLHF** — Reward model trained on preference pairs (Bradley-Terry loss), then supervised fine-tuning
- **PPO** — Reward model (Phase 1) + PPO policy gradient fine-tuning (Phase 2)

Tasks evaluated: 0, 3, 4, 7 from LIBERO-10 (task 8 dropped — BC fails 100% of rollouts, making preference collection impossible).

---

## Results

**Success Rate (20 rollouts per task/method)**

| Task | Description | BC | DPO | RLHF | PPO |
|------|-------------|:---:|:---:|:----:|:---:|
| 0 | put alphabet soup + tomato sauce in basket | 65.0% | 65.0% | **70.0%** | 0.0% |
| 3 | put black bowl in bottom drawer of cabinet | 90.0% | 95.0% | **95.0%** | 0.0% |
| 4 | put white mug on left plate | 60.0% | **65.0%** | 50.0% | 0.0% |
| 7 | put alphabet soup + cream cheese in basket | 25.0% | 35.0% | **50.0%** | 0.0% |
| **AVG** | | 60.0% | 65.0% | **66.2%** | 0.0% |

Videos of rollouts are in `videos/task{id}_{name}/{method}.mp4` (green border = success, red = failure).

---

## Key Findings

- **RLHF** is the best method overall (+6.2% over BC), showing that a learned reward model with supervised fine-tuning can reliably improve the policy.
- **DPO** modestly outperforms BC on average (+5%), with notable gains on tasks 3 and 7, but slightly underperforms on task 4 and ties on task 0.
- **PPO** achieves 0% success on all tasks due to: (1) sparse reward signal — the policy almost never reaches task completion during rollouts so reward gradients vanish, (2) memory constraints forcing a mini-batch size of 8 on a 16GB GPU leading to high-variance gradient estimates, and (3) uncalibrated reward model outputs requiring reward normalization which still could not overcome the sparse exploration problem.

---

## Project Structure

```
scripts/
  collect_preferences.py     # Collect trajectory preference pairs using BC oracle
  evaluate_policies.py       # Evaluate and compare all methods, record videos

libero/lifelong/algos/
  dpo.py                     # DPO algorithm
  rlhf.py                    # RLHF algorithm (reward model + fine-tuning)
  ppo.py                     # PPO algorithm (reward model + policy gradient)

libero/lifelong/models/
  reward_model.py            # Bradley-Terry reward model (ResNet encoder + MLP head)

libero/lifelong/
  datasets_preference.py     # HDF5 preference pair dataloader

libero/configs/lifelong/
  dpo.yaml                   # DPO hyperparameters
  rlhf.yaml                  # RLHF hyperparameters
  ppo.yaml                   # PPO hyperparameters

videos/                      # Rollout videos for each (task, method) pair
```

---

## Reproducing Results

### 1. BC Training
```bash
python libero/lifelong/main.py \
    benchmark_name=libero_10 \
    lifelong=single_task \
    policy=bc_transformer_policy \
    device=cuda \
    train_task_ids=[0,3,4,7] \
    seed=10000
```

### 2. Preference Data Collection
```bash
python scripts/collect_preferences.py \
    --cfg outputs/<date>/<time>/.hydra/config.yaml \
    --task_ids 0 3 4 7 \
    --bc_dir experiments/libero_10/SingleTask/BCTransformerPolicy_seed10000/run_003 \
    --n_pairs 50 \
    --output_dir preference_data
```

### 3. RLHF Training
```bash
python libero/lifelong/main.py \
    benchmark_name=libero_10 lifelong=rlhf \
    policy=bc_transformer_policy device=cuda \
    eval.num_procs=1 train_task_ids=[0,3,4,7] seed=10000
```

### 4. DPO Training
```bash
python libero/lifelong/main.py \
    benchmark_name=libero_10 lifelong=dpo \
    policy=bc_transformer_policy device=cuda \
    eval.num_procs=1 train_task_ids=[0,3,4,7] seed=10000
```

### 5. PPO Training
```bash
python libero/lifelong/main.py \
    benchmark_name=libero_10 lifelong=ppo \
    policy=bc_transformer_policy device=cuda \
    eval.num_procs=1 train_task_ids=[0,3,4,7] seed=10000
```

### 6. Evaluation
```bash
python scripts/evaluate_policies.py \
    --cfg outputs/<date>/<time>/.hydra/config.yaml \
    --task_ids 0 3 4 7 --n_eval 20 --device cuda \
    --save_video --video_dir videos \
    --bc_dir   experiments/libero_10/SingleTask/BCTransformerPolicy_seed10000/run_003 \
    --dpo_dir  experiments/libero_10/DPO/BCTransformerPolicy_seed10000/run_002 \
    --rlhf_dir experiments/libero_10/RLHF/BCTransformerPolicy_seed10000/run_002 \
    --ppo_dir  experiments/libero_10/PPO/BCTransformerPolicy_seed10000/run_007
```

---

## Acknowledgements

Built on top of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT License).
