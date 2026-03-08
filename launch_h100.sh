#!/bin/bash
# =============================================================================
# CS234 LIBERO RLHF Pipeline — Single Command Launch
# Usage: bash launch_h100.sh
# Runs full pipeline: install → dataset → SFT → DPO → RWR → PPO → eval
# =============================================================================
set -e

# Ensure the repo root is always on PYTHONPATH so 'libero' package is importable
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd):${PYTHONPATH}"

# ---- Config (edit these) ----------------------------------------------------
ALGO=${1:-"all"}          # all | dpo | rlhf | ppo | sft | eval
SEED=${2:-10000}
TASK_SUITE="libero_10"
BC_CHECKPOINT_DIR="experiments/libero_10/SingleTask/BCTransformerPolicy_seed${SEED}/run_004"
BC_TASK0_DIR="experiments/libero_10/SingleTask/BCTransformerPolicy_seed${SEED}/run_003"
PREFERENCE_DATA_DIR="preference_data"
OUTPUT_DIR="experiments/h100_results"
# -----------------------------------------------------------------------------

echo "============================================================"
echo " CS234 LIBERO RLHF Pipeline"
echo " Algorithm : $ALGO"
echo " Suite     : $TASK_SUITE"
echo " Seed      : $SEED"
echo "============================================================"

# ---- Step 1: System dependencies --------------------------------------------
step_install() {
    echo "[1/6] Installing dependencies..."

    # Headless rendering for H100
    apt-get install -y libgl1-mesa-glx libegl1-mesa libgles2-mesa \
        libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg 2>/dev/null || true

    # MuJoCo (no version pin — 2.3.7 has no cp312 wheel)
    pip install mujoco --quiet

    # Install from requirements.txt (no strict version pins for Python 3.12 compat)
    pip install \
        "hydra-core==1.3.2" easydict h5py einops wandb "imageio[ffmpeg]" termcolor \
        robosuite robomimic bddl cloudpickle gym \
        transformers timm thop future matplotlib opencv-python \
        --quiet

    # Install LIBERO package
    pip install -e . --quiet

    echo "[1/6] Done."
}

# ---- Step 2: Download LIBERO-10 dataset -------------------------------------
step_dataset() {
    echo "[2/6] Downloading LIBERO-10 dataset..."
    export MUJOCO_GL=egl
    python benchmark_scripts/download_libero_datasets.py --datasets libero_100
    echo "[2/6] Done."
}

# ---- Step 3: SFT (BC baseline) ----------------------------------------------
step_sft() {
    echo "[3/6] Training BC baseline (SingleTask) for tasks 1-9..."
    export MUJOCO_GL=egl

    # Task 0 already trained (run_003/task0_model.pth).
    # Train tasks 1-9 in run_004.
    # Speed knobs vs original (50 epochs, batch=32, eval_every=5, n_eval=20):
    #   n_epochs   50→25       saves ~50% training time
    #   batch_size 32→64       better GPU utilization (~20% faster per epoch)
    #   eval_every  5→24       2 evals per task (epoch 0 + final) vs 10 before
    #   n_eval     20→10       halves eval wall-time (~74s vs ~148s per eval)
    #   eval.eval=false        skip post-task confusion matrix (not needed for RLHF)
    # Estimated: ~25-30 min/task × 9 tasks ≈ 4-5 hours total
    python libero/lifelong/main.py \
        benchmark_name=$TASK_SUITE \
        policy=bc_transformer_policy \
        lifelong=single_task \
        seed=$SEED \
        train.n_epochs=25 \
        train.batch_size=64 \
        eval.eval_every=24 \
        eval.n_eval=10 \
        eval.eval=false \
        "train_task_ids=[1,2,3,4,5,6,7,8,9]" \
        use_wandb=true

    # Copy task 0 checkpoint into run_004 so rollout collection has all 10 tasks
    if [ -f "$BC_TASK0_DIR/task0_model.pth" ]; then
        cp "$BC_TASK0_DIR/task0_model.pth" "$BC_CHECKPOINT_DIR/task0_model.pth"
        echo "[3/6] Copied task0_model.pth from run_003 -> run_004"
    fi

    echo "[3/6] BC training done. Checkpoint: $BC_CHECKPOINT_DIR"
}

# ---- Step 3b: Rollout collection (preference data) --------------------------
step_rollouts() {
    echo "[3b/6] Collecting rollouts and preference pairs..."
    export MUJOCO_GL=egl

    # Find the hydra config from the most recent Hydra outputs dir
    CFG_YAML=$(find outputs -name "config.yaml" -path "*/.hydra/*" | sort | tail -1)
    if [ -z "$CFG_YAML" ]; then
        echo "ERROR: No hydra config found under outputs/."
        echo "       Run 'bash launch_h100.sh sft' first."
        exit 1
    fi
    echo "[3b/6] Using config: $CFG_YAML"

    python scripts/collect_preferences.py \
        --cfg "$CFG_YAML" \
        --checkpoint_dir "$BC_CHECKPOINT_DIR" \
        --n_pairs 20 \
        --output_dir "$PREFERENCE_DATA_DIR"

    echo "[3b/6] Preference data saved to $PREFERENCE_DATA_DIR"
}

# ---- Step 4: DPO training ---------------------------------------------------
step_dpo() {
    echo "[4/6] Running DPO..."
    export MUJOCO_GL=egl

    python libero/lifelong/main.py \
        benchmark_name=$TASK_SUITE \
        policy=bc_transformer_policy \
        lifelong=dpo \
        seed=$SEED \
        lifelong.bc_checkpoint_dir=$BC_CHECKPOINT_DIR \
        lifelong.preference_data_dir=$PREFERENCE_DATA_DIR \
        lifelong.dpo_beta=0.1 \
        use_wandb=true

    echo "[4/6] DPO done."
}

# ---- Step 5: RWR (RLHF) training --------------------------------------------
step_rlhf() {
    echo "[5/6] Running RWR (RLHF)..."
    export MUJOCO_GL=egl

    python libero/lifelong/main.py \
        benchmark_name=$TASK_SUITE \
        policy=bc_transformer_policy \
        lifelong=rlhf \
        seed=$SEED \
        lifelong.bc_checkpoint_dir=$BC_CHECKPOINT_DIR \
        lifelong.preference_data_dir=$PREFERENCE_DATA_DIR \
        lifelong.rwr_temperature=1.0 \
        use_wandb=true

    echo "[5/6] RWR done."
}

# ---- Step 6: PPO training ---------------------------------------------------
step_ppo() {
    echo "[6/6] Running PPO..."
    export MUJOCO_GL=egl

    python libero/lifelong/main.py \
        benchmark_name=$TASK_SUITE \
        policy=bc_transformer_policy \
        lifelong=ppo \
        seed=$SEED \
        lifelong.bc_checkpoint_dir=$BC_CHECKPOINT_DIR \
        lifelong.preference_data_dir=$PREFERENCE_DATA_DIR \
        lifelong.ppo_iters=30 \
        lifelong.n_rollouts_per_iter=4 \
        use_wandb=true

    echo "[6/6] PPO done."
}

# ---- Evaluation -------------------------------------------------------------
step_eval() {
    echo "[eval] Running evaluation across all algorithms..."
    export MUJOCO_GL=egl

    for ALGO_NAME in SingleTask dpo rlhf ppo; do
        CKPT_DIR="$OUTPUT_DIR/$ALGO_NAME"
        [ "$ALGO_NAME" = "SingleTask" ] && CKPT_DIR="$BC_CHECKPOINT_DIR"
        echo "Evaluating $ALGO_NAME..."
        python scripts/evaluate_policies.py \
            --benchmark $TASK_SUITE \
            --checkpoint_dir $CKPT_DIR \
            --output_dir $OUTPUT_DIR/eval/$ALGO_NAME \
            --n_eval_episodes 20 2>/dev/null || \
        python libero/lifelong/evaluate.py \
            benchmark_name=$TASK_SUITE \
            checkpoint_dir=$CKPT_DIR \
            exp_dir=$OUTPUT_DIR/eval/$ALGO_NAME
    done

    echo "[eval] Results saved to $OUTPUT_DIR/eval/"
}

# ---- Main dispatcher --------------------------------------------------------
case $ALGO in
    all)
        step_install
        step_dataset
        step_sft
        step_rollouts
        step_dpo
        step_rlhf
        step_ppo
        step_eval
        ;;
    install)   step_install ;;
    dataset)   step_install && step_dataset ;;
    sft)       step_sft ;;
    rollouts)  step_rollouts ;;
    dpo)       step_dpo ;;
    rlhf)      step_rlhf ;;
    ppo)       step_ppo ;;
    eval)      step_eval ;;
    *)
        echo "Usage: bash launch_h100.sh [all|install|dataset|sft|rollouts|dpo|rlhf|ppo|eval] [seed]"
        echo "  all      - run full pipeline (default)"
        echo "  install  - install dependencies only"
        echo "  dataset  - install + download dataset"
        echo "  sft      - train BC baseline"
        echo "  rollouts - collect rollouts + preference pairs (requires SFT checkpoint)"
        echo "  dpo     - run DPO (requires BC checkpoint + preference data)"
        echo "  rlhf    - run RWR/RLHF (requires BC checkpoint + preference data)"
        echo "  ppo     - run PPO (requires BC checkpoint + preference data)"
        echo "  eval    - evaluate all checkpoints"
        exit 1
        ;;
esac

echo "============================================================"
echo " Pipeline complete! Results in: $OUTPUT_DIR"
echo "============================================================"
