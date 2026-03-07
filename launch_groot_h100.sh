#!/bin/bash
# =============================================================================
# GR00T N1.6 RLHF Pipeline — RoboCasa GR1 Tabletop Tasks
# Usage: bash launch_groot_h100.sh [all|install|rollouts|dpo|rwr|eval] [n_pairs]
#
# Pipeline:
#   install  -> install all dependencies
#   rollouts -> collect preference pairs from GR00T policy in RoboCasa sim
#   dpo      -> DPO fine-tuning on preference pairs
#   rwr      -> RWR/RLHF fine-tuning on preference pairs
#   eval     -> evaluate all checkpoints
#   all      -> run full pipeline
# =============================================================================
set -e

STEP=${1:-"all"}
N_PAIRS=${2:-20}

# ---- Config ------------------------------------------------------------------
SFT_MODEL="karthikpythireddi93/gr00t-n16-gr1-tabletop-sft"
PREFERENCE_DIR="preference_data/gr1"
PREFERENCE_HDF5="$PREFERENCE_DIR/all_tasks_preferences.hdf5"
DPO_OUTPUT="outputs/dpo_groot"
RWR_OUTPUT="outputs/rwr_groot"
EVAL_OUTPUT="outputs/eval"
SERVER_PORT=5555

ENV_NAMES=(
    "gr1_unified/PnPCounterToCab_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/CuttingboardToBasket_GR1ArmsAndWaistFourierHands_Env"
)
# ------------------------------------------------------------------------------

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "============================================================"
echo " GR00T RLHF Pipeline"
echo " Step    : $STEP"
echo " N pairs : $N_PAIRS"
echo "============================================================"

# ---- Install -----------------------------------------------------------------
step_install() {
    echo "[install] Installing dependencies..."
    pip install -e ".[train]" --ignore-requires-python --quiet
    pip install -e external_dependencies/robocasa-gr1-tabletop-tasks --quiet
    pip install gymnasium h5py wandb --quiet
    echo "[install] Done."
}

# ---- Rollout collection ------------------------------------------------------
step_rollouts() {
    echo "[rollouts] Starting GR00T inference server in background..."
    python gr00t/eval/run_gr00t_server.py \
        --model-path "$SFT_MODEL" \
        --embodiment-tag new_embodiment \
        --use-sim-policy-wrapper \
        --denoising-steps 4 \
        --port $SERVER_PORT &
    SERVER_PID=$!
    echo "[rollouts] Server PID: $SERVER_PID — waiting 30s for warmup..."
    sleep 30

    mkdir -p "$PREFERENCE_DIR"

    for ENV in "${ENV_NAMES[@]}"; do
        TASK=$(echo "$ENV" | cut -d'/' -f2)
        OUT="$PREFERENCE_DIR/${TASK}_preferences.hdf5"
        if [ -f "$OUT" ]; then
            echo "[rollouts] $TASK already collected, skipping."
            continue
        fi
        echo "[rollouts] Collecting $N_PAIRS pairs for $TASK ..."
        python scripts/collect_preferences_groot.py \
            --env_name "$ENV" \
            --host localhost \
            --port $SERVER_PORT \
            --n_pairs $N_PAIRS \
            --output_dir "$PREFERENCE_DIR" \
            --noise_injection
    done

    kill $SERVER_PID 2>/dev/null || true
    echo "[rollouts] Server stopped."

    # Merge all task HDF5s into one file for training
    echo "[rollouts] Merging preference files..."
    python scripts/merge_preference_hdf5.py \
        "$PREFERENCE_DIR"/*_preferences.hdf5 \
        --output "$PREFERENCE_HDF5"
    echo "[rollouts] Done. Preference data: $PREFERENCE_HDF5"
}

# ---- DPO training ------------------------------------------------------------
step_dpo() {
    echo "[dpo] Starting DPO training..."
    python gr00t_rlhf/algos/dpo.py \
        --model_path "$SFT_MODEL" \
        --hdf5_path  "$PREFERENCE_HDF5" \
        --output_dir "$DPO_OUTPUT" \
        --beta 0.1 \
        --n_epochs 3 \
        --batch_size 2 \
        --lr 1e-5 \
        --n_windows_per_pair 5 \
        --use_wandb \
        --wandb_project gr00t-rlhf
    echo "[dpo] Done. Checkpoint: $DPO_OUTPUT"
}

# ---- RWR training ------------------------------------------------------------
step_rwr() {
    echo "[rwr] Starting RWR training..."
    python gr00t_rlhf/algos/rwr.py \
        --model_path "$SFT_MODEL" \
        --hdf5_path  "$PREFERENCE_HDF5" \
        --output_dir "$RWR_OUTPUT" \
        --temperature 1.0 \
        --n_epochs 3 \
        --batch_size 4 \
        --lr 1e-5 \
        --n_windows_per_pair 5 \
        --use_wandb \
        --wandb_project gr00t-rlhf
    echo "[rwr] Done. Checkpoint: $RWR_OUTPUT"
}

# ---- Evaluation --------------------------------------------------------------
step_eval() {
    echo "[eval] Evaluating all checkpoints..."
    for MODEL_DIR in "$SFT_MODEL" "$DPO_OUTPUT" "$RWR_OUTPUT"; do
        NAME=$(basename "$MODEL_DIR")
        echo "[eval] Evaluating $NAME ..."
        for ENV in "${ENV_NAMES[@]}"; do
            TASK=$(echo "$ENV" | cut -d'/' -f2)
            python scripts/eval/check_sim_eval_ready.py \
                --model_path "$MODEL_DIR" \
                --env_name "$ENV" \
                --output_dir "$EVAL_OUTPUT/$NAME/$TASK" \
                --n_episodes 20 2>/dev/null || \
            echo "[eval] eval script not ready for $NAME/$TASK — skipping"
        done
    done
    echo "[eval] Results in $EVAL_OUTPUT"
}

# ---- Dispatcher --------------------------------------------------------------
case $STEP in
    all)
        step_install
        step_rollouts
        step_dpo
        step_rwr
        step_eval
        ;;
    install)  step_install  ;;
    rollouts) step_rollouts ;;
    dpo)      step_dpo      ;;
    rwr)      step_rwr      ;;
    eval)     step_eval     ;;
    *)
        echo "Usage: bash launch_groot_h100.sh [all|install|rollouts|dpo|rwr|eval] [n_pairs]"
        exit 1
        ;;
esac

echo "============================================================"
echo " Pipeline step '$STEP' complete!"
echo "============================================================"
