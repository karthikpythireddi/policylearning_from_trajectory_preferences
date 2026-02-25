#!/bin/bash
# Watches for PPO run_007 to finish, then runs evaluate_policies.py automatically.

LIBERO_DIR="/home/karthik/CS234/final_project/LIBERO"
PYTHON="/home/karthik/miniconda3/envs/libero/bin/python"
PPO_DIR="$LIBERO_DIR/experiments/libero_10/PPO/BCTransformerPolicy_seed10000/run_007"
LOG="$LIBERO_DIR/scripts/watch_and_evaluate.log"

cd "$LIBERO_DIR"

echo "[$(date)] Watcher started. Waiting for PPO run_007 to complete..." | tee "$LOG"
echo "[$(date)] Watching for: $PPO_DIR/task7_model.pth" | tee -a "$LOG"

# Poll every 2 minutes until task7_model.pth exists
while [ ! -f "$PPO_DIR/task7_model.pth" ]; do
    sleep 120
    echo "[$(date)] Still waiting... PPO checkpoints so far:" | tee -a "$LOG"
    ls "$PPO_DIR"/*.pth 2>/dev/null | tee -a "$LOG" || echo "  (none yet)" | tee -a "$LOG"
done

echo "[$(date)] task7_model.pth found! Waiting 3 min for PPO to fully finish..." | tee -a "$LOG"
sleep 180

echo "[$(date)] Starting evaluate_policies.py..." | tee -a "$LOG"

$PYTHON scripts/evaluate_policies.py \
    --cfg outputs/2026-02-23/08-37-38/.hydra/config.yaml \
    --task_ids 0 3 4 7 \
    --n_eval 20 \
    --device cuda \
    --save_video \
    --video_dir videos \
    --bc_dir   experiments/libero_10/SingleTask/BCTransformerPolicy_seed10000/run_003 \
    --dpo_dir  experiments/libero_10/DPO/BCTransformerPolicy_seed10000/run_002 \
    --rlhf_dir experiments/libero_10/RLHF/BCTransformerPolicy_seed10000/run_002 \
    --ppo_dir  experiments/libero_10/PPO/BCTransformerPolicy_seed10000/run_007 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Evaluation complete! Results above." | tee -a "$LOG"
