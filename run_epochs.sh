#!/bin/bash
set -e

# Master active learning runner for 10 epochs
# Adds 10,000 training and 1,000 validation cases overall

MODEL_DIR="runs/subtitle-corrector-4b"
FUSED_DIR="runs/subtitle-corrector-4b-fused"

echo "=== STARTING ACTIVE LEARNING PIPELINE (10 EPOCHS) ==="

for epoch in {1..10}
do
    echo ""
    echo "============================================="
    echo "         STARTING EPOCH $epoch OF 10         "
    echo "============================================="
    echo ""
    
    # 1. Generate targeted synthetic data (1,000 train, 100 val)
    uv run python active_learning_loop.py "$epoch"
    
    # 2. Prepare dataset
    echo "Preparing dataset files..."
    uv run python -m subtitle_correction.cli prepare \
      --input ~/Downloads/.subcache/training_pairs.jsonl \
      --output-dir data \
      --val-split 0.1 \
      --augment 3 \
      --identity-ratio 0.25
      
    # 3. Clean up old adapter directory if it exists to train fresh from fused base
    if [ -d "$MODEL_DIR" ]; then
        echo "Removing old adapter checkpoint directory..."
        rm -rf "$MODEL_DIR"
    fi
    
    # 4. Train LoRA adapters (1000 steps)
    echo "Training model adapters (Epoch $epoch)..."
    # Run synchronously to wait for completion before next epoch
    uv run python -m subtitle_correction.cli train
    
    # 5. Fuse model weights to update the base model
    echo "Fusing model weights..."
    if [ -d "$FUSED_DIR" ]; then
        rm -rf "$FUSED_DIR"
    fi
    uv run python -m mlx_lm fuse \
      --model mlx-community/gemma-4-e4b-it-4bit \
      --adapter-path "$MODEL_DIR" \
      --save-path "$FUSED_DIR"
      
    # 6. Run final evaluation for this epoch
    echo "Running evaluation for Epoch $epoch..."
    uv run python -m subtitle_correction.cli evaluate --model "$FUSED_DIR" --fused
    
    echo "Epoch $epoch complete!"
done

echo ""
echo "=== ACTIVE LEARNING COMPLETED SUCCESSFULLY ==="
