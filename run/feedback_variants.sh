#!/bin/bash
# Run all Phase 1 experiments: Feedback Balance Variants
# Tests: Positive-only, Negative-only, and Mixed feedback

# Activate conda environment
if [ -d "env" ]; then
    # Use conda to activate the environment
    eval "$(conda shell.bash hook)"
    conda activate ./env
    echo "✓ Conda environment activated"
else
    echo "⚠ Warning: env directory not found, using system Python"
fi

echo "======================================================================"
echo "PHASE 1: FEEDBACK BALANCE VARIANTS"
echo "======================================================================"
echo ""
echo "This will run 3 experiments:"
echo "1. Positive-only feedback (reward if student copies teacher well)"
echo "2. Negative-only feedback (penalty if student is hard to imitate)"
echo "3. Mixed feedback (original LoT with both terms)"
echo ""
echo "Dataset: PTB (works on 6GB GPU)"
echo "Each experiment takes ~2-3 hours"
echo "======================================================================"
echo ""

# Configuration
PYTHON=python
DATA='ptb'
GPU=0
EPOCHS=40
BATCH_SIZE=20
BPTT=35
LR=20
ALPHA=0.1
SEED=0

# Create directories
mkdir -p logs/Phase1
mkdir -p ckpt/Phase1

# Experiment 1: Positive-only feedback
echo "=========================================="
echo "Experiment 1/3: POSITIVE-ONLY FEEDBACK"
echo "=========================================="
${PYTHON} trainer/feedback_variants.py \
    --feedback_type positive \
    --data ${DATA} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BPTT} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase1/positive_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase1/positive_alpha${ALPHA}_seed${SEED}.log

echo "✓ Positive-only feedback completed"
echo ""

# Experiment 2: Negative-only feedback
echo "=========================================="
echo "Experiment 2/3: NEGATIVE-ONLY FEEDBACK"
echo "=========================================="
${PYTHON} trainer/feedback_variants.py \
    --feedback_type negative \
    --data ${DATA} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BPTT} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase1/negative_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase1/negative_alpha${ALPHA}_seed${SEED}.log

echo "✓ Negative-only feedback completed"
echo ""

# Experiment 3: Mixed feedback (original LoT)
echo "=========================================="
echo "Experiment 3/3: MIXED FEEDBACK (Original LoT)"
echo "=========================================="
${PYTHON} trainer/feedback_variants.py \
    --feedback_type mixed \
    --data ${DATA} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BPTT} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase1/mixed_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase1/mixed_alpha${ALPHA}_seed${SEED}.log

echo "✓ Mixed feedback completed"
echo ""

echo "======================================================================"
echo "PHASE 1 COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - Checkpoints: ckpt/Phase1/"
echo "  - Logs: logs/Phase1/"
echo ""
echo "To analyze results, run:"
echo "  ./env/bin/python analysis/analyze_phase1.py"
echo ""
