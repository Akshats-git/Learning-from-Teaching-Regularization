#!/bin/bash
# Run all Phase 3 experiments: Alternative Imitability Metrics
# Tests: KL (baseline), L2, JS Divergence, Cosine Similarity

# Activate conda environment
if [ -d "env" ]; then
    eval "$(conda shell.bash hook)"
    conda activate ./env
    echo "✓ Conda environment activated"
else
    echo "⚠ Warning: env directory not found, using system Python"
fi

echo "======================================================================"
echo "PHASE 3: ALTERNATIVE IMITABILITY METRICS"
echo "======================================================================"
echo ""
echo "This will run 4 experiments on CIFAR-10:"
echo "1. KL Divergence (baseline LoT)"
echo "2. L2 Loss (mean squared error)"
echo "3. JS Divergence (symmetric KL)"
echo "4. Cosine Similarity (cosine distance)"
echo ""
echo "Dataset: CIFAR-10"
echo "Model: ResNet-18"
echo "Each experiment takes ~2-3 hours on GPU"
echo "======================================================================"
echo ""

# Configuration
PYTHON=python
DATASET='cifar10'
GPU=0
EPOCHS=100
BATCH_SIZE=128
LR=0.1
ALPHA=0.1
SEED=0

# Create directories
mkdir -p logs/Phase3
mkdir -p ckpt/Phase3

# Experiment 1: KL Divergence (Baseline)
echo "=========================================="
echo "Experiment 1/4: KL DIVERGENCE (BASELINE)"
echo "=========================================="
${PYTHON} trainer/alternative_metrics.py \
    --metric kl \
    --dataset ${DATASET} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase3/kl_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase3/kl_alpha${ALPHA}_seed${SEED}.log

echo "✓ KL Divergence (baseline) completed"
echo ""

# Experiment 2: L2 Loss
echo "=========================================="
echo "Experiment 2/4: L2 LOSS"
echo "=========================================="
${PYTHON} trainer/alternative_metrics.py \
    --metric l2 \
    --dataset ${DATASET} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase3/l2_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase3/l2_alpha${ALPHA}_seed${SEED}.log

echo "✓ L2 Loss completed"
echo ""

# Experiment 3: JS Divergence
echo "=========================================="
echo "Experiment 3/4: JS DIVERGENCE"
echo "=========================================="
${PYTHON} trainer/alternative_metrics.py \
    --metric js \
    --dataset ${DATASET} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase3/js_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase3/js_alpha${ALPHA}_seed${SEED}.log

echo "✓ JS Divergence completed"
echo ""

# Experiment 4: Cosine Similarity
echo "=========================================="
echo "Experiment 4/4: COSINE SIMILARITY"
echo "=========================================="
${PYTHON} trainer/alternative_metrics.py \
    --metric cosine \
    --dataset ${DATASET} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase3/cosine_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase3/cosine_alpha${ALPHA}_seed${SEED}.log

echo "✓ Cosine Similarity completed"
echo ""

echo "======================================================================"
echo "PHASE 3 COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - Checkpoints: ckpt/Phase3/"
echo "  - Logs: logs/Phase3/"
echo ""
echo "To analyze results, run:"
echo "  ./env/bin/python analysis/analyze_phase3.py"
echo ""
