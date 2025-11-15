#!/bin/bash
# Run all Phase 2 experiments: Teacher-Student Size Ratio
# Tests: Large→Small, Medium→Small, Small→Similar

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
echo "PHASE 2: TEACHER-STUDENT SIZE RATIO EXPERIMENTS"
echo "======================================================================"
echo ""
echo "This will run 3 experiments:"
echo "1. Large Teacher (1024) → Small Student (400)"
echo "2. Medium Teacher (650) → Small Student (400)"
echo "3. Small Teacher (400) → Similar Student (400)"
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
mkdir -p logs/Phase2
mkdir -p ckpt/Phase2

# Experiment 1: Large Teacher → Small Student
echo "=========================================="
echo "Experiment 1/3: LARGE TEACHER → SMALL STUDENT"
echo "Teacher: 1024 units | Student: 400 units"
echo "=========================================="
${PYTHON} trainer/size_ratios.py \
    --teacher_size large \
    --student_size small \
    --data ${DATA} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BPTT} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase2/Tlarge_Ssmall_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase2/Tlarge_Ssmall_alpha${ALPHA}_seed${SEED}.log

echo "✓ Large→Small completed"
echo ""

# Experiment 2: Medium Teacher → Small Student
echo "=========================================="
echo "Experiment 2/3: MEDIUM TEACHER → SMALL STUDENT"
echo "Teacher: 650 units | Student: 400 units"
echo "=========================================="
${PYTHON} trainer/size_ratios.py \
    --teacher_size medium \
    --student_size small \
    --data ${DATA} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BPTT} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase2/Tmedium_Ssmall_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase2/Tmedium_Ssmall_alpha${ALPHA}_seed${SEED}.log

echo "✓ Medium→Small completed"
echo ""

# Experiment 3: Small Teacher → Similar Student
echo "=========================================="
echo "Experiment 3/3: SMALL TEACHER → SIMILAR STUDENT"
echo "Teacher: 400 units | Student: 400 units"
echo "=========================================="
${PYTHON} trainer/size_ratios.py \
    --teacher_size small \
    --student_size small \
    --data ${DATA} \
    --gpu ${GPU} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BPTT} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --seed ${SEED} \
    --save ckpt/Phase2/Tsmall_Ssmall_alpha${ALPHA}_seed${SEED}.pt \
    2>&1 | tee logs/Phase2/Tsmall_Ssmall_alpha${ALPHA}_seed${SEED}.log

echo "✓ Small→Similar completed"
echo ""

echo "======================================================================"
echo "PHASE 2 COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - Checkpoints: ckpt/Phase2/"
echo "  - Logs: logs/Phase2/"
echo ""
echo "To analyze results, run:"
echo "  ./env/bin/python analysis/analyze_phase2.py"
echo ""
