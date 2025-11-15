#!/bin/bash
#
# Phase 4 PRODUCTION: Properly Integrated LoT + Emergent Language
# Target: Beat baseline LoT by 5-15 PPL via emergent language communication
#

PYTHON=/home/rajeev-kumar/Desktop/LoT/env/bin/python

# Experiment name
EXP_NAME="Lot_Emergent_ptb"

# Paths
SAVE="ckpt/Phase4/${EXP_NAME}.pt"
LOG="logs/Phase4/${EXP_NAME}.log"

# Create directories
mkdir -p ckpt/Phase4
mkdir -p logs/Phase4
mkdir -p result/Phase4

echo "================================================================================"
echo "PHASE 4 PRODUCTION: LoT + Emergent Language"
echo "================================================================================"
echo ""
echo "Strategy:"
echo "  1. Start with strong LoT baseline (alpha=1.0, same as original)"
echo "  2. Add emergent language messages for ENHANCED knowledge transfer"
echo "  3. Curriculum learning: gradual message integration"
echo "  4. Small language regularization to keep messages informative"
echo ""
echo "Expected Results:"
echo "  - Teacher PPL: ~110-130 (similar to baseline)"
echo "  - Student (with msg) PPL: ~120-140 (BETTER than baseline ~145)"
echo "  - Message benefit: +5-15 PPL improvement"
echo ""
echo "Configuration:"
echo "  Dataset:              PTB"
echo "  GPU:                  0"
echo "  Message vocab:        32 (small, easier to learn)"
echo "  Message length:       8 (short, focused)"
echo "  LoT alpha:            1.0 (SAME as baseline)"
echo "  Language reg:         0.001 (very small)"
echo "  Epochs:               60 (SAME as baseline)"
echo "  Learning rate:        30 (SAME as baseline)"
echo "  Student steps:        5 (SAME as baseline)"
echo ""
echo "Output:"
echo "  Checkpoint:           ${SAVE}"
echo "  Log:                  ${LOG}"
echo "================================================================================"
echo ""

# Run training
$PYTHON trainer/Lot_Emergent.py \
    --data ptb \
    --gpu 0 \
    --vocab_size 32 \
    --msg_length 8 \
    --alpha 1.0 \
    --T 1.5 \
    --beta_lang 0.0001 \
    --beta_entropy 0.5 \
    --beta_comp 0.1 \
    --emsize 650 \
    --nhid 650 \
    --nlayers 2 \
    --dropout 0.45 \
    --lr 30 \
    --clip 0.2 \
    --epochs 60 \
    --batch_size 20 \
    --bptt 35 \
    --student_steps_ratio 5 \
    --save $SAVE \
    2>&1 | tee $LOG

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE!"
echo "================================================================================"
echo "  Checkpoint: $SAVE"
echo "  Log:        $LOG"
echo ""
echo "Next steps:"
echo "  1. Compare with baseline: Check logs/LoT_LSTM_PTB/"
echo "  2. Run analysis: python analysis/analyze_phase4_full.py --checkpoint $SAVE"
echo "================================================================================"
