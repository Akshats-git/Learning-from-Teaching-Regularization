#!/usr/bin/env python3
"""
Analysis script for Phase 1: Feedback Balance Variants
Compares Positive-only, Negative-only, and Mixed feedback approaches
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path


def load_checkpoint(path):
    """Load checkpoint and extract metrics"""
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def analyze_phase1():
    """Analyze Phase 1 results"""
    print("="*80)
    print("PHASE 1 ANALYSIS: FEEDBACK BALANCE VARIANTS")
    print("="*80)
    print()
    
    # Load checkpoints
    feedback_types = ['positive', 'negative', 'mixed']
    results = {}
    
    for ftype in feedback_types:
        ckpt_path = f'ckpt/Phase1/{ftype}_alpha0.1_seed0.pt'
        ckpt = load_checkpoint(ckpt_path)
        if ckpt:
            results[ftype] = {
                'best_val_ppl': ckpt['best_val_ppl'],
                'epoch': ckpt['epoch'],
                'args': ckpt['args']
            }
            print(f"{ftype.upper()} Feedback:")
            print(f"  Best Validation PPL: {ckpt['best_val_ppl']:.2f}")
            print(f"  Achieved at epoch: {ckpt['epoch']}")
            print()
    
    # Comparison
    if len(results) == 3:
        print("-" * 80)
        print("COMPARISON:")
        print("-" * 80)
        
        ppls = {k: v['best_val_ppl'] for k, v in results.items()}
        best_method = min(ppls, key=ppls.get)
        
        print(f"Best method: {best_method.upper()}")
        print(f"  PPL: {ppls[best_method]:.2f}")
        print()
        
        print("Relative Performance:")
        baseline_ppl = ppls['mixed']  # Use mixed as baseline
        for method in ['positive', 'negative', 'mixed']:
            if method in ppls:
                improvement = ((baseline_ppl - ppls[method]) / baseline_ppl) * 100
                print(f"  {method.capitalize():12s}: {ppls[method]:.2f} PPL "
                      f"({improvement:+.2f}% vs mixed)")
        print()
    
    # Key insights
    print("-" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    print()
    print("1. POSITIVE-ONLY FEEDBACK:")
    print("   - Reward when student copies teacher well")
    print("   - No penalty for being hard to imitate")
    print("   - Expected: May encourage simpler teacher behaviors")
    print()
    print("2. NEGATIVE-ONLY FEEDBACK:")
    print("   - Penalty when student can't imitate teacher")
    print("   - No reward for being easy to copy")
    print("   - Expected: Teacher learns to be more teachable")
    print()
    print("3. MIXED FEEDBACK (Original LoT):")
    print("   - Both reward and penalty terms")
    print("   - Balanced approach")
    print("   - Expected: Best of both worlds")
    print()
    
    # Recommendations
    print("-" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)
    if len(results) == 3:
        if ppls['positive'] < ppls['mixed']:
            print("✓ Positive-only feedback works better!")
            print("  → Student can learn effectively without teacher being constrained")
        elif ppls['negative'] < ppls['mixed']:
            print("✓ Negative-only feedback works better!")
            print("  → Teachability constraint improves learning")
        else:
            print("✓ Mixed feedback is best (as expected)")
            print("  → Balance between reward and penalty is optimal")
    print()


if __name__ == '__main__':
    analyze_phase1()
    print("="*80)
    print("Analysis complete!")
    print("="*80)
