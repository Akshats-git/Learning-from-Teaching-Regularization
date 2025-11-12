#!/usr/bin/env python3
"""
Analysis script for Phase 2: Teacher-Student Size Ratios
Analyzes how teacher size relative to student impacts learnability
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def load_checkpoint(path):
    """Load checkpoint and extract metrics"""
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint


def analyze_phase2():
    """Analyze Phase 2 results"""
    print("="*80)
    print("PHASE 2 ANALYSIS: TEACHER-STUDENT SIZE RATIO")
    print("="*80)
    print()
    
    # Load checkpoints
    experiments = [
        ('Tlarge_Ssmall', 'Large Teacher → Small Student'),
        ('Tmedium_Ssmall', 'Medium Teacher → Small Student'),
        ('Tsmall_Ssmall', 'Small Teacher → Similar Student'),
    ]
    
    results = {}
    
    for exp_name, exp_desc in experiments:
        ckpt_path = f'ckpt/Phase2/{exp_name}_alpha0.1_seed0.pt'
        ckpt = load_checkpoint(ckpt_path)
        if ckpt:
            results[exp_name] = {
                'description': exp_desc,
                'best_val_ppl': ckpt['best_val_ppl'],
                'epoch': ckpt['epoch'],
                'teacher_params': ckpt['teacher_params'],
                'student_params': ckpt['student_params'],
                'size_ratio': ckpt['size_ratio'],
                'args': ckpt['args']
            }
            print(f"{exp_desc}:")
            print(f"  Teacher params: {ckpt['teacher_params']:,}")
            print(f"  Student params: {ckpt['student_params']:,}")
            print(f"  Size ratio (T/S): {ckpt['size_ratio']:.2f}x")
            print(f"  Best Student PPL: {ckpt['best_val_ppl']:.2f}")
            print(f"  Achieved at epoch: {ckpt['epoch']}")
            print()
    
    # Analysis
    if len(results) >= 2:
        print("-" * 80)
        print("COMPARATIVE ANALYSIS:")
        print("-" * 80)
        print()
        
        # Sort by student PPL
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_ppl'])
        
        print("Ranking by Student Performance:")
        for rank, (exp_name, data) in enumerate(sorted_results, 1):
            print(f"{rank}. {data['description']}")
            print(f"   Student PPL: {data['best_val_ppl']:.2f}")
            print(f"   Size Ratio: {data['size_ratio']:.2f}x")
            print()
        
        # Find best configuration
        best_exp = sorted_results[0]
        print(f"✓ Best Configuration: {best_exp[1]['description']}")
        print(f"  Achieves {best_exp[1]['best_val_ppl']:.2f} PPL")
        print()
    
    # Key insights
    print("-" * 80)
    print("KEY INSIGHTS:")
    print("-" * 80)
    print()
    
    if len(results) >= 3:
        # Compare large vs medium teacher
        if 'Tlarge_Ssmall' in results and 'Tmedium_Ssmall' in results:
            large_ppl = results['Tlarge_Ssmall']['best_val_ppl']
            medium_ppl = results['Tmedium_Ssmall']['best_val_ppl']
            
            print("1. TEACHER CAPACITY EFFECT:")
            if large_ppl < medium_ppl:
                print("   ✓ Larger teacher helps student generalization")
                print(f"   → {medium_ppl - large_ppl:.2f} PPL improvement")
            else:
                print("   ⚠ Medium teacher sufficient for this task")
                print(f"   → No benefit from larger teacher")
            print()
        
        # Compare different student sizes
        if 'Tsmall_Ssmall' in results and 'Tmedium_Ssmall' in results:
            similar_ppl = results['Tsmall_Ssmall']['best_val_ppl']
            different_ppl = results['Tmedium_Ssmall']['best_val_ppl']
            
            print("2. SIZE GAP EFFECT:")
            if different_ppl < similar_ppl:
                print("   ✓ Knowledge distillation from larger teacher beneficial")
                print(f"   → {similar_ppl - different_ppl:.2f} PPL improvement")
            else:
                print("   ⚠ Similar-sized models can collaborate effectively")
            print()
    
    print("3. IMITABILITY:")
    print("   - KL(Teacher→Student): Measures how well student can imitate")
    print("   - Lower = Better generalization")
    print("   - Affected by capacity gap")
    print()
    
    print("4. LEARNABILITY:")
    print("   - Larger teachers provide richer knowledge")
    print("   - But may be harder for small students to imitate")
    print("   - Trade-off between capacity and transferability")
    print()
    
    # Recommendations
    print("-" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    if len(results) >= 3:
        best_config = sorted_results[0]
        print(f"✓ Use {best_config[1]['description']}")
        print(f"  → Optimal balance between teacher capacity and student learnability")
        print()
        
        # Specific recommendations
        if 'large' in best_config[0].lower():
            print("  Finding: Larger teachers are worth the complexity")
            print("  Action: Invest in teacher capacity for better student performance")
        elif 'medium' in best_config[0].lower():
            print("  Finding: Medium-sized teachers offer best trade-off")
            print("  Action: Don't need very large teachers for this task")
        else:
            print("  Finding: Similar-sized models collaborate well")
            print("  Action: LoT works even without large capacity gap")
    print()


if __name__ == '__main__':
    analyze_phase2()
    print("="*80)
    print("Analysis complete!")
    print("="*80)
