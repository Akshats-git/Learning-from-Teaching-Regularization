#!/usr/bin/env python3
"""
Visualization script for LoT assessment results
Generates publication-quality plots for report
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
        
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def load_checkpoint_safe(path):
    """Safely load checkpoint"""
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location='cpu')
    except:
        return None


def plot_phase1_comparison():
    """Plot Phase 1 feedback variant comparison"""
    print("Generating Phase 1 comparison plot...")
    
    variants = ['Positive', 'Negative', 'Mixed']
    variant_names = ['positive', 'negative', 'mixed']
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
    
    ppls = []
    epochs = []
    
    for name in variant_names:
        ckpt = load_checkpoint_safe(f'ckpt/Phase1/{name}_alpha0.1_seed0.pt')
        if ckpt:
            ppls.append(ckpt['best_val_ppl'])
            epochs.append(ckpt['epoch'])
        else:
            ppls.append(None)
            epochs.append(None)
    
    # Filter out None values
    valid_data = [(v, p, e, c) for v, p, e, c in zip(variants, ppls, epochs, colors) if p is not None]
    
    if not valid_data:
        print("⚠ No Phase 1 checkpoints found. Run experiments first.")
        return
    
    variants, ppls, epochs, colors = zip(*valid_data)
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PPL comparison
    bars1 = ax1.bar(variants, ppls, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Validation Perplexity')
    ax1.set_title('Phase 1: Feedback Variant Performance')
    ax1.set_ylim(min(ppls) - 5, max(ppls) + 5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, ppl in zip(bars1, ppls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Convergence speed
    bars2 = ax2.bar(variants, epochs, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Epochs to Convergence')
    ax2.set_title('Phase 1: Convergence Speed')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, epoch in zip(bars2, epochs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{epoch}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase1_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase1_comparison.png")
    
    # Create detailed comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_idx = np.argmin(ppls)
    highlight_colors = ['lightgray'] * len(ppls)
    highlight_colors[best_idx] = colors[best_idx]
    
    bars = ax.barh(variants, ppls, color=highlight_colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Validation Perplexity (lower is better)')
    ax.set_title('Phase 1: Which Feedback Type Works Best?')
    ax.invert_yaxis()
    
    # Highlight best
    ax.axvline(ppls[best_idx], color='green', linestyle='--', alpha=0.5, label='Best Performance')
    
    # Add value labels
    for i, (bar, ppl) in enumerate(zip(bars, ppls)):
        width = bar.get_width()
        label = f'{ppl:.2f}'
        if i == best_idx:
            label += ' [BEST]'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                label,
                ha='left', va='center', fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig('phase1_best.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase1_best.png")


def plot_phase2_comparison():
    """Plot Phase 2 size ratio comparison"""
    print("Generating Phase 2 comparison plot...")
    
    configs = [
        ('Tlarge_Ssmall', 'Large→Small', '#e74c3c'),
        ('Tmedium_Ssmall', 'Medium→Small', '#f39c12'),
        ('Tsmall_Ssmall', 'Small→Similar', '#3498db'),
    ]
    
    labels = []
    ppls = []
    ratios = []
    teacher_params = []
    student_params = []
    colors_used = []
    
    for name, label, color in configs:
        ckpt = load_checkpoint_safe(f'ckpt/Phase2/{name}_alpha0.1_seed0.pt')
        if ckpt:
            labels.append(label)
            ppls.append(ckpt['best_val_ppl'])
            ratios.append(ckpt['size_ratio'])
            teacher_params.append(ckpt['teacher_params'] / 1e6)  # In millions
            student_params.append(ckpt['student_params'] / 1e6)
            colors_used.append(color)
    
    if not labels:
        print("⚠ No Phase 2 checkpoints found. Run experiments first.")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. PPL comparison
    ax = axes[0, 0]
    bars = ax.bar(labels, ppls, color=colors_used, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Student Validation Perplexity')
    ax.set_title('Phase 2: Teacher-Student Size Ratio Performance')
    ax.set_ylim(min(ppls) - 3, max(ppls) + 3)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, ppl in zip(bars, ppls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Size ratio vs PPL
    ax = axes[0, 1]
    ax.scatter(ratios, ppls, c=colors_used, s=200, alpha=0.8, edgecolors='black', linewidths=2)
    for i, (ratio, ppl, label) in enumerate(zip(ratios, ppls, labels)):
        ax.annotate(label, (ratio, ppl), xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.set_xlabel('Size Ratio (Teacher/Student)')
    ax.set_ylabel('Student Validation Perplexity')
    ax.set_title('Phase 2: Capacity Gap vs Performance')
    ax.grid(alpha=0.3)
    
    # 3. Parameter count comparison
    ax = axes[1, 0]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, teacher_params, width, label='Teacher', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, student_params, width, label='Student', color='#3498db', alpha=0.8)
    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Phase 2: Model Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Efficiency: PPL per student parameter
    ax = axes[1, 1]
    efficiency = [ppl / sp for ppl, sp in zip(ppls, student_params)]
    bars = ax.bar(labels, efficiency, color=colors_used, alpha=0.8, edgecolor='black')
    ax.set_ylabel('PPL per Million Student Parameters')
    ax.set_title('Phase 2: Parameter Efficiency')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('phase2_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase2_comparison.png")
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_idx = np.argmin(ppls)
    highlight_colors = ['lightgray'] * len(ppls)
    highlight_colors[best_idx] = colors_used[best_idx]
    
    bars = ax.barh(labels, ppls, color=highlight_colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Validation Perplexity (lower is better)')
    ax.set_title('Phase 2: Optimal Teacher-Student Size Ratio')
    ax.invert_yaxis()
    
    # Add ratio labels
    for i, (bar, ppl, ratio) in enumerate(zip(bars, ppls, ratios)):
        width = bar.get_width()
        label = f'{ppl:.2f} (ratio: {ratio:.1f}x)'
        if i == best_idx:
            label += ' [BEST]'
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                label,
                ha='left', va='center', fontweight='bold' if i == best_idx else 'normal')
    
    plt.tight_layout()
    plt.savefig('phase2_best.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase2_best.png")


def plot_combined_summary():
    """Create combined summary plot"""
    print("Generating combined summary...")
    
    # Load all data
    phase1_data = {}
    for variant in ['positive', 'negative', 'mixed']:
        ckpt = load_checkpoint_safe(f'ckpt/Phase1/{variant}_alpha0.1_seed0.pt')
        if ckpt:
            phase1_data[variant] = ckpt['best_val_ppl']
    
    phase2_data = {}
    for config in ['Tlarge_Ssmall', 'Tmedium_Ssmall', 'Tsmall_Ssmall']:
        ckpt = load_checkpoint_safe(f'ckpt/Phase2/{config}_alpha0.1_seed0.pt')
        if ckpt:
            phase2_data[config] = ckpt['best_val_ppl']
    
    if not phase1_data and not phase2_data:
        print("⚠ No checkpoints found for combined summary.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Phase 1 summary
    if phase1_data:
        ax = axes[0]
        variants = list(phase1_data.keys())
        ppls = list(phase1_data.values())
        colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(variants)]
        
        bars = ax.bar([v.capitalize() for v in variants], ppls, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Validation PPL')
        ax.set_title('Phase 1: Best Feedback Type')
        ax.grid(axis='y', alpha=0.3)
        
        best_idx = np.argmin(ppls)
        for i, (bar, ppl) in enumerate(zip(bars, ppls)):
            height = bar.get_height()
            label = f'{ppl:.2f}'
            if i == best_idx:
                label += ' [BEST]'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom',
                    fontweight='bold' if i == best_idx else 'normal')
    
    # Phase 2 summary
    if phase2_data:
        ax = axes[1]
        config_labels = {
            'Tlarge_Ssmall': 'Large→Small',
            'Tmedium_Ssmall': 'Medium→Small',
            'Tsmall_Ssmall': 'Small→Similar'
        }
        labels = [config_labels[k] for k in phase2_data.keys()]
        ppls = list(phase2_data.values())
        colors = ['#e74c3c', '#f39c12', '#3498db'][:len(labels)]
        
        bars = ax.bar(labels, ppls, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Validation PPL')
        ax.set_title('Phase 2: Best Size Ratio')
        ax.grid(axis='y', alpha=0.3)
        
        best_idx = np.argmin(ppls)
        for i, (bar, ppl) in enumerate(zip(bars, ppls)):
            height = bar.get_height()
            label = f'{ppl:.2f}'
            if i == best_idx:
                label += ' [BEST]'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label, ha='center', va='bottom',
                    fontweight='bold' if i == best_idx else 'normal')
    
    plt.tight_layout()
    plt.savefig('combined_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: combined_summary.png")


def main():
    """Generate all visualizations"""
    print("="*80)
    print("GENERATING ASSESSMENT VISUALIZATIONS")
    print("="*80)
    print()
    
    # Create output directory if needed
    os.makedirs('figures', exist_ok=True)
    
    # Generate Phase 1 plots
    plot_phase1_comparison()
    print()
    
    # Generate Phase 2 plots
    plot_phase2_comparison()
    print()
    
    # Generate combined summary
    plot_combined_summary()
    print()
    
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print()
    print("Generated files:")
    print("  - phase1_comparison.png (Feedback variants comparison)")
    print("  - phase1_best.png (Best feedback type)")
    print("  - phase2_comparison.png (Size ratios comparison)")
    print("  - phase2_best.png (Best size ratio)")
    print("  - combined_summary.png (Overall summary)")
    print()
    print("Use these plots in your assessment report!")


if __name__ == '__main__':
    main()
