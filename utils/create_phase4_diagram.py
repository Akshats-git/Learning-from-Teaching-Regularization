#!/usr/bin/env python3
"""
Create architecture diagram for Phase 4
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_architecture_diagram():
    """Create a visual diagram of Phase 4 architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Phase 4: LoT + Emergent Language Architecture', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((6, 8.5), 2, 0.5, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 8.75, 'Input (x)', ha='center', va='center', fontweight='bold')
    
    # Teacher branch
    # Teacher RNN
    teacher_rnn = FancyBboxPatch((1, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 edgecolor='darkblue', facecolor='lightsteelblue', linewidth=2)
    ax.add_patch(teacher_rnn)
    ax.text(2.25, 7.5, 'Teacher RNN', ha='center', va='top', fontweight='bold', fontsize=10)
    ax.text(2.25, 7.2, '(LSTM/GRU)', ha='center', va='top', fontsize=8, style='italic')
    ax.text(2.25, 6.9, 'Base Model', ha='center', va='center', fontsize=8)
    
    # Message Encoder
    msg_encoder = FancyBboxPatch((1, 5), 2.5, 1, boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(msg_encoder)
    ax.text(2.25, 5.8, 'Message', ha='center', va='top', fontweight='bold', fontsize=10)
    ax.text(2.25, 5.5, 'Encoder', ha='center', va='center', fontsize=10)
    ax.text(2.25, 5.2, '(Gumbel/Continuous)', ha='center', va='bottom', fontsize=7, style='italic')
    
    # Teacher Prediction
    teacher_pred = FancyBboxPatch((1, 3.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                                  edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(teacher_pred)
    ax.text(2.25, 3.9, 'Teacher Prediction', ha='center', va='center', fontweight='bold', fontsize=9)
    ax.text(2.25, 3.6, '(ŷ_teacher)', ha='center', va='bottom', fontsize=8, style='italic')
    
    # Student branch
    # Student RNN
    student_rnn = FancyBboxPatch((10.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 edgecolor='darkblue', facecolor='lightsteelblue', linewidth=2)
    ax.add_patch(student_rnn)
    ax.text(11.75, 7.5, 'Student RNN', ha='center', va='top', fontweight='bold', fontsize=10)
    ax.text(11.75, 7.2, '(LSTM/GRU)', ha='center', va='top', fontsize=8, style='italic')
    ax.text(11.75, 6.9, 'Base Model', ha='center', va='center', fontsize=8)
    
    # Message Decoder
    msg_decoder = FancyBboxPatch((10.5, 5), 2.5, 1, boxstyle="round,pad=0.1",
                                 edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(msg_decoder)
    ax.text(11.75, 5.8, 'Message', ha='center', va='top', fontweight='bold', fontsize=10)
    ax.text(11.75, 5.5, 'Decoder', ha='center', va='center', fontsize=10)
    ax.text(11.75, 5.2, '(Fusion Layer)', ha='center', va='bottom', fontsize=7, style='italic')
    
    # Student Prediction
    student_pred = FancyBboxPatch((10.5, 3.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                                  edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(student_pred)
    ax.text(11.75, 3.9, 'Student Prediction', ha='center', va='center', fontweight='bold', fontsize=9)
    ax.text(11.75, 3.6, '(ŷ_student)', ha='center', va='bottom', fontsize=8, style='italic')
    
    # Message Channel
    message_channel = FancyBboxPatch((5.5, 5), 3, 1, boxstyle="round,pad=0.1",
                                     edgecolor='red', facecolor='lightyellow', linewidth=2, linestyle='--')
    ax.add_patch(message_channel)
    ax.text(7, 5.8, 'Message Channel', ha='center', va='top', fontweight='bold', fontsize=10, color='red')
    ax.text(7, 5.5, 'm = [s₁, s₂, ..., sₗ]', ha='center', va='center', fontsize=9, family='monospace')
    ax.text(7, 5.2, 'Discrete/Continuous', ha='center', va='bottom', fontsize=7, style='italic')
    
    # LoT Loss
    lot_loss = FancyBboxPatch((4.5, 1.5), 5, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='darkred', facecolor='mistyrose', linewidth=2)
    ax.add_patch(lot_loss)
    ax.text(7, 2.8, 'LoT Loss Computation', ha='center', va='top', fontweight='bold', fontsize=11)
    ax.text(7, 2.4, 'Teacher: CE + α·KL(ŷ_t || ŷ_s)', ha='center', va='center', fontsize=9, family='monospace')
    ax.text(7, 2.0, 'Student: CE + α·KL(ŷ_s || ŷ_t)', ha='center', va='center', fontsize=9, family='monospace')
    ax.text(7, 1.6, '(Bidirectional teaching)', ha='center', va='bottom', fontsize=7, style='italic')
    
    # Ground Truth
    gt_box = FancyBboxPatch((6, 0.2), 2, 0.5, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(gt_box)
    ax.text(7, 0.45, 'Ground Truth (y)', ha='center', va='center', fontweight='bold')
    
    # Arrows
    # Input to Teacher
    arrow1 = FancyArrowPatch((6.5, 8.5), (2.25, 7.7), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Input to Student
    arrow2 = FancyArrowPatch((7.5, 8.5), (11.75, 7.7),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Teacher RNN to Message Encoder
    arrow3 = FancyArrowPatch((2.25, 6.5), (2.25, 6.0),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
    ax.add_patch(arrow3)
    
    # Message Encoder to Channel
    arrow4 = FancyArrowPatch((3.5, 5.5), (5.5, 5.5),
                            arrowstyle='->', mutation_scale=20, linewidth=3, color='red')
    ax.add_patch(arrow4)
    ax.text(4.5, 5.7, 'encode', ha='center', va='bottom', fontsize=8, color='red', style='italic')
    
    # Channel to Message Decoder
    arrow5 = FancyArrowPatch((8.5, 5.5), (10.5, 5.5),
                            arrowstyle='->', mutation_scale=20, linewidth=3, color='red')
    ax.add_patch(arrow5)
    ax.text(9.5, 5.7, 'decode', ha='center', va='bottom', fontsize=8, color='red', style='italic')
    
    # Message Decoder to Student RNN
    arrow6 = FancyArrowPatch((11.75, 6.0), (11.75, 6.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
    ax.add_patch(arrow6)
    ax.text(12.2, 6.25, 'fuse', ha='left', va='center', fontsize=7, style='italic')
    
    # Teacher RNN to Prediction
    arrow7 = FancyArrowPatch((2.25, 6.5), (2.25, 4.3),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow7)
    
    # Student RNN to Prediction
    arrow8 = FancyArrowPatch((11.75, 6.5), (11.75, 4.3),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
    ax.add_patch(arrow8)
    
    # Predictions to Loss
    arrow9 = FancyArrowPatch((2.25, 3.5), (5, 2.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='darkgreen')
    ax.add_patch(arrow9)
    
    arrow10 = FancyArrowPatch((11.75, 3.5), (9, 2.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='darkgreen')
    ax.add_patch(arrow10)
    
    # Ground truth to Loss
    arrow11 = FancyArrowPatch((7, 0.7), (7, 1.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow11)
    
    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc='lightsteelblue', ec='darkblue', label='Neural Network'),
        mpatches.Rectangle((0, 0), 1, 1, fc='plum', ec='purple', label='Message Processing'),
        mpatches.Rectangle((0, 0), 1, 1, fc='lightgreen', ec='darkgreen', label='Predictions'),
        mpatches.Rectangle((0, 0), 1, 1, fc='lightyellow', ec='red', label='Communication Channel'),
        mpatches.Rectangle((0, 0), 1, 1, fc='mistyrose', ec='darkred', label='Loss Computation'),
        mlines.Line2D([], [], color='red', linewidth=3, label='Message Flow'),
        mlines.Line2D([], [], color='green', linewidth=2, label='Prediction Flow'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
    
    # Add annotations
    ax.text(0.5, 9, 'Key Features:', fontsize=10, fontweight='bold')
    ax.text(0.5, 8.6, '• Bidirectional teaching (LoT)', fontsize=8)
    ax.text(0.5, 8.3, '• Message-based communication', fontsize=8)
    ax.text(0.5, 8.0, '• Emergent language structure', fontsize=8)
    ax.text(0.5, 7.7, '• Discrete or continuous messages', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/Phase4/architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Architecture diagram saved to: figures/Phase4/architecture_diagram.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('figures/Phase4', exist_ok=True)
    create_architecture_diagram()
    print("\nArchitecture diagram created successfully!")
    print("View it at: figures/Phase4/architecture_diagram.png")
