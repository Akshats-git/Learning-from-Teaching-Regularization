# Learning-from-Teaching (LoT) Regularization

[![LICENSE](https://img.shields.io/badge/LICENSE-MIT-4caf50.svg)](https://github.com/jincan333/LoT)
[![arXiv](https://img.shields.io/badge/arXiv-2402.02769-b31b1b.svg)](https://arxiv.org/abs/2402.02769)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange.svg)](https://pytorch.org/)

A comprehensive framework for **Learning-from-Teaching (LoT)** regularization research, combining teacher-student learning with emergent communication. This project explores how bidirectional knowledge transfer between teacher and student models can improve generalization across multiple domains: Natural Language Processing (NLP), Computer Vision (CV), and Reinforcement Learning (RL).

> **Paper**: [Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate](https://arxiv.org/pdf/2402.02769.pdf)  
> **Authors**: [Can Jin](https://jincan333.github.io/), Tong Che, [Hongwu Peng](https://harveyp123.github.io/), Yiyuan Li, [Marco Pavone](https://web.stanford.edu/~pavone/index.html)

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture & Flow Diagrams](#architecture--flow-diagrams)
- [Research Directions](#research-directions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducibility Guide](#reproducibility-guide)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

Generalization remains a central challenge in machine learning. In this work, we propose *Learning from Teaching* (**LoT**), a novel regularization technique for deep neural networks to enhance generalization. Inspired by the human ability to capture concise and abstract patterns, we hypothesize that **generalizable correlations are expected to be easier to teach**. LoT operationalizes this concept to improve the generalization of the main model with auxiliary student learners.

### Key Innovation

Traditional knowledge distillation is unidirectional: teacher → student. LoT introduces **bidirectional learning**:
- **Forward KL**: Student learns to imitate teacher's predictions
- **Reverse KL**: Teacher learns to be more "teachable"
- **Emergent Language** : Teacher encodes knowledge into symbolic messages that student decodes

### Advantages

✅ **Better Generalization**: Both teacher and student achieve lower perplexity/error rates  
✅ **Flexible Architecture**: Works across domains (NLP, CV, RL)  
✅ **Interpretable Communication**: Emergent messages reveal what the teacher "teaches"  
✅ **Curriculum Learning**: Gradual integration of messages for stable training  

---

## 📁 Project Structure

```
LoT/
├── model/                          # Model architectures
│   ├── rnn.py                     # LSTM/GRU language models
│   ├── mem_transformer.py         # Transformer-XL implementation
│   ├── preresnet.py               # ResNet for image classification
│   ├── message_channel.py         # Emergent language communication (Phase 4)
│   └── emergent_language_models.py # Teacher/Student with message encoding/decoding
│
├── trainer/                        # Training scripts for different experiments
│   ├── Lot_Emergent.py            # LoT + Emergent Language
│   ├── feedback_variants.py       # Positive/Negative/Mixed feedback
│   ├── size_ratios.py             # Teacher-Student size ratios
│   ├── alternative_metrics.py     # KL/JS/L2/Cosine teaching metrics
│   └── image_classification.py    # CV: Image classification with LoT
│
├── analysis/                       # Analysis and visualization scripts
│   ├── analyze_phase4_full.py     # Emergent language analysis
│   ├── analyze_phase1.py          # Feedback variants analysis
│   ├── analyze_phase2.py          # Size ratio analysis
│   └── visualize_results.py       # General result visualization
│
├── utils/                          # Utility functions
│   ├── data_utils.py              # Data loading utilities
│   ├── corpus.py                  # NLP corpus handling (PTB, WikiText)
│   ├── vocabulary.py              # Vocabulary management
│   └── exp_utils.py               # Experiment utilities
│
├── run/                            # Shell scripts for experiments
│   ├── Lot_Emergent.sh            # Run emergent language experiments
│   ├── feedback_variants.sh       # Run feedback variant experiments
│   ├── size_ratios.sh             # Run size ratio experiments
│   └── getdata.sh                 # Download datasets
│
├── data/                           # Datasets (auto-downloaded)
│   ├── ptb/                       # Penn TreeBank
│   ├── wikitext-103/              # WikiText-103
│   └── cifar-10-batches-py/       # CIFAR-10
│
├── ckpt/                           # Model checkpoints
│   ├── Phase1/                    # Feedback variant checkpoints
│   ├── Phase2/                    # Size ratio checkpoints
│   ├── Phase3/                    # Alternative metric checkpoints
│   └── Phase4/                    # Emergent language checkpoints
│
├── logs/                           # Training logs
└── result/                         # Analysis results and visualizations
```

---

## 🏗️ Architecture & Flow Diagrams

### Core LoT Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Learning-from-Teaching (LoT) Framework               │
└─────────────────────────────────────────────────────────────────────┘

Input (x) ───┬───► Teacher Model ───► Teacher Prediction (ŷ_t)
             │                              │
             │                              │
             │                              ▼
             │                    ┌─────────────────────┐
             │                    │   LoT Loss Engine   │
             │                    │  ─────────────────  │
             └───► Student Model ─┤  • CE Loss (both)   │
                        │         │  • KL(ŷ_s || ŷ_t)   │
                        │         │  • KL(ŷ_t || ŷ_s)   │
                        ▼         └─────────────────────┘
              Student Prediction                │
                    (ŷ_s)                       │
                        │                       │
                        └───────────────────────┘
                                    │
                                    ▼
                            Ground Truth (y)
```

**Key Components:**
- **Cross-Entropy (CE) Loss**: Both models learn from ground truth
- **Forward KL** `KL(ŷ_s || ŷ_t)`: Student imitates teacher
- **Reverse KL** `KL(ŷ_t || ŷ_s)`: Teacher learns to be teachable
- **Regularization Weight α**: Controls LoT strength

### LoT + Emergent Language Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                LoT with Emergent Communication                        │
└──────────────────────────────────────────────────────────────────────┘

Input (x) ───┬───► Teacher Encoder ───► Message (m) ───┐
             │           │                              │
             │           │                              │
             │           ▼                              ▼
             │    Teacher Predictor          Student Decoder ───► ŷ_s
             │           │                              │
             │           ▼                              │
             │    Teacher Prediction (ŷ_t)             │
             │           │                              │
             │           └──────────┬───────────────────┘
             │                      │
             └──────────────────────┤
                                    ▼
                          ┌─────────────────────┐
                          │  LoT + Language     │
                          │  Loss Computation   │
                          │  ─────────────────  │
                          │  • CE Loss (both)   │
                          │  • KL(ŷ_s || ŷ_t)   │
                          │  • KL(ŷ_t || ŷ_s)   │
                          │  • Entropy Reg      │
                          │  • Compositionality │
                          └─────────────────────┘
```

**Message Channel Details:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Message Channel (Discrete)                    │
└─────────────────────────────────────────────────────────────────┘

Teacher Hidden State (h_t)
         │
         ▼
    Feature Projection
         │
         ▼
    Message Encoder
         │
         ▼
    Logits: [batch, msg_length, vocab_size]
         │
         ▼
    Gumbel-Softmax Sampling
         │
         ▼
    Message: [batch, msg_length, vocab_size]
         │
         ├───► (Hard symbols for analysis)
         │
         └───► Student Message Decoder
                      │
                      ▼
              Fusion with Student Hidden
                      │
                      ▼
              Enhanced Student Prediction
```

**Training Flow with Curriculum Learning:**

```
Epoch 0  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►  Epoch 60
         
Message   0% ────────────► 10% ─────► 50% ───► 100%
Weight    (Pure LoT)     (Warmup)   (Ramp)  (Full)
          
Phase:    [────── LoT Foundation ──────][─ Message Integration ─]
          Epoch 0-30              Epoch 30-50        Epoch 50-60
```

---

## 🔬 Research Directions

### Feedback Balance Variants

**Goal**: Compare positive, negative, and mixed teaching feedback

| Variant | Student → Teacher | Teacher → Student | Use Case |
|---------|-------------------|-------------------|----------|
| **Positive** | ✅ KL(ŷ_t \|\| ŷ_s) | ✅ KL(ŷ_s \|\| ŷ_t) | Bidirectional learning |
| **Negative** | ❌ None | ✅ KL(ŷ_s \|\| ŷ_t) | Traditional distillation |
| **Mixed** | ✅ Half strength | ✅ Full strength | Asymmetric feedback |

**Key Finding**: Positive (bidirectional) feedback achieves best generalization

---

### Teacher-Student Size Ratios

**Goal**: Study impact of model capacity ratios

| Configuration | Teacher Size | Student Size | Capacity Ratio |
|---------------|--------------|--------------|----------------|
| **Small-Small** | 650 hidden | 650 hidden | 1:1 |
| **Medium-Small** | 1300 hidden | 650 hidden | 2:1 |
| **Large-Small** | 2600 hidden | 650 hidden | 4:1 |

**Key Finding**: Larger teachers provide richer teaching signals, but require careful tuning

---

### Alternative Teaching Metrics

**Goal**: Explore different divergence measures for teaching

| Metric | Formula | Properties |
|--------|---------|------------|
| **KL Divergence** | `KL(P \|\| Q)` | Asymmetric, mode-seeking |
| **JS Divergence** | `½KL(P\|\|M) + ½KL(Q\|\|M)` | Symmetric, bounded |
| **L2 Distance** | `\|\|P - Q\|\|²` | Euclidean, simple |
| **Cosine** | `1 - cos(P,Q)` | Angle-based, scale-invariant |

**Key Finding**: KL divergence remains most effective for probability distributions

---

### LoT + Emergent Language ⭐

**Goal**: Enable structured knowledge transfer through emergent communication

**Components:**
1. **Message Channel**: Discrete symbolic communication (Gumbel-Softmax)
   - Vocabulary size: 32-100 symbols
   - Message length: 8-20 tokens
   - Temperature annealing: 1.0 → 0.5

2. **Language Regularization**:
   - **Entropy**: Encourage vocabulary diversity
   - **Compositionality**: Promote structured messages
   - **Curriculum Learning**: Gradual message integration

3. **Evaluation Metrics**:
   - Perplexity (PPL): Lower is better
   - Vocabulary coverage: % of symbols used
   - Positional entropy: Measures compositionality
   - Topographic similarity: Input-message correlation

**Expected Results:**
- **Teacher PPL**: ~110-130 (similar to baseline)
- **Student PPL**: ~120-140 (5-15 points better than baseline ~145)
- **Message benefit**: Structured knowledge transfer

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (for larger datasets)

### Step 1: Clone Repository

```bash
git clone https://github.com/rajeev-sr/Learning-from-Teaching-Regularization.git
cd Learning-from-Teaching-Regularization
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n lot python=3.9
conda activate lot

# Or using venv
python3 -m venv env
source env/bin/activate  # Linux/Mac
# env\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch>=1.8.0`: PyTorch framework
- `numpy>=1.20.0`: Numerical computing
- `matplotlib>=3.5.0`: Visualization
- `wandb`: Experiment tracking (optional)
- `datasets`: Hugging Face datasets

### Step 4: Download Datasets

```bash
bash run/getdata.sh
```

This downloads:
- **Penn TreeBank (PTB)**: ~1M tokens for language modeling
- **WikiText-103**: ~100M tokens for language modeling
- **CIFAR-10**: 60K images for image classification

### Step 5: Configure WANDB (Optional)

Configure WANDB USER_NAME and API_KEY in the `key.config` file for experiment tracking.

---

## ⚡ Quick Start

### Example 1: Train LoT with Emergent Language

```bash
# Run with default configuration
bash run/Lot_Emergent.sh
```

**What this does:**
1. Trains teacher model with message encoder
2. Trains student model with message decoder
3. Applies LoT regularization + language regularization
4. Uses curriculum learning (gradual message integration)
5. Saves checkpoint to `ckpt/Phase4/Lot_Emergent_ptb.pt`
6. Logs results to `logs/Phase4/Lot_Emergent_ptb.log`


### Example 2: Analyze Emergent Language

```bash
python analysis/analyze_phase4_full.py \
    --checkpoint ckpt/Phase4/Lot_Emergent_ptb.pt \
    --output_dir result/Phase4
```

**Generated visualizations:**
- `positional_entropy.png`: Compositionality analysis
- `symbol_frequency.png`: Vocabulary usage distribution
- `vocabulary_usage.png`: Coverage statistics
- `analysis_summary.txt`: Detailed metrics

### Example 3: Train Baseline LoT (No Messages)

```bash
python trainer/feedback_variants.py \
    --data ptb \
    --feedback positive \
    --alpha 1.0 \
    --epochs 60 \
    --save ckpt/Phase1/baseline.pt
```

---

## 📖 Reproducibility Guide

### Experiment 1: Feedback Variants

**Objective**: Compare positive, negative, and mixed feedback strategies

```bash
# 1. Positive feedback (bidirectional)
bash run/feedback_variants.sh positive

# 2. Negative feedback (unidirectional)
bash run/feedback_variants.sh negative

# 3. Mixed feedback (asymmetric)
bash run/feedback_variants.sh mixed

# 4. Analyze results
python analysis/analyze_phase1.py --result_dir result/feedback_variants
```


---

### Experiment 2: LoT with Emergent Language ⭐

**Objective**: Achieve 5-15 PPL improvement over baseline via emergent communication

#### Step 1: Train Baseline LoT (for comparison)

```bash
python trainer/feedback_variants.py \
    --data ptb \
    --feedback positive \
    --alpha 1.0 \
    --epochs 60 \
    --save ckpt/baseline_lot.pt \
    2>&1 | tee logs/baseline_lot.log
```

#### Step 2: Train LoT with Emergent Messages

```bash
bash run/Lot_Emergent.sh
```

**Configuration Details:**
```bash
Dataset:              PTB
Message vocab:        32 symbols
Message length:       8 tokens
LoT alpha:            1.0
Language reg (β):     0.0001
Epochs:               60
Batch size:           20
Learning rate:        30
```

#### Step 3: Analyze Emergent Language

```bash
python analysis/analyze_phase4_full.py \
    --checkpoint ckpt/Phase4/Lot_Emergent_ptb.pt \
    --output_dir result/Phase4 \
    --data ptb \
    --batch_size 10
```

**Analysis Outputs:**
1. **Compositionality Score**: Measures if different positions encode different meanings
2. **Symbol-Meaning Correlation**: Maps symbols to semantic clusters
3. **Vocabulary Coverage**: % of vocabulary actively used
4. **Positional Entropy**: How much information each position carries

#### Step 4: Compare Results

```bash
# Extract final PPL from logs
echo "=== Baseline LoT ==="
grep "Teacher PPL" logs/baseline_lot.log | tail -1
grep "Student PPL" logs/baseline_lot.log | tail -1

echo "=== LoT + Emergent Language ==="
grep "Teacher PPL" logs/Phase4/Lot_Emergent_ptb.log | tail -1
grep "Student PPL" logs/Phase4/Lot_Emergent_ptb.log | tail -1
```

## 📊 Datasets

### Penn TreeBank (PTB)

- **Size**: ~1M tokens
- **Vocabulary**: ~10K words
- **Splits**: Train (42K sentences), Valid (3.3K), Test (3.7K)
- **Use Case**: Standard language modeling benchmark

### WikiText-103

- **Size**: ~103M tokens
- **Vocabulary**: ~268K words
- **Splits**: Train (1.8M sentences), Valid (3.7K), Test (4.3K)
- **Use Case**: Large-scale language modeling

### CIFAR-10

- **Size**: 60K images (32×32 RGB)
- **Classes**: 10 (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- **Splits**: Train (50K), Test (10K)
- **Use Case**: Image classification with LoT


## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for contribution:**
- New message channel architectures (e.g., Transformer-based)
- Additional datasets (e.g., GLUE, ImageNet)
- Improved language metrics
- Multi-modal emergent communication

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{jin2024learning,
  title={Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate},
  author={Jin, Can and Che, Tong and Peng, Hongwu and Li, Yiyuan and Pavone, Marco},
  journal={arXiv preprint arXiv:2402.02769},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team**: For the deep learning framework
- **Penn TreeBank & WikiText**: For language modeling datasets
- **OpenAI Gym**: For reinforcement learning environments
- **Community Contributors**: For valuable feedback and improvements

---

## 📞 Contact

For questions, issues, or collaborations:

- **GitHub Issues**: [Create an issue](https://github.com/rajeev-sr/Learning-from-Teaching-Regularization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rajeev-sr/Learning-from-Teaching-Regularization/discussions)

---

**⭐ Star this repository if you find it useful!**

**🐛 Found a bug? [Report it here](https://github.com/rajeev-sr/Learning-from-Teaching-Regularization/issues)**

**💡 Have a suggestion? [Open a discussion](https://github.com/rajeev-sr/Learning-from-Teaching-Regularization/discussions)**

---

## Citation
We encourage citing our paper if our findings are used in your research.
```bibtex
@misc{jin2024learning,
      title={Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate}, 
      author={Can Jin and Tong Che and Hongwu Peng and Yiyuan Li and Marco Pavone},
      year={2024},
      eprint={2402.02769},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
