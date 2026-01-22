# ISYE URP: PPO Ablation Study with 12-Anchor Fixture Configuration

Starting to add reports here!

## 📋 Project Overview

This repository contains the implementation and experimental evaluation of two major research tasks:

### **Task 7.1A: PPO Ablation Study**
Rigorous comparison of PPO training from random initialization against PPO warm-started with a distilled student policy.

**Key Results:**
- **PPO-from-scratch**: RMSE $0.1934 \pm 0.0391$, Return $+0.0879 \pm 0.1676$
- **PPO-teacher-init**: RMSE $0.1995 \pm 0.0232$, Return $+0.1057 \pm 0.1236$
- **Paired Delta**: RMSE $\Delta = +0.0061$ (not significant), Return $\Delta = +0.0178$ (marginal)

### **Task 7.2B: 12-Anchor Configuration**
Extended discrete fixture environment from 9 to 12 anchor locations, incorporating both boundary and interior lattice points.

**Configuration:**
- **Boundary anchors** (8): Corners + edge midpoints
- **Interior anchors** (4): Lattice points for fine-tuning
- **Observation space**: $\mathbb{R}^{15}$ (12 coordinates + 3 global stats)
- **Action space**: 13 discrete actions (12 anchors + NOOP)

## 🗂️ Repository Structure

### Core Implementation Files

```
env_fixture.py                  # DiscreteFixtureEnv with 12-anchor support
student_policy_network.py       # Student policy architecture (15→64→64→12 MLP)
ppo_finetune.py                 # PPO training with optional warm-start
teacher_iface.py                # Teacher interface with deformation-driven heuristics
distill_student.py              # Knowledge distillation module (KL minimization)
collect_dataset.py              # Teacher data collection pipeline
parse_teacher_reply.py          # Teacher output parsing and action formatting
translator.py                   # Context translation for teacher queries
```

### Experimental & Validation Files

```
eval_ppo_ablation.py            # Main ablation study driver (250 lines)
convert_to_12_anchors.py        # Full 12-anchor conversion pipeline (150 lines)
validate_12_anchor_setup.py     # Comprehensive 8-test validation suite (280 lines)
test_implementation.py          # Integration tests (220 lines)
eval_policy.py                  # Policy evaluation and visualization
```

### Generated Artifacts

```
teacher_data_12_X.npy           # Teacher dataset observations (600, 15)
teacher_data_12_P.npy           # Teacher dataset actions (600, 12)
student_policy_init.pt          # Distilled student policy checkpoint (26.4 KB)
ppo_scratch.pt                  # PPO-from-scratch final policy
ppo_teacher_init.pt             # PPO-teacher-init final policy
ablation_results.json           # Complete statistical results
```

### Documentation

```
ISYE_URP_Training_Report.tex    # Comprehensive LaTeX report (Overleaf-compatible)
QUICKSTART.md                   # Quick reference guide
IMPLEMENTATION_TASKS_7.1A_7.2B.md  # Detailed implementation spec
TRAINING_RUN_REPORT.md          # Training analysis and findings
TRAINING_COMPLETE.md            # Training completion summary
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+ with venv
python --version  # Should be 3.11 or later
```

### Installation

1. **Clone and navigate to repository:**
```bash
cd /path/to/ISYE_URP
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch 2.0+ (`torch`)
- NumPy 2.4+ (`numpy`)
- Matplotlib 3.10+ (`matplotlib`)
- Pandas 3.0+ (`pandas`)

### Verify Installation

```bash
python test_implementation.py --quick
```

Expected output: ✓ ALL CRITICAL TESTS PASSED

## Running Experiments

### Option 1: Quick Full Pipeline (30 minutes)

```bash
# Step 1: Convert to 12-anchor configuration
python convert_to_12_anchors.py

# Step 2: Validate setup
python validate_12_anchor_setup.py

# Step 3: Run ablation study
python eval_ppo_ablation.py --iters 30 --episodes-per-iter 15 --eval-episodes 50

# Step 4: View results
cat ablation_results.json
```

### Option 2: Step-by-Step Execution

```bash
# Step 1: Collect teacher dataset (4-5 minutes)
python collect_dataset.py \
  --anchor-mode twelve \
  --episodes 30 \
  --out-prefix teacher_data_12

# Step 2: Distill student policy (4-5 minutes)
python distill_student.py \
  --teacher-data-X teacher_data_12_X.npy \
  --teacher-data-P teacher_data_12_P.npy \
  --out student_policy_init.pt

# Step 3: Train PPO-from-scratch (10-12 minutes)
python ppo_finetune.py \
  --iters 30 \
  --force-scratch \
  --out ppo_scratch.pt

# Step 4: Train PPO-teacher-init (10-12 minutes)
python ppo_finetune.py \
  --iters 30 \
  --student-checkpoint student_policy_init.pt \
  --out ppo_teacher_init.pt

# Step 5: Comprehensive evaluation
python eval_ppo_ablation.py \
  --iters 30 \
  --episodes-per-iter 15 \
  --eval-episodes 50
```

## API Reference

### Environment Creation

```python
from env_fixture import DiscreteFixtureEnv

env = DiscreteFixtureEnv(
    anchor_mode="twelve",    # 12-anchor configuration
    num_episodes=15,
    num_steps=20
)

obs, info = env.reset()      # obs.shape = (15,)
action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

### Policy Training

```python
from ppo_finetune import ppo_finetune

# Train from scratch
ppo_finetune(
    num_iters=30,
    episodes_per_iter=15,
    force_scratch=True,
    out_path="ppo_scratch.pt"
)

# Train with warm-start
ppo_finetune(
    num_iters=30,
    episodes_per_iter=15,
    student_checkpoint="student_policy_init.pt",
    out_path="ppo_teacher_init.pt"
)
```

### Policy Evaluation

```python
from eval_policy import evaluate_policy

rmse_list, returns_list = evaluate_policy(
    policy_path="ppo_teacher_init.pt",
    num_episodes=50
)

import numpy as np
print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
print(f"Return: {np.mean(returns_list):.4f} ± {np.std(returns_list):.4f}")
```

## Results Summary

### Ablation Study Outcomes

| Metric | PPO-Scratch | PPO-Teacher-Init | Delta |
|--------|------------|------------------|-------|
| **RMSE (mean)** | 0.1934 | 0.1995 | +0.0061 |
| **RMSE (std)** | 0.0391 | 0.0232 | -0.0159 |
| **Return (mean)** | +0.0879 | +0.1057 | +0.0178 |
| **Return (std)** | 0.1676 | 0.1236 | -0.0440 |
| **Convergence Iter** | 17 | 13 | -4 |

**Key Finding:** Teacher-initialized PPO shows marginal improvement in return (+1.8%) and slightly faster convergence, but the benefit is not statistically significant. Both variants converge to equivalent RMSE performance.

### Validation Results

✓ **8/8 validation tests passed:**
- ✓ 12 anchors correctly configured
- ✓ 15-dimensional observations
- ✓ 13-action space (12 + NOOP)
- ✓ Teacher context generation (12 candidates, 4 interior + 8 boundary)
- ✓ Teacher LLM output (12 logits)
- ✓ Probability normalization (sum to 1.0)
- ✓ Dataset shape verification (600×15, 600×12)
- ✓ Checkpoint compatibility (12-action output layer)

## Configuration Details

### 12-Anchor Placement Strategy

**Boundary Anchors (8 locations):**
- Corners: $(0,0), (8,0), (0,8), (8,8)$
- Edge midpoints: $(4,0), (4,8), (0,4), (8,4)$

**Interior Anchors (4 locations):**
- Lattice points: $(2,2), (6,2), (2,6), (6,6)$

### PPO Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| Learning rate | $1 \times 10^{-3}$ | Adam optimizer |
| Clip ratio | 0.2 | Clipped surrogate objective |
| Entropy coefficient | 0.01 | Exploration bonus |
| GAE λ | 0.99 | Advantage smoothing |
| Value loss coefficient | 0.5 | Weighted combination |
| Max gradient norm | 1.0 | Gradient clipping |
| Discount factor (γ) | 0.99 | Future reward weighting |

### Student Distillation

- **Loss function**: $\mathcal{L} = \text{KL}(\pi_{\text{teacher}} \| \pi_{\text{student}}) + \lambda \|\theta\|^2$
- **L2 regularization**: $\lambda = 10^{-3}$
- **Training steps**: 300
- **Convergence**: KL from 0.0059 to 0.0039

## Extending the Research

### Potential Next Steps

1. **Extended Ablation Study** (Statistical Significance)
   ```bash
   python eval_ppo_ablation.py --iters 50 --eval-episodes 100
   ```
   Increases statistical power for significance testing.

2. **Anchor Utilization Analysis**
   - Track which anchors PPO actually selects
   - Compare utilization patterns between scratch vs. teacher-init
   - Analyze interior vs. boundary point usage

3. **Hyperparameter Optimization**
   ```bash
   # Learning rate sweep
   python ppo_finetune.py --iters 50 --lr 5e-4 --out ppo_lr5e4.pt
   
   # Entropy coefficient sweep
   python ppo_finetune.py --iters 50 --ent 0.05 --out ppo_ent05.pt
   ```

4. **Real LLM Integration**
   - Replace mock teacher with actual LLM queries
   - Measure query costs and policy improvement trade-off
   - Optimize prompt engineering for better guidance

5. **Interior vs. Boundary Ablation**
   ```bash
   # Compare 9-anchor (boundary only) vs. 12-anchor configuration
   python env_fixture.py --anchor-mode stride   # 9 anchors
   python env_fixture.py --anchor-mode twelve   # 12 anchors
   ```

## Documentation

- **[ISYE_URP_Training_Report.tex](ISYE_URP_Training_Report.tex)** - Comprehensive technical report (Overleaf-ready)
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference for common tasks
- **[IMPLEMENTATION_TASKS_7.1A_7.2B.md](IMPLEMENTATION_TASKS_7.1A_7.2B.md)** - Detailed implementation specification
- **[TRAINING_RUN_REPORT.md](TRAINING_RUN_REPORT.md)** - Experimental findings and analysis

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### Issue: `Shape mismatch for net.0.weight`
**Solution:** Ensure environment and checkpoint have consistent dimensions
```python
# Check environment observation dimension
env = DiscreteFixtureEnv(anchor_mode="twelve")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")  # Should be (15,)
```

### Issue: Validation tests failing
**Solution:** Run full conversion pipeline first
```bash
python convert_to_12_anchors.py
python validate_12_anchor_setup.py
```

## Performance Metrics

**Computational Performance:**
- Dataset collection: 3–4 minutes (30 episodes)
- Student distillation: 4–5 minutes (300 steps)
- PPO training per variant: 10–12 minutes (30 iterations)
- Full workflow: ~30–40 minutes

**Memory Requirements:**
- Student checkpoint: 26.4 KB
- Dataset (NumpyArrays): 7 MB
- Full trained policies: ~79 KB (3 checkpoints)


---

**Last Updated:** January 22, 2026  
**Status:** ✓ Complete and Validated  
**Python:** 3.11+  
**PyTorch:** 2.0+
