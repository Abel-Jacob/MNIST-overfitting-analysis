# MNIST Classification: Overfitting and Regularization Using MLP
A controlled study of overfitting and regularization on the MNIST handwritten digit
dataset using fully-connected Multi-Layer Perceptrons (MLPs) implemented in PyTorch.

## Overview
The same over-parameterized MLP architecture was trained twice under identical data
conditions — once without any regularization to induce overfitting, and once with
regularization techniques applied only to the training pipeline (architecture
untouched). A small, appropriately-sized baseline MLP is also included to establish
generalization performance when model capacity is matched to the task, confirming
that any overfitting observed is a consequence of excess capacity rather than the
dataset or split.

A fully-connected MLP was deliberately chosen over a CNN for this study: it has no
convolutional inductive bias (no weight sharing, spatial locality, or translation
equivariance), which makes it far more prone to memorizing training-set pixel
patterns and therefore a clearer architecture for demonstrating overfitting on both
loss and accuracy curves.

## Dataset
Full MNIST dataset (70,000 images) merged and re-split using stratified sampling
(`random_state=42`, reused identically across all three experiments):
- Train: 49,000 images (70%)
- Validation: 10,500 images (15%)
- Test: 10,500 images (15%)

All images are normalized using the standard MNIST channel mean (0.1307) and
standard deviation (0.3081).

## Architectures
- **BaselineMLP:** Flatten (784) → Linear(784→64) → BatchNorm1d → ReLU →
  Dropout(0.2) → Linear(64→10). BatchNorm is used for training stability only, not
  as a regularizer. Total parameters: **51,018**.
- **ExperimentalMLP:** Flatten (784) → Linear layers (784→2048→1024→512→256→10)
  with ReLU activations between each. No dropout, batch normalization, or other
  regularization in the architecture itself. Total parameters: **4,364,554**
  (~85× the baseline).

## Experiments
### Baseline
- Small MLP (BaselineMLP)
- No dropout beyond the fixed 20% layer, no weight decay, no augmentation
- Fixed learning rate (Adam, lr=0.001)
- Batch size 64
- 12 epochs

### Part A — Unregularized
- Large MLP (ExperimentalMLP), architecture unchanged from Part B below
- No weight decay, no augmentation
- Fixed learning rate (Adam, lr=0.001)
- Batch size 256
- 70 epochs

### Part B — Regularized
- Identical ExperimentalMLP, not one layer changed from Part A
- Data augmentation: random rotation (up to 15°), translation (±10%), scaling (90–110%)
- L2 weight decay: 1e-3 via Adam optimizer
- Learning rate scheduling: ReduceLROnPlateau, factor=0.5, patience=2
- Early stopping: patience=5 epochs on validation loss
- Batch size 128
- Up to 30 epochs

## Results
| Model | Train Acc | Val Acc | Test Acc | Gap (Train−Val) | Test Loss |
|---|---|---|---|---|---|
| Baseline | 96.78% | 96.98% | 96.98% | -0.21% | 0.1012 |
| Unregularized | 99.98% | 98.40% | 98.39% | 1.58% | 0.1905 |
| Regularized | 97.82% | 98.92% | 98.84% | -1.11% | 0.0361 |

Val/Train loss ratio: 1.07x (baseline) vs 224.58x (unregularized) vs 0.51x (regularized).

The unregularized model's train-val accuracy gap peaked even higher mid-training,
reaching **2.21% at epoch 17** (99.67% train vs. 97.46% val), before fluctuating
down to its final 1.58% gap — the peak, not just the final epoch, is the more honest
measure of how severely it overfit.

## Key Finding
Overfitting is a property of the training process, not model size alone. The same
4.36-million-parameter MLP that produced a validation/training loss ratio of 224.58x
without constraints achieved 0.51x with training-time regularization alone — no
architectural changes required. A well-matched model like the 51K-parameter baseline
generalizes naturally (-0.21% gap) without explicit constraints, while the
regularized model's negative gap (-1.11%) shows the augmented training task was
actually harder than the clean validation/test tasks it was evaluated on. Together
these results show that generalization is governed by the capacity-to-task match and
training protocol, not parameter count in isolation.

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
Open the notebook in Jupyter or Google Colab and run all cells top to bottom.
*(Update the filename below to match the actual `.ipynb` file in this repo.)*
