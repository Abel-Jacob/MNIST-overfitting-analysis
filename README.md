# MNIST Overfitting and Regularization Analysis

A controlled study of overfitting and regularization on the MNIST handwritten digit 
dataset using Convolutional Neural Networks (CNNs) implemented in PyTorch.

## Overview

The same large CNN architecture was trained twice under identical conditions — once without 
any regularization to induce overfitting, and once with regularization techniques 
applied only to the training pipeline. A small baseline model is also included to establish 
generalization performance when model capacity is matched to the task.

## Dataset

Full MNIST dataset (70,000 images) merged and re-split using stratified sampling:

- Train: 49,000 images (70%)
- Validation: 10,500 images (15%)
- Test: 10,500 images (15%)

## Architectures

- **BaselineCNN:** 2 convolutional blocks (32 and 64 filters) followed by batch normalization 
  and MaxPooling, feeding into a 2-layer compact classifier head (128, 10). 
  Total parameters: 422,026.
- **ExperimentalCNN:** 3 convolutional blocks (64, 128, 256 filters) with 2 conv layers 
  each and MaxPooling, followed by a 4-layer dense classifier head (1024, 512, 256, 10). 
  Total parameters: 4,163,274.

## Experiments

### Baseline
- Small CNN (BaselineCNN)
- No dropout, no weight decay, no augmentation
- Fixed learning rate (Adam, lr=0.001)
- 10 epochs

### Part A - Unregularized
- Large CNN (ExperimentalCNN)
- No dropout, no weight decay, no augmentation
- Fixed learning rate (Adam, lr=0.001)
- 30 epochs

### Part B - Regularized
- Large CNN (ExperimentalCNN)
- Data augmentation: random rotation (15 deg), translation (10%), scaling (90-110%)
- L2 weight decay: 1e-4 via Adam optimizer
- Learning rate scheduling: ReduceLROnPlateau, factor=0.5, patience=2
- Early stopping: patience=4 epochs on validation loss

## Results

| Model | Train Acc | Val Acc | Test Acc | Gap (Train-Val) | Test Loss |
|---|---|---|---|---|---|
| Baseline | 99.70% | 98.85% | 98.85% | 0.85% | 0.0426 |
| Unregularized | 99.83% | 99.01% | 98.97% | 0.82% | 0.0829 |
| Regularized | 99.67% | 99.56% | 99.51% | 0.11% | 0.0174 |

Val/Train loss ratio: 5.27x (baseline) vs 9.80x (unregularized) vs 1.48x (regularized).

## Key Finding

Overfitting is a property of the training process, not model size. The same 
4.16 million parameter network that produced a loss ratio of 9.80x without 
constraints achieved 1.48x with training-time regularization alone — no 
architectural changes required. A well-matched model like the 421K parameter baseline 
generalizes naturally (0.85% gap) without explicit constraints, showing that 
generalization limits are a function of the capacity-to-task match.

## Requirements
```bash
pip install -r requirements.txt
```

## Usage

Open `MNIST_Overfitting_Analysis_Updated_(3) (1).ipynb` in Jupyter or Google Colab and run all cells. 
A GPU is recommended (the notebook was developed on a Tesla T4 via Google Colab).

## Environment

- Python 3.10
- PyTorch 2.11.0+cu128
- Google Colab (Tesla T4 GPU)
