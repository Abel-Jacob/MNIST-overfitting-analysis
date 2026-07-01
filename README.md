# MNIST Overfitting and Regularization Analysis

A controlled study of overfitting and regularization on the MNIST handwritten digit 
dataset using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Overview

The same CNN architecture was trained twice under identical conditions — once without 
any regularization to induce overfitting, and once with regularization techniques 
applied only to the training pipeline. The architecture and data split were held 
constant between both runs.

## Dataset

Full MNIST dataset (70,000 images) merged and re-split using stratified sampling:

- Train: 49,000 images (70%)
- Validation: 10,500 images (15%)
- Test: 10,500 images (15%)

## Architecture

ExperimentalCNN: 3 convolutional blocks (64, 128, 256 filters) with 2 conv layers 
each and MaxPooling, followed by a 4-layer dense classifier head (1024, 512, 256, 10). 
Total parameters: 4,163,274.

## Experiments

### Part A - Unregularized
- No dropout, no weight decay, no augmentation
- Fixed learning rate (Adam, lr=0.001)
- 30 epochs

### Part B - Regularized (same architecture)
- Data augmentation: random rotation (15 deg), translation (10%), scaling (90-110%)
- L2 weight decay: 1e-4 via Adam optimizer
- Learning rate scheduling: ReduceLROnPlateau, factor=0.5, patience=2
- Early stopping: patience=4 epochs on validation loss

## Results

| Model | Train Acc | Val Acc | Test Acc | Gap | Test Loss |
|---|---|---|---|---|---|
| Unregularized | 99.86% | 99.32% | 99.30% | 0.53% | 0.0708 |
| Regularized | 99.58% | 99.54% | 99.45% | 0.04% | 0.0222 |

Val/Train loss ratio: 8.91x (unregularized) vs 1.17x (regularized).

## Key Finding

Overfitting is a property of the training process, not model size. The same 
4.16 million parameter network that produced a loss ratio of 8.91x without 
constraints achieved 1.17x with training-time regularization alone — no 
architectural changes required.

## Requirements
