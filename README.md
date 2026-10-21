# CS 6362 Assignment 2 — Paper 2 Reproduction

PyTorch reproduction of Figure 3 (F1, C1, and C2) from *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* (Keskar et al., ICLR 2017).

## Overview

This project investigates how small-batch and large-batch training affect neural network generalization and the geometry of the loss landscape.

The experiments reproduce:

- **F1:** Fully connected network trained on MNIST.
- **C1:** Shallow convolutional network trained on CIFAR-10.
- **C2:** Deep convolutional network trained on CIFAR-10.

Each experiment evaluates cross-entropy loss and classification accuracy along a parameter interpolation path between small-batch and large-batch solutions.

## Implementation

The experiments use PyTorch, MNIST, and CIFAR-10, with:

- Full training and test datasets
- F1, C1, and C2 network architectures
- Small-batch size of 256
- Large-batch size of 10% of the training dataset
- Adam optimizer with learning rate 0.001
- A maximum of 12 training epochs in QUICK mode
- One paired trial with shared initialization
- 31 interpolation points over α ∈ [-1, 2]
- BatchNorm statistics recalibrated at intermediate interpolation points

## How to Reproduce

1. Open the reproduction `.ipynb` notebook in Google Colab.
2. Select a GPU runtime if available.
3. Run the notebook cells in order.
4. The notebook downloads MNIST and CIFAR-10, trains the models, evaluates parameter interpolation, and generates the reproduced figures.

Required Python packages include `torch`, `torchvision`, `numpy`, and `matplotlib`.

Training may take substantial time depending on available hardware. Saved checkpoints can be reused when available, but are not required to reproduce the experiments from scratch.

## Limitations

This is a computationally reduced reproduction rather than an exact replication of the original experiments. QUICK mode uses one paired trial and a maximum of 12 epochs instead of the original paper's more extensive training.

The large-batch C1 and C2 models achieved substantially lower training accuracy than their small-batch counterparts, preventing a clean interpretation of their test accuracy differences as a generalization gap.

BatchNorm running statistics are recalibrated during interpolation, an implementation choice that may affect the resulting curves. All reproduced figures use measured model evaluations rather than hardcoded results.

## Reference

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., and Tang, P. T. P. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. ICLR 2017.

https://openreview.net/forum?id=H1oyRlYgg
