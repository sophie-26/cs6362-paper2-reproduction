# Keskar et al. (2017) — Figure 3 Reproduction

PyTorch reproduction of Figure 3 panels F1, C1, and C2 from *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*.

## Running the notebook

1. Open `keskar_figure3_reproduction.ipynb` in Google Colab.
2. Select a GPU runtime (NVIDIA T4 recommended).
3. Run all cells in order.
4. Results, model checkpoints, and figures are saved to `/content/keskar_figure3/`.

## Experimental configuration

- Architectures: F1 (MNIST), C1 and C2 (CIFAR-10)
- Small-batch size: 256
- Large-batch size: 10% of the training dataset
- Optimizer: Adam
- QUICK mode: 1 trial, up to 12 epochs, 31 interpolation points
- Random seed: 2027

## Limitations

The experiments use a reduced computational budget. In particular, the C1 and C2 large-batch models did not achieve training accuracy comparable to the small-batch models. Therefore, the observed test accuracy differences cannot be attributed solely to generalization.

BatchNorm running statistics are recalibrated at non-endpoint interpolation values.

All reported measurements come from actual PyTorch evaluations.
