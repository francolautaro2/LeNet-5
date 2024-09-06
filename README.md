# LeNet-5 Implementation

This repository contains an implementation of the LeNet-5 convolutional neural network, one of the pioneering architectures in the field of deep learning for image recognition tasks. The model is specifically designed for handwritten digit classification (e.g., the MNIST dataset).

## Model Overview

LeNet-5 is composed of the following layers:

1. **Input**: 32x32 pixel images (typically grayscale).
2. **Convolutional Layer (C1)**: 6 filters of size 5x5 with a stride of 1.
3. **Subsampling Layer (S2)**: 2x2 subsampling with an average pooling operation.
4. **Convolutional Layer (C3)**: 16 filters of size 5x5.
5. **Subsampling Layer (S4)**: 2x2 subsampling with average pooling.
6. **Fully Connected Layer (C5)**: Fully connected to the output.
7. **Output Layer (F6)**: Final fully connected layer with softmax activation.

## LeNet-5 Architecture

The architecture was proposed by Yann LeCun and his collaborators in 1998, and it played a crucial role in the early days of deep learning research. LeNet-5 demonstrated the effectiveness of convolutional neural networks (CNNs) for image recognition tasks.

## Reference

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). **Gradient-based learning applied to document recognition**. *Proceedings of the IEEE*, 86(11), 2278-2324. DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/francolautaro2/LeNet-5.git
