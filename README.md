# center_loss_pytorch

## Introduction

This is an Pytorch implementation of center loss. Some codes are from the repository [MNIST_center_loss_pytorch](https://github.com/jxgu1016/MNIST_center_loss_pytorch).

## Usage

You should use centerloss like this in your training file.

```python
# Creat an instance of CenterLoss
centerloss = CenterLoss(10, 48, 0.1)
# Get the loss and centers params
loss_center, params_grad = centerloss(targets, features)
# Calculate all gradients
loss.backward()
# Manually assign centers gradients other than using autograd
centerloss.centers.backward(params_grad)
```
