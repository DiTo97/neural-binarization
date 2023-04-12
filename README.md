# Binarization Segformer

A semantic segmentation model for pixel-wise document image binarization.

## TODOs

- [ ] fine-tune Segformer starting from the checkpoint [`nvidia/segformer-b3-finetuned-cityscapes-1024-1024`](https://huggingface.co/nvidia/segformer-b3-finetuned-cityscapes-1024-1024);
- [ ] set `reduce_labels=True` in processor to ignore the background;
- [ ] implement DIBCO metrics from [here](https://github.com/masyagin1998/robin/blob/master/src/metrics/metrics.py) or [here](https://gist.github.com/pebbie/643e28c619efaa2fd30b1595bd5d0e6c).

## Overview

Segformer is an efficient semantic segmentation model introduced by [Xie et al.](https://arxiv.org/abs/2105.15203) in 2021.

In this repository, we will provide a fine-tuning of Segformer for pixel-wise document image binarization.

## Dataset

The dataset is an ensemble of 14 datasets replicating the setting used in SauvolaNet by [Li et al.](https://arxiv.org/abs/2105.05521) in 2021.

<img src="images/example.png" width="75%" />

> Figure 1. An example pair from the Bickley diary dataset

For more information on the dataset, see SauvolaNet's official [repository](https://github.com/Leedeng/SauvolaNet).
