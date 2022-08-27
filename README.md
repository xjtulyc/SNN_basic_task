# SNN basic task

## Abstract

SNN is also known as Spiking Neural Network, which simulates the firing process of neurons for training and reasoning. In recent years, deep learning has encountered a certain bottleneck, such as the inference process needs a lot of computing resources, and the model performance has met the ceiling. People have pinned their hopes on the new type of model, pulsed-neural network. Therefore, pulsed neural networks are also called the third generation neural networks.

This repository includes some pulsed neural network implementations for computer vision tasks, based on the PyTorch deep learning framework. Computer vision tasks include image classification, object detection, etc. For general vision problems, pulse neural networks will first encode images, including direct coding, Poisson coding and other coding methods, mapping each pixel value into a sequence value. After that, the encoded sequence is input into the pulse neural network for training through BPTT and other methods. The output of the model is also a sequence that propagates the error through a specific form of loss calculation.

## Image Classification for MNIST

We first introduce different coding schemes, including direct coding, which is straightforward, and Poisson coding, which has more mathematical guarantees. In terms of implementation, we compared a PyTorch compatible SNN library on GitHub and an article from Nature Communications2022 that we reproduced. The experimental results show that the SNN PyTorch implementation is similar to RNN.

You can get the source code for MNIST image classification in the direction of **ImageClassification**. The detailed report can get in https://zhuanlan.zhihu.com/p/558272145 (in Chinese).

If you want to train the SNN model for image classification, run the following command.

```
cd ImageClassification
python train.py
```
