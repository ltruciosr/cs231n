# CS231n

Tasks for the [Convolutional Neural Network for Visual Recognition](https://cs231n.github.io/assignments2021) course.

1. [Assignment 1](#assignment-1)
1. [Assignment 2](#assignment-2)
1. [Assignment 3](#assignment-3)
1. [Assignment 4](#assignment-4)

## Assignment 2

**Q1. Multi-Layer FCNN**

Fully connected neural networks (FCNN) of arbitrary depth are implemeneted in the class [fc_net](assignment2/cs231n/classifiers/fc_net.py). There are other features also implemented on the NN.
- [Parameter Updates](https://cs231n.github.io/neural-networks-3/): Vanilla | MomentumSGD | Adam
- Activations: ReLU | [PReLu](https://arxiv.org/abs/1502.01852) | Sigmoid
- [Dropout](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
- [Weights Initialization](https://arxiv.org/abs/1502.01852)
- [Batch Normalization](https://arxiv.org/abs/1502.03167)

For the hyperaparameters tuning process, we prefers to use [Random Search](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) because these are more efficient for optimization than trials on a grid.



Finally, we test our implementation on the script [FullyConnectedNets.ipynb](assignment2/FullyConnectedNets.ipynb), where CIFAR-10 dataset is used to test our implementation and the capabilities of a FCNN on images.


