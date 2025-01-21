# mytorch
This is a small, self-instructional project to learn the detailed structure of common neural network archtectures by building and training them. It uses torch tensors for efficiency of manipulation and built in automatic differentiation. 

Included are 3 small demonstrations of functionality: 

1. A basic MLP used to classify MNIST images
2. A basic Convolutional neural net used to classify images from the CIFAR10 dataset
3. A small decoder-only transformer language model, used to generate (obviously, quite bad) poetry

Of particular note are the following parts: 

## layers.py
This contains most of the machinery, and implements the following layers:
-- Linear

-- Conv2d

-- Dropout

-- MaxPool

-- Avg Pool

-- LayerNorm

-- Multiheaded Attention

-- Tranformer Block for a Decoder only network

## functions.py

Includes implementations of the basic functions: 

-- ReLU

-- Sigmoid

-- One Hot Vector

-- Softmax

-- Log Softmax

-- Cross Entropy Loss


## models.py 

Contains the files to build the following networks: 
-- MLP

-- Convolutional Neural Net

-- A language model based on a decoder-only transformer


There is also a data_utils file and the required datasets used in the Jupyter notebooks. 

