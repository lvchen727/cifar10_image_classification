# Optimization of Image Classification for CIFAR-10 dataset

## Experiment
* Defined a CNN for classification of CIFAR-10 dataset
* Studied the influence of data augmentation
* Studied the influence of 2 different optimization algorithms - sgd and adam
* Studied the influence of 3 loss functions - cross-entropy, hinge loss, hinge squared loss

## Setup
I used google cloud platform, which I personally felt very user-friendly.
To setup google instance and GPU, I followed the instruction from standford cs231n. I used NVIDIA Tesla K80 GPU. See reports for details.
http://cs231n.github.io/gce-tutorial/
http://cs231n.github.io/gce-tutorial-gpus/

## Libraries
* Keras 1.20 (Note newer version will cause errors.)
* Tensorflow

## Code
I used jupyter notebook to write up all the codes instead of terminal because it's very convenient for debugging. It is very easy to follow through the codes since I explained the purpose of each block of code. You could read the codes from its corresponding html file. A python file also generated using the original jupyter notebook file.
* cse543-finalproject.ipynb
* cse543-finalproject.html
* cse543-finalproject.py
