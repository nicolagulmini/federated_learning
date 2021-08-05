# Federated Learning

This repository contains the code for simulating and studying the tradeoff between local and global accuracy in a distributed scenario
in which each user trains a classification model on its own data distribution.

Research project: "Communication-Aware Clustered Federated Learning: How to Leverage Data Heterogeneity"

The (deprecated) files contain messy code. 

The first experiments are carried out with a federated version of mnist, to have a fast computation. If you want to generate your own version of federated mnist, you can use dataset_split.py, changing the initial parameters. Soon will be available also a federated version of cifar10 and the related notebook.

## federated_mnist_x
The federated_mnist folder contains:
- 9 heterogeneous datasets (training and test)
- in each folder there is a .csv file with name of images and related label
- the heterogeneity is realized in this way: x% of the images are of the same label, the other are taken randomly from the original dataset (so there may be duplicates)
- each dataset contains 1000 images, so each cluster contains exactly 2000 images. Some of these could be in common between more clusters.
This dataset has been created with dataset_split.py.

## results
Let the *global accuracy* be the accuracy of a model on the homogeneous (balanced) server-side dataset; and the *local accuracy* the accuracy of a model on a local dataset, i.e. a cluster heterogeneous (unbalanced) dataset. Since there are many clusters, the local accuracy is measured on each local dataset and then the average is considered.
When the local / clusters models are tested on the server dataset, each one has its own accuracy, so even in this case the average is computed.

## References
- The Communication-Aware Clustered Federated Learning Problem: https://ieeexplore.ieee.org/document/9174245
- Federated Mixture of Experts: https://arxiv.org/abs/2107.06724
- others
