# Federated Learning

This repository contains the code for simulating and studying the tradeoff between local and global accuracy in a distributed scenario
in which each user trains a classification model on its own data distribution.

Research project: "Communication-Aware Clustered Federated Learning: How to Leverage Data Heterogeneity"

The (deprecated) files contain messy code. 

The first experiments are carried out with a federated version of mnist, to have a fast computation. If you want to generate your own version of federated mnist, you can use dataset_split.py, changing the initial parameters. Soon will be available also a federated version of cifar10 and the related notebook.

## federated_mnist 
The federated_mnist folder contains:
- 9 heterogeneous datasets (training and test)
- in each folder there is a .csv file with name of images and related label
- the heterogeneity is realized in this way: 50% of the images are of the same label, the other 50% are taken randomly from the original dataset (so there may be duplicates)
- each dataset contains 400 images, so each cluster contains exactly 800 images. Some of these could be in common between more clusters.
This dataset has been created with dataset_split.py.

## References
- The Communication-Aware Clustered Federated Learning Problem: https://ieeexplore.ieee.org/document/9174245
- Federated Mixture of Experts: https://arxiv.org/abs/2107.06724
- others

<img src="https://latex.codecogs.com/gif.latex?x^*&space;\approx&space;\arg\max_{\mathcal{C}\in\bar\Omega(\mathcal{C}_{i\in[1,k]})}\mathcal{L}^{\Omega*}_{jhr}(x)" title="x^* \approx \arg\max_{\mathcal{C}\in\bar\Omega(\mathcal{C}_{i\in[1,k]})}\mathcal{L}^{\Omega*}_{jhr}(x)" />
