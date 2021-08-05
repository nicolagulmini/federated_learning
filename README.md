# Federated Learning

This repository contains the code for simulating and studying the tradeoff between local and global accuracy in a distributed scenario
in which each user trains a classification model on its own data distribution.

Research project: "*Communication-Aware Clustered Federated Learning: How to Leverage Data Heterogeneity*"

The (deprecated) files contain messy code. 

The first experiments are carried out with a federated version of mnist, to have a fast computation. If you want to generate your own version of federated mnist, you can use dataset_split.py, changing the initial parameters. Soon will be available also a federated version of cifar10 and the related notebook.

## federated_mnist_x
The federated_mnist folder contains:
- 9 heterogeneous datasets (training and test)
- in each folder there is a .csv file with name of images and related label
- the heterogeneity is realized in this way: x% of the images are of the same label, the other are taken randomly from the original dataset (so there may be duplicates)
- each dataset contains 1000 images, so each cluster contains exactly 2000 images. Some of these could be in common between more clusters.
This dataset has been created with dataset_split.py.

## Results
Let the *global accuracy* be the accuracy of a model on the homogeneous (balanced) server-side dataset; and the *local accuracy* the accuracy of a model on a local dataset, i.e. a cluster heterogeneous (unbalanced) dataset. 
When we want to test a model on the local datasets, the local accuracy is measured on each local dataset and then the average is considered.
When we want to test the local / clusters models on the server dataset, each one has its own accuracy and even in this case the average is computed.

In the following plots these metrics are taken into account:
- **avg local acc - clusters models**: each local model measures the accuracy on its local dataset, and then the average is considered. No cross measures (for instance the cluster 1 model on the cluster 2 dataset) is considered;
- **avg global acc - cluster models**: each local model measures the accuracy on the same balanced server dataset, and then the average is considered;
- **global acc - avg softmax outputs**: given a server dataset image, each cluster model predicts its label. Then the average of the softmax outputs (the last layer of each classification model) is considered and the argmax of that unique vector is used to predict the label;
- **global acc - most secure model for each img**: given an image, each cluster predicts its label. Then only the most confident model is listened, so only its prediction is used.
- **global acc - genie**: this is the expected upper bound that we want to reach on the server dataset. For each image we look at its label, and if there is a cluster with a local dataset unbalanced on that label (i.e. in that dataset there are a lot of images with that label), its model is used to predict it (note that it could be wrong). Otherwise, if there are not any dataset unbalanced on that label, the softmax average is considered, like the "global acc - avg softmax outputs" case;
- **avg local acc - avg softmax outputs**: the average softmax outputs method on each local dataset, and then the average of each local accuracy is considered.

###### 15% heterogeneity
<img src = "https://user-images.githubusercontent.com/62892813/128368565-8bb1ce4c-848e-41d2-8a65-435195fdf052.png" width = "400" height = " 300">

###### 50% heterogeneity
<img src = "https://user-images.githubusercontent.com/62892813/128369330-25dcacb2-5b96-48a3-87a4-74106ea25f17.png" width = "400" height = " 300"><img src = "https://user-images.githubusercontent.com/62892813/128371801-e2c9c66f-c3de-458b-bac3-a9584a256bd4.png" width = "400" height = " 300"><img src = "https://user-images.githubusercontent.com/62892813/128371811-74fa0b33-d78f-4239-981b-78198edbc5a5.png" width = "400" height = " 300">

###### 80% heterogeneity
<img src = "https://user-images.githubusercontent.com/62892813/128368650-dced8066-99f5-4450-b349-acfbcefebedd.png" width = "400" height = " 300">

###### Discussion
Note that in the first case, with a low degree of heterogeneity, the curves are stable in 5/6 communication rounds. The genie curve, as expected, is the highest, and all the others are quite at the same level. But when we start to increase the heterogeneity level, the curves start to separate and switch.
... to complete

## References
- The Communication-Aware Clustered Federated Learning Problem: https://ieeexplore.ieee.org/document/9174245
- Federated Mixture of Experts: https://arxiv.org/abs/2107.06724
- others
