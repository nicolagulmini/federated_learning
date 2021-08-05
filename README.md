# Federated Learning

This repository contains the code for simulating and studying the tradeoff between local and global accuracy in a distributed scenario
in which each user trains a classification model on its own data distribution.

Research project: "*Communication-Aware Clustered Federated Learning: How to Leverage Data Heterogeneity*"

The (deprecated) files contain messy code. 

The first experiments are carried out with a federated version of mnist, to have a fast computation. Soon will be available also a federated version of cifar10 and the related notebook.

## federated_mnist_x

The federated_mnist folder contains:
- 9 heterogeneous datasets (training and test)
- in each folder there is a .csv file with name of images and related label
- the heterogeneity is realized in this way: x% of the images are of the same label, the other are taken randomly from the original dataset (so there may be duplicates)
- each dataset contains 1000 images, so each cluster contains exactly 2000 images. Some of these could be in common between more clusters.

If you want to generate your own version of federated mnist, you can use `dataset_split.py`, changing the initial parameters:
```
number_of_clusters = 9
bias = 0.8
number_of_images_per_dataset = 1000
```
where in this case there are 9 local datasets, with 1000 train and 1000 test images each, and an 80% (= bias * 100) heterogeneity.

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
[comment]: <> (<img src = "https://user-images.githubusercontent.com/62892813/128375181-d3530e8f-f0ab-4d34-bd4d-f7343aa38cc4.png" width = "315" height = "210">)
<img src = "https://user-images.githubusercontent.com/62892813/128375177-30b2804a-b15d-42d3-81cb-3464bccba3bf.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128375174-5ece393d-277b-4eb5-bc3c-d5537689f2c4.png" width = "315" height = "210">

###### 50% heterogeneity
[comment]: <> (<img src = "https://user-images.githubusercontent.com/62892813/128373075-121239c9-05e1-4e68-b154-3d4105941ec5.png" width = "315" height = "210">)
<img src = "https://user-images.githubusercontent.com/62892813/128373086-f3fe794b-03c1-4c2e-b6f4-3d2cc5b6ba32.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128373079-8df2e8a5-85d4-43d1-b5b6-7ace33cba406.png" width = "315" height = "210">

###### 80% heterogeneity
[comment]: <> (<img src = "https://user-images.githubusercontent.com/62892813/128377196-786db75d-1b44-4621-a620-fdb58a1ed3dc.png" width = "315" height = "210">)
<img src = "https://user-images.githubusercontent.com/62892813/128377199-a2914223-bf9a-4c10-befd-a08b5eb382c2.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128377190-2921d90f-59d5-4a50-9d23-28c3199c8aa0.png" width = "315" height = "210">

## Discussion

In the first case, with a low degree of heterogeneity, the curves are stable in a few communication rounds. The genie curve, as expected, is the highest, and all the others are quite at the same level. When we start to increase the heterogeneity level, the curves start to separate, because they need more communication rounds to stabilize. With an high degree of heterogeneity, the curve of the average local accuracies of the clusters models is the only one that passes the genie one.

In order to make a comparation on the same dataset, the plots are split in the global accuracy curves and local accuracy curves. Note also that the same model, with the same hyperparameters, is used for each cluster and for each simulation.

A new model could provide an higher global accuracy on the server testset, better exploiting the information from each single cluster. The objective is to stay in the **genie - avg softmax outputs** gap, in order to obtain better results with respect to the avg softmax outputs method, and reach as much as possible the genie performance. Test with various heterogeneity degrees are performed to simulate an high unbalanced scenario in which that gap is noticeably wide, especially in the first communication rounds.

<img src = "https://user-images.githubusercontent.com/62892813/128377847-f757e079-2832-4c68-a8be-913c62c8552a.gif" width = "450" height = "300">

## References

- [The Communication-Aware Clustered Federated Learning Problem](https://ieeexplore.ieee.org/document/9174245)
- [Federated Mixture of Experts](https://arxiv.org/abs/2107.06724)
- [FedMD: Heterogeneous Federated Learning via Model Distillation](https://towardsdatascience.com/fedmd-heterogeneous-federated-learning-via-model-distillation-e84676183eb4)
- others...

# TODO
- [x] make more versions of federated_mnist, organize them in a directory
- [ ] other versions of federated_mnist to study the heterogeneity
- [ ] make federated_cifar10
- [ ] tutorial notebook for the simulations
- [ ] make a directory with a set of already trained model, for different scenarios
- [ ] add metrics in train_one_shot method
- [ ] add estimation model section
