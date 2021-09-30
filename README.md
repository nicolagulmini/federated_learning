# Federated Learning

This repository contains the code for simulating and studying the tradeoff between local and global accuracy in a distributed scenario
in which each user trains a classification model on its own data distribution.

Research project: "*Communication-Aware Clustered Federated Learning: How to Leverage Data Heterogeneity*"

The first experiments are carried out with a federated version of mnist, to have a fast computation. 
Soon will be available also a federated version of cifar10 and the related notebook.

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

In the following results, the loaded dataset was augmented to make the learning more challenging.

## Results

Let the *global accuracy* be the accuracy of a model on the homogeneous (balanced) server-side dataset; and the *local accuracy* the accuracy of a model on a local dataset, i.e. a cluster heterogeneous (unbalanced) dataset. 
When we want to test a model on the local datasets, the local accuracy is measured on each local dataset and then the average is considered.
When we want to test the local / clusters models on the server dataset, each one has its own accuracy and even in this case the average is computed.

In the following plots these metrics are taken into account:
- **avg local acc - clusters models**: each local model measures the accuracy on its local dataset, and then the average is considered. No cross measures (for instance the cluster 1 model on the cluster 2 dataset) is considered;
- **avg global acc - cluster models**: each local model measures the accuracy on the same balanced server dataset, and then the average is considered;
- **global acc - avg softmax outputs**: given a server dataset image, each cluster model predicts its label. Then the average of the softmax outputs (the last layer of each classification model) is considered and the argmax of that unique vector is used to predict the label;
- **global acc - genie**: this is the expected upper bound that we want to reach on the server dataset. For each image we look at its label, and if there is a cluster with a local dataset unbalanced on that label (i.e. in that dataset there are a lot of images with that label), its model is used to predict it (note that it could be wrong). Otherwise, if there are not any dataset unbalanced on that label, the softmax average is considered, like the "global acc - avg softmax outputs" case;
- **avg local acc - avg softmax outputs**: the average softmax outputs method on each local dataset, and then the average of each local accuracy is considered.
- **global acc - avg local models weights**: the accuracy on the server dataset, computed my a model whose weights are the average of the clusters models weights, for each communication round. There is no weights update: the server does not share the model with the clusters, it only receives the updated weights.
- **global acc - server fedavg with cluster models**: the same as before, but the server shares the weights with the clusters, so the scenario is different (in fact the simulation is performed in a separated way, and then the curves are included in the same plot).
- **avg local acc - avg local models weights** and **avg local acc - server fedavg with cluster models** are the same as before, but computing the accuracies on the local datasets and then averaging them.

In order to make a comparation on the same dataset the following plots are divided in global accuracy and local accuracy, for different heterogeneity degrees, and with rotated images in both the local and the server side datasets.
Note also that the same model, with the same hyperparameters, is used for each cluster and for each simulation.

###### 15% heterogeneity
<img src = "https://user-images.githubusercontent.com/62892813/128598903-27218f80-39fd-4817-915e-9033c127b626.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128598906-9dd274db-c848-4125-be8a-a3fe039ef162.png" width = "315" height = "210">

###### 50% heterogeneity
<img src = "https://user-images.githubusercontent.com/62892813/128598913-a93eab10-572c-42e5-9744-6b106a9d9108.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128598915-3b8ddedb-3372-4a18-9b85-b6a21ce2f006.png" width = "315" height = "210">

###### 80% heterogeneity
<img src = "https://user-images.githubusercontent.com/62892813/128598924-b07da911-f454-45a0-bd62-5fa73b464303.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128598925-8f2e75ab-90b3-4c8a-8874-356d6903d981.png" width = "315" height = "210">

## Aim of the project

A new model could provide an higher global accuracy on the server testset, better exploiting the information from each single cluster. The objective is to stay in the **gap** between the genie curve and the avg local models weights one, in order to obtain better results with respect to that, and reach as much as possible the genie performance. Test with various heterogeneity degrees and augmentation of the datasets are performed to simulate an high unbalanced scenario in which that gap is noticeably wide, especially in the first communication rounds.
Also the local accuracy plots are interesting because all the approaches, except the avg local acc one, have about the same performance, so the objective is to find a model that performs better.

The following gifs show the curves changing according to the heterogeneity degree. Even if the global gap in which our model has to stay does not change a lot, there is another gap in the local plots that gets wider, so the 80% heterogeneity is the best scenario in which we can run our experiments.

<img src = "https://user-images.githubusercontent.com/62892813/128599509-77e9975a-dbb3-420d-b274-a7b71445d66f.gif" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/128599507-b5192e98-6881-44f2-a01f-8224e5f4ce61.gif" width = "315" height = "210">

## References

- [The Communication-Aware Clustered Federated Learning Problem](https://ieeexplore.ieee.org/document/9174245)
- [Federated Mixture of Experts](https://arxiv.org/abs/2107.06724)
- [FedMD: Heterogeneous Federated Learning via Model Distillation](https://towardsdatascience.com/fedmd-heterogeneous-federated-learning-via-model-distillation-e84676183eb4)
- [Specialized federated learning using a mixture of experts](https://arxiv.org/pdf/2010.02056.pdf)
- others...

# TODO
- [x] make more versions of federated_mnist, organize them in a directory
- [ ] other versions of federated_mnist to study the heterogeneity
- [x] include the fedavg performance 
- [x] update the figures on readme
- [ ] make federated_cifar10
- [ ] ~~tutorial notebook for the simulations~~ add the tutorial notebook
- [x] make a directory with a set of already trained model, for different scenarios
- [x] ~~add metrics in train_one_shot method~~ added metrics in notebook
- [ ] add estimation model section
- [x] investigate about avg models weights and fedavg performance
