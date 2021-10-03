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

## The aggregator

The **aggregator** model provides an higher global accuracy on the server testset, exploiting the information from each single cluster, better than the simple average. The objective was to stay in the **gap** between the genie curve and the avg softmax local outputs one (i.e. between the red and the pink curves), in order to reach as much as possible the genie performance. A 200 rounds simulation in 80% heterogeneity degree was performed, and the related models saved, to exploit models in an already saturated trend. We focus only on the global accuracy: in our scenario, the users use their local models, because they are trained and optimized on their own data, but the server wants to know how to combine them.

Since the objective is to find an optimal combination of the softmax outputs, i.e. 9 (because we have 9 clusters) coefficients, the following architecture is used:
an image is given to each cluster model to produce the softmax output. Then both the image and the outputs vector are given to the aggregator. The image is given to a flatten layer and then to a dense layer which return a 9-dimensional vector, the vector of the coefficients. A dot product between this vector and the softmax outputs is performed, then a dense layer with a softmax activation function returns the 10-dimensional vector with the probability of each class: this architecture is trained to classify, like the local ones. 

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_image (InputLayer)        [(None, 28, 28)]     0                                            
__________________________________________________________________________________________________
flatten_64 (Flatten)            (None, 784)          0           input_image[0][0]                
__________________________________________________________________________________________________
dense_73 (Dense)                (None, 9)            7065        flatten_64[0][0]                 
__________________________________________________________________________________________________
softmax_outputs (InputLayer)    [(None, 9, 10)]      0                                            
__________________________________________________________________________________________________
dot (Dot)                       (None, 10)           0           dense_73[0][0]                   
                                                                 softmax_outputs[0][0]            
__________________________________________________________________________________________________
dense_74 (Dense)                (None, 10)           110         dot[0][0]                        
==================================================================================================
Total params: 7,175
Trainable params: 7,175
Non-trainable params: 0
__________________________________________________________________________________________________
```

![global (1)](https://user-images.githubusercontent.com/62892813/135502734-6c92ad3e-571d-434c-9409-cf4919eace3d.png)

For now, we are interested in learning a combination between the final models' (after the 200 communication rounds) that performs better than the averaging softmax outputs, so we do not train our aggregator in each round.


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
