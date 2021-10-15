# Federated Learning

This repository contains the code for simulating and studying the tradeoff between local and global accuracy in a distributed scenario
in which each user trains a classification model on its own data distribution.

Research project: "*Communication-Aware Clustered Federated Learning: How to Leverage Data Heterogeneity*"

The first experiments are carried out with a federated version of mnist, to have a fast computation. 
Soon will be available also a federated version of other datasets.

## federated datasets

### federated_mnist_x

The federated_mnist folder contains:
- 9 heterogeneous datasets (training and test)
- in each folder there is a .csv file with name of images and related label
- the heterogeneity is realized in this way: x% of the images are of the same label, the other are taken randomly from the original dataset (so there may be duplicates)
- each dataset contains 1000 images, so each cluster contains exactly 2000 images. Some of these could be in common between more clusters.

If you want to generate your own version of federated mnist, you can use `dataset_split.py`, changing the initial parameters:
```python
number_of_clusters = 9
bias = 0.8
number_of_images_per_dataset = 1000
```
where in this case there are 9 local datasets, with 1000 train and 1000 test images each, and an 80% (= bias * 100) heterogeneity. 

In the following results, the loaded dataset was augmented to make the learning more challenging.

### federated_fashion_mnist_x

Despite all the following results are for federated_mnist, once obtained them, we decided to try with another dataset. The federated version of fashion-mnist is built in a similar manner as the previous dataset. The comfort is that the fashion-mnist images are 28x28 like the digits mnist, so the models do not need any modification. The available datasets are:
- federated_fashion_mnist_90. In this case the heterogeneous local datasets are 15, not 9. 
- federated_fashion_mnist_80, with the same settings as federated_mnist_80.
Since fashion images are intrinsically more difficult than the digits, we used models with the input, flatten, and final dense layer with softmax activation function. To train them we put epochs=4 for each user's model.

### federated_cifar10_x

The federated version of cifar10 is obtained as the previous datasets, but changing something in the `dataset_split.py` file, in the line 20 and 24:
```python
single_digit_training_sets[Y_train[i]]
...
single_digit_test_sets[Y_test[i]]
```
into
```python
single_digit_training_sets[Y_train[i][0]]
...
single_digit_test_sets[Y_test[i][0]]
```
and also at the line 38 and 41, changing `client_y_train[c].append(Y_train[random_index])` and `client_y_test[c].append(Y_test[random_index])` into `client_y_train[c].append(Y_train[random_index][0])` and `client_y_test[c].append(Y_test[random_index][0])`.

The available datasets are:
- federated_cifar10_80: same settings as the previous 80%-heterogeneous datasets

## Results

Let the *global accuracy* be the accuracy of a model on the homogeneous (balanced) server-side dataset; and the *local accuracy* the accuracy of a model on a local dataset, i.e. a cluster heterogeneous (unbalanced) dataset. 
When we want to test a model on the local datasets, the local accuracy is measured on each local dataset and then the average is considered.
When we want to test the local / clusters models on the server dataset, each one has its own accuracy and even in this case the average is computed.

In the following plots these metrics are taken into account:
- **avg local acc - clusters models**: each local model measures the accuracy on its local dataset, and then the average is considered. No cross measures (for instance the cluster 1 model on the cluster 2 dataset) is considered;
- **global acc - avg softmax outputs**: given a server dataset image, each cluster model predicts its label. Then the average of the softmax outputs (the last layer of each classification model) is considered and the argmax of that unique vector is used to predict the label;
- **global acc - genie**: this is the expected upper bound that we want to reach on the server dataset. For each image we look at its label, and if there is a cluster with a local dataset unbalanced on that label (i.e. in that dataset there are a lot of images with that label), its model is used to predict it (note that it could be wrong). Otherwise, if there are not any dataset unbalanced on that label, the softmax average is considered, like the "global acc - avg softmax outputs" case;
- **avg local acc - avg softmax outputs**: the average softmax outputs method on each local dataset, and then the average of each local accuracy is considered.
- **global acc - avg local models weights**: the accuracy on the server dataset, computed my a model whose weights are the average of the clusters models weights, for each communication round. There is no weights update: the server does not share the model with the clusters, it only receives the updated weights.
- **avg local acc - avg local models weights** and **avg local acc - server fedavg with cluster models** are the same as before, but computing the accuracies on the local datasets and then averaging them.
- the other metrics indicated are not interesting for the purposes of our analysis.

In order to make a comparation on the same dataset the following plots are divided in global accuracy and local accuracy.
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

![model](https://user-images.githubusercontent.com/62892813/135764394-38460fdc-254b-4e1a-83e6-6d8dc1416bc8.png)

## Performance

<img src = "https://user-images.githubusercontent.com/62892813/135764484-38954d77-2a08-43c8-b3e9-047a68247f07.png" width = "315" height = "210"><img src = "https://user-images.githubusercontent.com/62892813/135764491-fe615a00-0a2b-4b12-9c97-c06ecc22c73f.png" width = "315" height = "210">

<img src = "https://user-images.githubusercontent.com/62892813/135764511-e044db59-6a0b-4467-b109-2acea957ee26.png" width = "630" height = "420">

### Is the aggregator able to detect?
The aggregator performance are satisfactory, but why? Is the aggregator able to give at the right clusters, the bigger weight? Note that the aggregator is trained to classify in the right way the given images, as the local models.
To verify this, it is sufficient to build an intermediate model 
```python
intermediate_model = Model(inputs=server_agg.model.input, outputs=server_agg.model.layers[2].output)
```
(where `server_agg.model` is the aggregator, and `server_agg.model.layers[2].output` is the output of the dense layer that gives the weights). Note that to perform this test, the aggregator is modified with a sigmoid activation function to the intermediate dense layer. Despite this, overall performance does not change. 
Let's visualize, for each label, if a bigger weight is given to the output of a model trained on a cluster which is the 'expert' of images of that label. For instance: if the i-th local model is trained on a dataset with 80% of images of the digit, let's say, 5, then for an image of the figit 5 the aggregator should have the i-th weight of the dense layer output bigger than the others.
Here the average weights vector for each label, with a red bar on the cluster with that label as predominant.

<img src = "https://user-images.githubusercontent.com/62892813/135819795-06a780e3-adb0-4ade-a724-a1b06c4bf262.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819792-d0e9ec07-a48b-4758-b9d1-147e8a3a615f.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819791-b4a36592-095e-49f1-bd70-6e8dc13fb5d8.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819790-2b0888a6-20b4-4eb3-a930-27dd93c487eb.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819787-05520dc0-02f8-4643-bd5d-9e749ec07c61.png" width = "150" height = "110">

<img src = "https://user-images.githubusercontent.com/62892813/135819805-1a67943b-6e0f-45cb-96fe-23f41eb32fec.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819804-9a2dfd53-f683-4f48-bfbc-d946065c0102.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819802-bd110492-e61f-46de-81fa-839f56e992bb.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819799-1ccc465d-1bad-4c24-818e-f53681291267.png" width = "150" height = "110"><img src = "https://user-images.githubusercontent.com/62892813/135819797-e937e7df-29ca-4461-bdd1-1caf0f11cea1.png" width = "150" height = "110">

### Mathematical formulation
<!-- 
If $\mathcal{C}$ is the set of clusters, a classification model, parametrized by $\boldsymbol{\vartheta}_{i\in [1, |\mathcal{C}|]}$, returns a probability distribution over the classes $q_{\boldsymbol{\vartheta}_i}(\boldsymbol{y}|\boldsymbol{x})$. Then the \emph{attention polling mapping} $h_{\boldsymbol{\delta}}$ of the aggregator, which receives the image $\boldsymbol{x}$, has to return a weight for each cluster model $\boldsymbol{\vartheta}_i$, that can be seen as the likelihood of the image under the cluster distribution: 
    \[
        \boldsymbol{h}_{\boldsymbol{\delta}}(\boldsymbol{x})=\big(p_{\mathcal{D}_i}(\boldsymbol{x})\big)_{i=1}^{|\mathcal{C}|},
    \]
    and this is the reason why the layer has the sigmoid activation function, so the weighted sum performed is not a convex combination, i.e. the weights do not sum to one
    \[
    \sum_{i=1}^{|\mathcal{C}|}h_{\boldsymbol{\delta}}^{(i)}(\boldsymbol{x})\in [0, |\mathcal{C}|]
    \]
    and this is done because we do not have any guarantee about the local distributions, so we do not know if $\mathcal{D}_i\neq \mathcal{D}_j, \forall i\neq j$.
    
    The server architecture $m_{\boldsymbol{A, b, \delta}}$ uses $h_{\boldsymbol{\delta}}(\boldsymbol{x})$ to compute a weighted sum of the cluster outputs:
    \[
        m_{\boldsymbol{A, b, \delta};\boldsymbol{\vartheta}_1,\dots,\boldsymbol{\vartheta}_{|\mathcal{C}|}}
        (\boldsymbol{y}|\boldsymbol{x})
        =
        \text{softmax}\bigg( \boldsymbol{A} \cdot \sum_{i=1}^{|\mathcal{C}|} \big(
        h_{\boldsymbol{\delta}}^{(i)}(\boldsymbol{x})\cdot q_{\boldsymbol{\vartheta}_i}(\boldsymbol{y}|\boldsymbol{x})\big)
        + \boldsymbol{b}
        \bigg)
    \]
    where $\boldsymbol{A, b, \delta}$ are trainable, and $\boldsymbol{\vartheta}_1,\dots,\boldsymbol{\vartheta}_{|\mathcal{C}|}$ are given.
-->
![Cattura](https://user-images.githubusercontent.com/62892813/137509493-d1437388-6c1b-4283-bddd-ea9d9eb8291f.JPG)


## References

- [The Communication-Aware Clustered Federated Learning Problem](https://ieeexplore.ieee.org/document/9174245)
- [Federated Mixture of Experts](https://arxiv.org/abs/2107.06724)
- [FedMD: Heterogeneous Federated Learning via Model Distillation](https://towardsdatascience.com/fedmd-heterogeneous-federated-learning-via-model-distillation-e84676183eb4)
- [Specialized federated learning using a mixture of experts](https://arxiv.org/pdf/2010.02056.pdf)
- others...
