# Federated Learning

## federated_mnist 
The federated_mnist folder contains:
- 9 heterogeneous datasets (training and test)
- in each folder there is a .csv file with name of images and related label
- the heterogeneity is realized in this way: 50% of the images are of the same label, the other 50% are taken randomly from the original dataset
- each dataset contains 4000 images, so each cluster contains exactly 8000 images. Some of these could be in common between more clusters.
This dataset has been created with dataset_split.py.
