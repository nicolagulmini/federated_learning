from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from numpy.random import permutation
from numpy import add
from numpy import subtract

class cluster:
    # this class realizes a cluster of users 
    
    def __init__(self, number):
        self.users = []
        self.number = number
        
    def number_of_users(self):
        return len(self.users)
    
    def add_user(self, user):
        self.users.append(user)
        
    def set_train_data(self, train_data):
        self.train_data = train_data
        
    def set_test_data(self, test_data):
        self.test_data = test_data
        
    def set_model(self, model):
        self.model = model
        
    def set_estimation(self, estimation):
        self.estimation = estimation
        
    def get_model(self):
        return self.model
    
    def get_estimation(self):
        return self.estimation
    
    def print_information(self):
        print("Cluster number " + str(self.number) + ". User ids: " + str([user.name for user in self.users]))
        
    def initialize_models(self):
        # inizialize the models
        classification_model = define_model_mnist()
        estimation_model = define_autoencoder_mnist()
        self.set_model(classification_model.model)
        self.set_estimation(estimation_model.model)
        return
    
    def cluster_to_users_initialization(self):
        # to copy the initial models from the cluster to its users
        for user in self.users:
            user.set_model(clone_model(self.model))
            user.set_estimation(clone_model(self.estimation))
            user.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            user.estimation.compile(optimizer='adam', loss='binary_crossentropy')
        return
    
    def transfer_cluster_model_to_users(self):
        # propagate from cluster to its users the classification model
        if self.number_of_users() == 0:
            print("No users in the cluster " + str(self.number))
        for user in self.users:
            user.model.set_weights(self.model.get_weights())
        return
    
    def transfer_cluster_estimation_to_users(self):
        # propagate from cluster to its users the estimation model
        if self.number_of_users() == 0:
            print("No users in the cluster " + str(self.number))
        for user in self.users:
            user.estimation.set_weights(self.estimation.get_weights())
        return
    
    def set_cluster_model_weights(self, weights):
        # set the cluster model weights as given. The given weights have to be compatible (same shape as the weights of the current cluster model)
        self.model.set_weights(weights)
        return
    
    def assign_data_from_cluster_to_users(self, verbose):
        # if the cluster already has got the data, an assignment is performed in order to give at each user a uniform portion of them.
        if len(self.train_data) == 0 or len(self.test_data) == 0 or self.number_of_users() == 0:
            print("No users and/or data in the cluster " + str(self.number))
            return
        # note that the users have only training data because the test data are at the cluster level, to make the computation easier
        amount_of_user_data = int(self.train_data['images'].shape[0] / self.number_of_users())
        xtrain = self.train_data['images']
        ytrain = self.train_data['labels']
        for i in range(len(self.users)):
            self.users[i].set_data({'images': xtrain[i*amount_of_user_data:(i+1)*amount_of_user_data], 'labels': ytrain[i*amount_of_user_data:(i+1)*amount_of_user_data]})
            # issue: the first and the last image of the user i is shared with, respectively, user i-1 and user i+1...
            if not verbose == 0:
                print("Set data for user " + str(self.users[i].name) + " of cluster " + str(self.number))
                print("The shape of data is " + str(self.users[i].data['images'].shape))
        return
    
    def update_cluster_classification_model(self):
        # this method updates the cluster classification model using the user ones
        w = [user.get_model().get_weights() for user in self.users]
        
        print("Print weights of the users models:") # debug
        for weights in w: # debug
            print(weights[0]) # debug
            
        # compute the weight for the update
        fracs = [len(user.data['labels']) for user in self.users]
        tot_data = sum(fracs)
        fracs = [f/tot_data for f in fracs]
        
        resulting_weights = self.model.get_weights()
        
        print("Initial weights of the cluster model:") # debug
        print(resulting_weights) # debug
        
        for layer in range(len(resulting_weights)):
            resulting_weights[layer] = add(resulting_weights[layer], sum([subtract(w[i][layer], resulting_weights[layer])*fracs[i] for i in range(len(self.users))]))
        self.model.set_weights(resulting_weights)
        
        print("FINAL weights of the cluster model:") # debug
        print(self.model.get_weights()) # debug
        return        
        
class user_information:
    def __init__(self, name, cluster):
        self.name = name
        self.cluster = cluster
    def set_data(self, data):
        self.data = data
    def set_model(self, model):
        self.model = model
    def get_model(self):
        return self.model
    def set_estimation(self, estimation):
        self.estimation = estimation
    def get_estimation(self):
        return self.estimation
    
    def train(self, epochs, batch, verbose):
        # train the local user model on the local user dataset and compute the accuracy on the local cluster dataset
        
        # to_categorical is from keras.utils
        x_train, y_train, x_val, y_val = federated_setup.train_validation_split(self.data['images'], to_categorical(self.data['labels'], 10))
        
        if not verbose == 0:
            accuracy = self.model.evaluate(self.cluster.test_data['images'], to_categorical(self.cluster.test_data['labels'], 10))[1]
            print("Accuracy of the user " + str(self.name) + " of the cluster " + str(self.cluster.number) + " BEFORE the training is " + str(accuracy))
        
        # training
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch, verbose=verbose, validation_data=(x_val, y_val))
        
        if not verbose == 0:
            accuracy = self.model.evaluate(self.cluster.test_data['images'], to_categorical(self.cluster.test_data['labels'], 10))[1]
            print("Accuracy of the user " + str(self.name) + " of the cluster " + str(self.cluster.number) + " AFTER the training is " + str(accuracy))
        
        validation_accuracy = self.model.evaluate(x_val, y_val, verbose=0)[1]
        return validation_accuracy
            
class define_model_mnist():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                                       bias_regularizer=regularizers.l2(1e-4),
                                                       activity_regularizer=regularizers.l2(1e-5)))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class define_autoencoder_mnist():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(784, activation='sigmoid'))
        self.model.add(Reshape((28, 28)))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
class federated_setup:
    # in this class there are all the methods that use the previous classes and are necessary to build the federated setup
    
    def initialize_users_to_clusters(number_of_users, number_of_clusters):
        # assigns uniformly and in a sorted way, the given users to the given clusters. Returns a list of clusters. This step has to be done before the data assignment.
        if not number_of_users % number_of_clusters == 0:
            print("It is better to have the same number of users for each cluster, to make the computation easier. This issue will be solved.")
            return []
        users_per_cluster = int(number_of_users/number_of_clusters)
        clusters_list = []
        user_id = 0
        for _ in range(number_of_clusters):
            tmp_cluster = cluster(_)
            for __ in range(users_per_cluster):
                tmp_user = user_information(user_id, tmp_cluster)
                tmp_cluster.add_user(tmp_user)
                user_id += 1
            clusters_list.append(tmp_cluster)
        # to recap
        for c in clusters_list:
            c.print_information()
        return clusters_list
    
    def assign_dataset_to_clusters(list_of_clusters, list_of_training_dictionary, list_of_test_dictionary):
        # given the already divided datasets, in a dictionary with 'images' and 'labels' for each cluster, this method modifies each cluster of the list setting the train and the test data
        if not ((len(list_of_clusters) == len(list_of_training_dictionary)) and (len(list_of_training_dictionary) == len(list_of_test_dictionary))):
            print("The number of clusters, training datasets and test sets have to be the same! Also, provide the datasets as dictionary with images and labels.")
            return
        for _ in range(len(list_of_clusters)):
            c = list_of_clusters[_]
            train_data = list_of_training_dictionary[_]
            test_data = list_of_test_dictionary[_]
            xtrain = train_data['images']
            ytrain = train_data['labels']
            xtest = test_data['images']
            ytest = test_data['labels']
            shuffler = permutation(xtrain.shape[0]) # from numpy
            xtrain = xtrain[shuffler]
            ytrain = ytrain[shuffler]    
            c.set_train_data({'images': xtrain, 'labels': ytrain})
            c.set_test_data({'images': xtest, 'labels': ytest})
            print("Set data for cluster " + str(c.number))
        # the cluster of the list are modified 
        print("Done.")
        return
    
    def assign_clusters_data_to_users(list_of_clusters, verbose):
        # for each cluster, assign the training data to its users
        for cluster in list_of_clusters:
            cluster.assign_data_from_cluster_to_users(verbose)
        return
            
    def server_to_cluster_classification(list_of_clusters, server_classification_model):
        # assign a classification model to the clusters
        # pay attention: each cluster has its own model, so the weights are copied
        # this method has to be called AFTER initialize_classification_model()
        for cluster in list_of_clusters:
            cluster.model.set_weights(server_classification_model.get_weights())
        return
       
    def train_validation_split(x_train, y_train):
        train_length = len(x_train)
        shuffler = permutation(train_length) # from numpy
        x_train = x_train[shuffler]
        y_train = y_train[shuffler]    
        validation_length = int(train_length / 4)
        x_val = x_train[:validation_length]
        x_train = x_train[validation_length:]
        y_val = y_train[:validation_length]
        y_train = y_train[validation_length:]
        return x_train, y_train, x_val, y_val
    
    def sparsificate(model, k): # k is a fraction of parameters to save: k in [0,1]
        #modello.model.count_params()
        return 0 # return the same model but sparse!!
