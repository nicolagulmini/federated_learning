from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from numpy.random import permutation

class cluster:
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
    def transfer_cluster_model_to_users(self):
        for user in self.users:
            user.model.set_weights(self.model.get_weights())
        return
    def transfer_cluster_estimation_to_users(self):
        for user in self.users:
            user.estimation.set_weights(self.estimation.get_weights())
        return
    def assign_data_from_cluster_to_users(self):
        # if the cluster already has got the data, an assignment is performed in order to give at each user a uniform portion of them.
        if len(self.train_data) == 0 or len(self.test_data) == 0:
            print("No data in the cluster " + str(self.number) + ".")
            return
        # note that the users have only training data because the test data are at the cluster level, to make the computation easier
        amount_of_user_data = int(self.train_data['images'].shape[0] / self.number_of_users())
        xtrain = self.train_data['images']
        ytrain = self.train_data['labels']
        for i in range(len(self.users)):
            self.users[i].set_data({'images': xtrain[i*amount_of_user_data:(i+1)*amount_of_user_data], 'labels': ytrain[i*amount_of_user_data:(i+1)*amount_of_user_data]})
            # issue: the first and the last image of the user i is shared with, respectively, user i-1 and user i+1...
            print("Set data for user " + str(self.users[i].name) + " of cluster " + str(self.number))
            print("The shape of data is " + str(self.users[i].data['images'].shape))
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
    
class define_model_mnist():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(10, activation='softmax'))
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
    
    def assign_users_to_clusters(number_of_users, number_of_clusters):
        # assigns uniformly and in a sorted way, the given users to the given clusters. Returns a list of clusters. This step has to be done before the data assignment.
        if not number_of_users % number_of_clusters == 0:
            print("It is better to have the same number of users for each cluster, to make the computation easier. This issue will be solved.")
            return
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
    
    
        
        