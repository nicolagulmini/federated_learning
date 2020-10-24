"""
Created on Wed Oct 14 23:43:24 2020

@author: Nicola Gulmini
@mail: nicolagulmini@gmail.com or nicola.gulmini@studenti.unipd.it
"""
# tensorflow 2.0 and keras 2.3 

from matplotlib import pyplot as plt
import random as rnd
import numpy as np
import tensorflow as tf
import collections

from utils_torch import *

from keras.datasets import cifar10
from keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers import UpSampling2D
#from tensorflow.keras.layers import Activation
#from tensorflow.keras.regularizers import l2

tf.random.set_seed(42)
np.random.seed(42)
rnd.seed(42)

classes = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck",
}

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
        
class user_information:
    def __init__(self, name, cluster):
        self.name = name
        self.cluster = cluster
    def set_data(self, data):
        self.data = data
    def set_accuracy(self, accuracy):
        self.accuracy = accuracy
    def get_accuracy(self):
        return self.accuracy
    def set_model(self, model):
        self.model = model
    def get_model(self):
        return self.model
    def set_estimation(self, estimation):
        self.estimation = estimation
    def get_estimation(self):
        return self.estimation
    def estimation_size(self): # magari non ha senso dato che a me interesseranno i non nulli, quindi i K in pratica !!
        return 1 # restituisce il size della rete per la stima
        #return number_of_parameters(self.estimation.get_weights())
    def model_size(self):
        return number_of_parameters(self.get_model().get_weights())
    def buffer_size(self):
        return self.model_size()+self.estimation_size()
    
class top_k:
    def __init__(self, K):
        self.k = K
        self.params = [0 for _ in range(self.k)]
    def check(self, param): # check if there is param in the list
        for el in self.params:
            if param == el:
                return True
        return False
    def check_and_update(self, param): # if the param is greater of at least one element of the least, insert into it and update in order to mantain the list sorted
        con = param
        for i in range(self.k):
            tmp = self.params[i]
            if con > tmp:
                self.params[i] = con
                con = tmp
    def get_params(self):
        return self.params
    
class define_model():
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2)) 
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        opt = SGD(lr=0.001, momentum=0.9)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
def load_preprocessed_cifar10_ds():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    num_classes = len(classes)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    Y_train = to_categorical(Y_train, num_classes)
    Y_test = to_categorical(Y_test, num_classes)
    #print('X_train shape:', X_train.shape)
    #print(X_train.shape[0], 'train samples')
    #print(X_test.shape[0], 'test samples')
    #print('Y_train shape:', Y_train.shape)
    #print('Y_test shape:', Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def create_dictionary_from_dataset(dataset):
    division = collections.defaultdict(list)
    for v, k in dataset:
        label = int(np.argmax(k)) # or np.where() because of one-hot
        division[label].append(v)
    return division

def ds_division(X, division, bias_factor, number_of_clusters, fav_classes):
    remaining_weights = (1-bias_factor)/9
    cluster_img = int(len(X)/number_of_clusters)
    cluster_ds = collections.OrderedDict()
    for i in range(number_of_clusters):
        fav_class = fav_classes[i]
        classes_weights = [remaining_weights for _ in classes]
        classes_weights[fav_class] = bias_factor
        #print('Adding data for cluster', i)
        cluster_X = np.zeros((cluster_img, 32, 32, 3))
        cluster_Y = np.zeros((cluster_img, 10))
        for j in range(cluster_img):
            chosen_class = rnd.choices(list(range(10)), classes_weights)[0] # [0] because it is a 1-dimensional vector
            images_from_class = division[chosen_class]
            chosen_image = images_from_class[np.random.randint(0, len(images_from_class))] # possibile ripetizione di dati
            cluster_X[j] = chosen_image
            cluster_Y[j] = to_categorical(chosen_class, len(classes))
        cluster_ds[i] = collections.OrderedDict((('labels', cluster_Y), ('images', cluster_X)))
    return cluster_ds

def assign_users_to_clusters_randomly(users_ids, number_of_clusters):
    tmp_ids = users_ids
    number_of_users = len(tmp_ids)
    users_per_cluster = int(number_of_users/number_of_clusters)
    clusters = []
    for i in range(number_of_clusters):
        tmp_cluster = cluster(number=i)
        for _ in range(users_per_cluster):
            user = rnd.choices(tmp_ids)[0]
            tmp_ids.remove(user)
            tmp_user = user_information(user, i)
            tmp_cluster.add_user(tmp_user)
        clusters.append(tmp_cluster)
        print('Users in cluster', tmp_cluster.number, 'are:', [u.name for u in tmp_cluster.users])
        del tmp_cluster
    return clusters

def summarize_diagnostics(history):
	# plot loss 
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
	 # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()
    plt.close()
    
def train_validation_split(X_train, Y_train):
    train_length = len(X_train)
    validation_length = int(train_length / 4)
    X_validation = X_train[:validation_length]
    X_train = X_train[validation_length:]
    Y_validation = Y_train[:validation_length]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation

'''
def training_in_function_of_bias(X_train, Y_train, X_test, Y_test, division, number_of_clusters, number_of_users, epochs, batch):
    bias_factors = []
    average_accuracy = []
    clusters = assign_users_to_clusters_randomly(list(range(number_of_users)), number_of_clusters) # per avere sempre gli stessi utenti negli stessi cluster
    for coeff in range(12):
        bias_factor = 0.1 + 0.025*coeff
        bias_factors.append(bias_factor)
        
        clusters_train_datasets = ds_division(X_train, division, bias_factor, number_of_clusters)
        
        for i in range(len(clusters_train_datasets)):
            clusters[i].set_data(clusters_train_datasets[i])
        
        # the trainset division among cluster users
        for cluster in clusters:
            cluster_ds = cluster.data
            cluster_X = cluster_ds['images']
            cluster_Y = cluster_ds['labels']
            shuffler = np.random.permutation(len(cluster_X))
            cluster_X = cluster_X[shuffler]
            cluster_Y = cluster_Y[shuffler]
            size_of_user_ds = int(len(cluster_X) / cluster.number_of_users())
            for i in range(cluster.number_of_users()):
                X_user = cluster_X[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
                Y_user = cluster_Y[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
                user = cluster.users[i]
                user_data = collections.OrderedDict((('labels', Y_user), ('images', X_user)))
                user.set_data(user_data)
                #user.set_model(define_model())
        
        average_accuracy_tmp = 0
        for i in range(number_of_clusters):
            cluster_i = clusters[i]
            users_of_cluster_i = cluster_i.users
            for user in users_of_cluster_i:
                #users_model = user.get_model() # cosi posso continuare il training
                # in questo caso voglio resettare il modello ogni volta, quindi faccio cosi
                users_model = define_model()
                print('Training for user', user.name)
                user_data = user.data
                X_train_u = user_data['images']
                Y_train_u = user_data['labels']
                shuffler = np.random.permutation(len(X_train_u))
                X_train_u = X_train_u[shuffler]
                Y_train_u = Y_train_u[shuffler]     
                X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)
                history = users_model.model.fit(X_train_u, Y_train_u, epochs=epochs, batch_size=batch, verbose=0, validation_data=(X_validation_u, Y_validation_u))
                loss, accuracy = users_model.model.evaluate(X_test, Y_test)
                #users_model.model.save("utente"+str(user)+"_bias"+str(bias_factor)+".h5")
                #summarize_diagnostics(history)
                user.set_model(users_model.model)
                user.set_accuracy(accuracy)
                average_accuracy_tmp += accuracy
                print('Test loss for user', user.name, 'of cluster', i, 'is', loss)
                print('Test accuracy for user', user.name, 'of cluster', user.cluster, 'is', user.get_accuracy())
                del users_model
        average_accuracy.append(average_accuracy_tmp/number_of_users)
        print(average_accuracy)
        print(bias_factors)
    return bias_factors, average_accuracy
'''

def train_user_model(user, cluster, epochs, batch): # sarebbe meglio dare agli utenti direttamente l'oggetto cluster anziche' il numero...
    print('Training for user', user.name)
    user_model = user.get_model()
    X_test = cluster.test_data['images']
    Y_test = cluster.test_data['labels']
    _, accuracy = user_model.evaluate(X_test, Y_test)
    print('Test accuracy BEFORE training for user', user.name, 'of cluster', user.cluster, 'is', accuracy)
    user_data = user.data
    X_train_u = user_data['images']
    Y_train_u = user_data['labels']
    shuffler = np.random.permutation(len(X_train_u))
    X_train_u = X_train_u[shuffler]
    Y_train_u = Y_train_u[shuffler]     
    X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)
    history = user_model.fit(X_train_u, Y_train_u, epochs=epochs, batch_size=batch, verbose=0, validation_data=(X_validation_u, Y_validation_u))
    _, accuracy = user_model.evaluate(X_test, Y_test)
    #summarize_diagnostics(history)
    user.set_model(user_model)
    user.set_accuracy(accuracy) # may be useless
    print('Test accuracy AFTER training for user', user.name, 'of cluster', user.cluster, 'is', user.get_accuracy())

def initialize_models(clusters, server):
    w = server.model.get_weights()
    # inizialmente condivide lo stesso modello con tutti i cluster, ma poi si specializza
    for cluster in clusters:
        tmp_cluster_model = define_model() # these steps are in order to create one model for each cluster and user, instead of the same model shared among all
        tmp_cluster_model.model.set_weights(w)
        cluster.set_model(tmp_cluster_model.model)
        del tmp_cluster_model
        cluster.set_model(server.model) # penso sia inutile
        # estimation
        tmp_cluster_estimation_model = NICE(data_dim=3072, num_coupling_layers=4)
        cluster.set_estimation(tmp_cluster_estimation_model)
        for user in cluster.users:
            # model
            tmp_user_model = define_model()
            tmp_user_model.model.set_weights(cluster.model.get_weights())
            user.set_model(tmp_user_model.model)
            del tmp_user_model
            # estimation
            tmp_user_estimation_model = NICE(data_dim=3072, num_coupling_layers=4)
            tmp_user_estimation_model.load_state_dict(cluster.estimation.state_dict())
            user.set_estimation(tmp_user_estimation_model)
            del tmp_user_estimation_model
            
            
def transfer_cluster_model_to_users(cluster):
    for user in cluster.users:
        user.get_model().set_weights(cluster.model.get_weights())
        
def transfer_cluster_estimation_to_users(cluster):
    for user in cluster.users:
        user.get_estimation().load_state_dict(cluster.estimation.state_dict())
        
def top_k_sparsification_weights_tf(model, K): 
    weights = model.get_weights()
    top_k_params = top_k(K)
    for array in weights:
        tmp = np.reshape(array, [-1])
        for el in tmp:
            top_k_params.check_and_update(el)
    new_weights = []
    for array in weights:
        tmp = np.reshape(array, [-1])
        for i in range(len(tmp)):
            if not top_k_params.check(tmp[i]):
                tmp[i] = 0
        tmp = np.reshape(tmp, array.shape)
        new_weights.append(tmp)
    return new_weights

def number_of_parameters(model_weights):
    w = 0
    for i in range(len(model_weights)):
        tmp = 1
        for el in model_weights[i].shape:
            tmp = tmp * el
        w = w + tmp
    return w

def plot_cluster_performances(clusters, iterations):
    list_iterations = [i for i in range(iterations)]
    for cluster in clusters:
        cl_acc = []
        X_test = cluster.test_data['images']
        Y_test = cluster.test_data['labels']
        for i in range(iterations):
            tmp_model = tf.keras.models.load_model("cluster"+str(cluster.number)+"_iter"+str(i)+".h5")
            _, accuracy = tmp_model.evaluate(X_test, Y_test)
            cl_acc.append(accuracy)
        plt.plot(list_iterations, cl_acc) # fai una legenda
    plt.title('Clusters\' models performances')
    plt.xlabel('iterations')
    plt.ylabel('clusters\' accuracies')
    plt.show()