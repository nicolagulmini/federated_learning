"""
Created on Wed Oct 14 23:43:24 2020

@author: Nicola Gulmini
@mail: nicolagulmini@gmail.com or nicola.gulmini@studenti.unipd.it
"""
# tensorflow 2.0 and keras 2.3 

# general
from matplotlib import pyplot as plt
import random as rnd
import numpy as np
import collections
import os

# tf and keras
import tensorflow as tf
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

# pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.distributions import Distribution, Uniform

tf.random.set_seed(42)
np.random.seed(42)
rnd.seed(42)
torch.manual_seed(42)

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
        
class CouplingLayer(nn.Module):
  def __init__(self, data_dim, hidden_dim, mask, num_layers=3):
    super().__init__()

    assert data_dim % 2 == 0

    self.mask = mask

    modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
    for i in range(num_layers - 2):
      modules.append(nn.Linear(hidden_dim, hidden_dim))
      modules.append(nn.LeakyReLU(0.2))
    modules.append(nn.Linear(hidden_dim, data_dim))

    self.m = nn.Sequential(*modules)

  def forward(self, x, logdet, invert=False):
    if not invert:
      x1, x2 = self.mask * x, (1. - self.mask) * x
      y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
      return y1 + y2, logdet

    # Inverse additive coupling layer
    y1, y2 = self.mask * x, (1. - self.mask) * x
    x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
    return x1 + x2, logdet

class ScalingLayer(nn.Module):
  
  def __init__(self, data_dim):
    super().__init__()
    self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

  def forward(self, x, logdet, invert=False):
    log_det_jacobian = torch.sum(self.log_scale_vector)

    if invert:
        return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

    return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian

class LogisticDistribution(Distribution):
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))
    
    def prob(self, x):
        return torch.exp(torch.sum(self.log_prob(x)))

    def sample(self, size):
        z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)
        return torch.log(z) - torch.log(1. - z)
    
class NICE(nn.Module):
  def __init__(self, data_dim, num_coupling_layers=3):
    super().__init__()

    self.data_dim = data_dim

    # alternating mask orientations for consecutive coupling layers
    masks = [self._get_mask(data_dim, orientation=(i % 2 == 0)) for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([CouplingLayer(
        data_dim=data_dim, 
        hidden_dim=30, # avevo provato con 100 e nel paper ne mettono 200!
        mask=masks[i], 
        num_layers=3)
        for i in range(num_coupling_layers)])

    self.scaling_layer = ScalingLayer(data_dim=data_dim)
    self.prior = LogisticDistribution()

  def forward(self, x, invert=False):
    if not invert:
      z, log_det_jacobian = self.f(x)
      log_likelihood = torch.sum(self.prior.log_prob(z), dim=1) + log_det_jacobian
      return z, log_likelihood

    return self.f_inverse(x)

  def f(self, x):
    z = x
    log_det_jacobian = 0
    for i, coupling_layer in enumerate(self.coupling_layers):
      z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
    z, log_det_jacobian = self.scaling_layer(z, log_det_jacobian)
    return z, log_det_jacobian

  def f_inverse(self, z):
    x = z
    x, _ = self.scaling_layer(x, 0, invert=True)
    for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
      x, _ = coupling_layer(x, 0, invert=True)
    return x

  def sample(self, num_samples):
    z = self.prior.sample([num_samples, self.data_dim]).view(num_samples, self.data_dim)
    print(z)
    return self.f_inverse(z)

  def _get_mask(self, dim, orientation=True):
    mask = np.zeros(dim)
    mask[::2] = 1.
    if orientation:
      mask = 1. - mask # flip mask orientation
    mask = torch.tensor(mask)
    return mask.float()
        
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

def initialize_models(clusters, server, fraction):
    w = top_k_sparsificate_model_weights_tf(server.model.get_weights(), fraction)
    # inizialmente condivide lo stesso modello con tutti i cluster, ma poi si specializza
    for cluster in clusters:
        tmp_cluster_model = define_model() # these steps are in order to create one model for each cluster and user, instead of the same model shared among all
        tmp_cluster_model.model.set_weights(w)
        cluster.set_model(tmp_cluster_model.model)
        del tmp_cluster_model
        cluster.set_model(server.model) # penso sia inutile
        # estimation
        tmp_cluster_estimation_model = NICE(data_dim=3072, num_coupling_layers=3)
        cluster.set_estimation(tmp_cluster_estimation_model)
        for user in cluster.users:
            # model
            tmp_user_model = define_model()
            tmp_user_model.model.set_weights(cluster.model.get_weights())
            user.set_model(tmp_user_model.model)
            del tmp_user_model
            # estimation
            tmp_user_estimation_model = NICE(data_dim=3072, num_coupling_layers=3)
            tmp_user_estimation_model.load_state_dict(cluster.estimation.state_dict())
            user.set_estimation(tmp_user_estimation_model)
            del tmp_user_estimation_model
            
def top_k_sparsificate_model_weights_tf(weights, fraction):
    tmp_list = []
    for el in weights:
        lay_list = el.reshape((-1)).tolist()
        tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list.sort(reverse=True)
    k_th_element = tmp_list[int(fraction*552874)-1]
    new_weights = []
    for el in weights:
        original_shape = el.shape
        reshaped_el = el.reshape((-1))
        for i in range(len(reshaped_el)):
            if abs(reshaped_el[i]) < k_th_element:
                reshaped_el[i] = 0.0
        new_weights.append(reshaped_el.reshape(original_shape))
    return new_weights
            
def train_estimation(model, epochs, dataloader):
    model.train()
    opt = optim.Adam(model.parameters())
    for i in range(epochs):
        mean_likelihood = 0.0
        num_minibatches = 0

        for batch_id, x in enumerate(dataloader):
            x = x.view(-1, 3072) + torch.rand(3072) / 256.
            x = torch.clamp(x, 0, 1) # serve per limitare 
            z, likelihood = model(x.float()) # ho messo .float() perche mi dava qualche problema
            #print(likelihood)
            loss = -torch.mean(likelihood) # NLL
            loss.backward()
            opt.step()
            model.zero_grad()
            mean_likelihood -= loss
            num_minibatches += 1

        mean_likelihood /= num_minibatches
        print('Epoch {} completed. Log Likelihood: {}'.format(i, mean_likelihood))   
            
            
def transfer_cluster_model_to_users(cluster):
    for user in cluster.users:
        user.get_model().set_weights(cluster.model.get_weights())
        
def transfer_cluster_estimation_to_users(cluster):
    for user in cluster.users:
        user.get_estimation().load_state_dict(cluster.estimation.state_dict())

def number_of_parameters(model_weights):
    w = 0
    for i in range(len(model_weights)):
        tmp = 1
        for el in model_weights[i].shape:
            tmp = tmp * el
        w = w + tmp
    return w

def plot_cluster_performances(clusters, iterations, percentage):
    list_iterations = [i for i in range(iterations)]
    for cluster in clusters:
        cl_acc = []
        X_test = cluster.test_data['images']
        Y_test = cluster.test_data['labels']
        for i in range(iterations):
            tmp_model = tf.keras.models.load_model("cluster"+str(cluster.number)+"_"+str(percentage)+"sparse_iter"+str(i)+".h5")
            _, accuracy = tmp_model.evaluate(X_test, Y_test)
            cl_acc.append(accuracy)
        plt.plot(list_iterations, cl_acc) # fai una legenda
    plt.title('Clusters\' models performances')
    plt.xlabel('iterations')
    plt.ylabel('clusters\' accuracies')
    plt.show()
    
def server_aggregation(X_test, Y_test, clusters, percent_classification, percent_estimation, cluster_iterations, est_iterations):
    true_predictions = 0
    for t in range(len(X_test)):
        test_img = X_test[t]
        test_label = Y_test[t]
        true = np.argmax(test_label)
        #print('true:', classes[true])
        prediction_vectors = []
        log_probs = []
        for cluster in clusters:
            tmp_model = tf.keras.models.load_model("./classification"+str(percent_classification)+"/cluster"+str(cluster.number)+"_iter"+str(cluster_iterations-1)+".h5")
            tmp_est = NICE(data_dim=3072, num_coupling_layers=3)
            tmp_est.load_state_dict(torch.load("./estimation"+str(percent_estimation)+"/"+"estimation_cluster"+str(cluster.number)+"_iter"+str(est_iterations-1)+".pt"))
            log_prob= tmp_est.forward(torch.from_numpy(test_img.reshape((3072))))[1]-tmp_est.f(torch.from_numpy(test_img.reshape((3072))))[1]
            log_probs.append(log_prob.detach().numpy().reshape((-1)))
            pred = tmp_model.predict(test_img.reshape((1, 32, 32, 3)))
            prediction_vectors.append(pred.reshape(-1))
        print('log_probs:', [float(el) for el in log_probs])
        a = float(max(log_probs))
        log_meno_a = [float(el-a) for el in log_probs]
        sum_exp = 0
        for el in log_meno_a:
            sum_exp += np.exp(el)
        comp = a+np.log(sum_exp)
        alpha = [np.exp(el-comp) for el in log_probs]
        print('alpha:', [float(el) for el in alpha])
        print('')
        final_vector = [alpha[i]*prediction_vectors[i] for i in range(len(clusters))]
        rule = final_vector[0]
        
        for i in range(1, len(final_vector)):
            rule += final_vector[i]
        predicted = np.argmax(rule)
        #print('predicted:', classes[predicted])
        if predicted == true:
            true_predictions += 1
    return true_predictions/len(X_test)*100

def find_min_topk(list_of_resized_tensors, k):
    maxi_tensor = torch.cat(list_of_resized_tensors)
    topk = torch.topk(maxi_tensor, k, largest=True)[0]
    min_topk = torch.min(topk).numpy()
    return min_topk

def top_k_for_tensor(tensor, val):
    tensor = tensor.numpy()
    for i in range(tensor.shape[0]):
        if tensor[i] < val:
            tensor[i] = 0.0
    return torch.from_numpy(tensor)

def topk_sparsification_torch_load(load, percentage):
    k = int(0.01*percentage*568128)
    list_of_tensors = []
    for i in iter(load):
        list_of_tensors.append(load[i])
    list_of_resized_tensors = [t.reshape(-1) for t in list_of_tensors]
    min_topk = find_min_topk(list_of_resized_tensors, k)
    spars_t = []
    for t in list_of_resized_tensors:
        spars_t.append(top_k_for_tensor(t, min_topk))
    # inserisci nel dizionario
    j = 0
    for i in iter(load):
        load[i] = spars_t[j].view(load[i].shape)
        j += 1
    return load

def imshow(img):
    min_x = torch.min(img)
    max_x = torch.max(img)
    img = (img-min_x)/(max_x-min_x) # no, normalizzali! DA MODIFICARE ANCHE SE POCO IMPORTANTE
    torch.clamp(img, 0, 1)
    img = img.detach()
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()