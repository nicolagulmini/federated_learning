from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy import array
from numpy import argmax
from numpy import swapaxes
from random import randint

class aggregator():
# consider also the local models weights
    
    def __init__(self, number_of_clusters):
        image = Input(shape=(28, 28), name="input_image") 
        clusters_weights = Input(shape=(number_of_clusters, 784, 10), name='input_cluster_model_weights')
        
        flatten_image = Flatten()(image)
        dense_flatten_img = Dense(30, activation='relu')(flatten_image)
        flatten_clusters_weights = Flatten()(clusters_weights)
        dense_flatten_clus_w = Dense(10*number_of_clusters, activation='relu')(flatten_clusters_weights)
        concat = Concatenate()([dense_flatten_img, dense_flatten_clus_w])
        # other dense layers?
        y = Dense(100, activation='relu')(concat)
        y = Dense(10, activation='softmax')(y)

        model = Model(inputs=[image, clusters_weights], outputs=y, name='aggregator_3')
        opt = Adam(learning_rate = 0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        self.model = model

    def evaluation(self, fed_scenario, x_test, y_test):
        return 0
    
    def local_evaluation(self, fed_scenario):
        return 0
    
    def custom_x(self):
        return 0