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
        clusters_weights = Input(shape=(number_of_clusters, 10), name='input_cluster_model_interm_outputs')
        
        flatten_image = Flatten()(image)
        flatten_clusters_weights = Flatten()(clusters_weights)
        concat = Concatenate()([flatten_image, flatten_clusters_weights])
        y = Dense(10, activation='softmax')(concat)

        model = Model(inputs=[image, clusters_weights], outputs=y, name='aggregator_3')
        opt = Adam(learning_rate = 0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        self.model = model

class attention_based_aggregator():
    
    def __init__(self, number_of_clusters):
        image = Input(shape=(28, 28), name="input_image") 
        cluster_outputs = Input(shape=(number_of_clusters, 10), name='softmax_outputs')
        
        flatten_image = Flatten()(image)
        #flatten_cluster_outputs = Flatten()(cluster_outputs)
        weights = Dense(number_of_clusters, activation='sigmoid')(flatten_image) # or tanh or softmax
        #concat = Concatenate()([flatten_image, flatten_clusters_weights])
        # take the sum
        #y = Dense(10, activation='softmax')(concat)

        model = Model(inputs=[image, cluster_outputs], outputs=y, name='attention_based_aggregator')
        opt = Adam(learning_rate = 0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        self.model = model