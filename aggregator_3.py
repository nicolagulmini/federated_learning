from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
# initializers 
from tensorflow.keras.initializers import Ones
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.initializers import Identity
from tensorflow.keras.initializers import RandomNormal


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy import array
from numpy import argmax
from numpy import swapaxes
from random import randint


class attention_based_aggregator():
    
    def __init__(self, number_of_clusters):
        image = Input(shape=(28, 28), name="input_image") 
        cluster_outputs = Input(shape=(number_of_clusters, 10), name='softmax_outputs')
        flatten_image = Flatten()(image)
        weights = Dense(number_of_clusters, activation='linear', bias_initializer=Ones(), kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(flatten_image) 
        out = Dot(axes=1)([weights, cluster_outputs])
        out = Dense(10, activation='softmax', kernel_initializer=Identity(), bias_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))(out)
        model = Model(inputs=[image, cluster_outputs], outputs=out, name='attention_based_aggregator')
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics='accuracy')
        self.model = model
        
    def produce_datasets(self, fed_setup):
        server_x_train, server_y_train, server_x_val, server_y_val = fed_setup.train_validation_split(fed_setup.server.x_train, fed_setup.server.y_train)
        outputs = swapaxes(array([cluster.get_model().predict(server_x_train) for cluster in fed_setup.list_of_clusters]), 0, 1)
        val_outputs = swapaxes(array([cluster.get_model().predict(server_x_val) for cluster in fed_setup.list_of_clusters]), 0, 1)
        test_outputs = swapaxes(array([cluster.get_model().predict(fed_setup.server.x_test) for cluster in fed_setup.list_of_clusters]), 0, 1)
        return ([server_x_train, outputs], server_y_train), ([server_x_val, val_outputs], server_y_val), ([fed_setup.server.x_test, test_outputs], fed_setup.server.y_test) 
    
    def ordered_dataset(self, fed_setup, favourite_label):
        x, y = fed_setup.server.x_train, fed_setup.server.y_train
        new_x = []
        new_y = []
        for i in range(len(x)):
            if argmax(y[i]) == favourite_label:
                new_x.append(x[i])
                new_y.append(y[i])
        server_x_train, server_y_train, server_x_val, server_y_val = fed_setup.train_validation_split(array(new_x), array(new_y))
        outputs = swapaxes(array([cluster.get_model().predict(server_x_train) for cluster in fed_setup.list_of_clusters]), 0, 1)
        val_outputs = swapaxes(array([cluster.get_model().predict(server_x_val) for cluster in fed_setup.list_of_clusters]), 0, 1)
        return ([server_x_train, outputs], server_y_train), ([server_x_val, val_outputs], server_y_val)
        
    
    def train(self, x_train, y_train, x_val, y_val, verbose, epochs):
        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=32, 
            epochs=epochs, 
            verbose=verbose, 
            validation_data=(x_val, y_val),
            shuffle=True    
            )
        return history.history['accuracy'], history.history['loss'], history.history['val_accuracy'], history.history['val_loss']
        
    def evaluate(self, x_test, y_test, verbose):
        return self.model.evaluate(x_test, y_test, verbose=verbose)[1]
