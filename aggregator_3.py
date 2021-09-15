from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dot
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Ones
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
    ''' Aggregator 3 colab section
    from keras.models import Model
    # layers[1] is the intermediate dense layer
    intermediate_models = [Model(inputs=fed_setup.list_of_clusters[i].get_model().input,
                                     outputs=fed_setup.list_of_clusters[i].get_model().layers[1].output) for i in range(len(fed_setup.list_of_clusters))]

    # server aggregator perf only at the end
    server_agg = aggregator(len(fed_setup.list_of_clusters)) # <class 'federated_learning.aggregator_3.aggregator'>
    server_epochs = 30
    server_x_train, server_y_train, server_x_val, server_y_val = fed_setup.train_validation_split(fed_setup.server.x_train, fed_setup.server.y_train)
    outputs = [model.predict(server_x_train) for model in intermediate_models]
    val_outputs = [model.predict(server_x_val) for model in intermediate_models]
    test_outputs = [model.predict(fed_setup.server.x_test) for model in intermediate_models]

    history = server_agg.model.fit(
        x=[server_x_train, array([array(outputs)]).reshape((len(server_x_train), 9, 10))],
        y=server_y_train,
        batch_size=32, 
        epochs=server_epochs, 
        verbose=0, 
        validation_data=([server_x_val, array([array(val_outputs)]).reshape((len(server_x_val), 9, 10))], server_y_val),
        shuffle=True    
        )

    # measure the global performance
    server_aggregator_performance = server_agg.model.evaluate([fed_setup.server.x_test, array([array(test_outputs)]).reshape((len(fed_setup.server.x_test), 9, 10))], fed_setup.server.y_test, verbose=0)[1]
    '''

class attention_based_aggregator():
    
    def __init__(self, number_of_clusters):
        image = Input(shape=(28, 28), name="input_image") 
        cluster_outputs = Input(shape=(number_of_clusters, 10), name='softmax_outputs')
        flatten_image = Flatten()(image)
        weights = Dense(number_of_clusters, activation='softmax', kernel_initializer=Ones())(flatten_image) # sigmoid or tanh or softmax
        summ = Dot(axes=1)([cluster_outputs, weights])
        
        model = Model(inputs=[image, cluster_outputs], outputs=summ, name='attention_based_aggregator')
        opt = Adam(learning_rate = 0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        self.model = model
        
    def produce_datasets(self, fed_setup):
        server_x_train, server_y_train, server_x_val, server_y_val = fed_setup.train_validation_split(fed_setup.server.x_train, fed_setup.server.y_train)
        outputs = swapaxes(array([cluster.get_model().predict(server_x_train) for cluster in fed_setup.list_of_clusters]), 0, 1)
        val_outputs = swapaxes(array([cluster.get_model().predict(server_x_val) for cluster in fed_setup.list_of_clusters]), 0, 1)
        test_outputs = swapaxes(array([cluster.get_model().predict(fed_setup.server.x_test) for cluster in fed_setup.list_of_clusters]), 0, 1)
        return ([server_x_train, outputs], server_y_train), ([server_x_val, val_outputs], server_y_val), ([fed_setup.server.x_test, test_outputs], fed_setup.server.y_test) 
    
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
