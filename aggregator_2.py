from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy import array
from numpy import argmax
from numpy import swapaxes
from random import randint

class aggregator():
    
    def __init__(self, number_of_clusters):
        image = Input(shape=(28, 28), name="input_image")
        flatten_image = Flatten()(image)
        y = Dense(100, activation='tanh')(flatten_image)
        y = Dense(100, activation='tanh')(y)
        y = Dense(100, activation='tanh')(y)
        y = Dense(100, activation='tanh')(y)
        y = Dense(number_of_clusters, activation='softmax')(y)
        model = Model(inputs=image, outputs=y, name='aggregator')
        opt = Adam(learning_rate = 0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
        self.model = model
    
    def evaluation(self, fed_scenario, x_test, y_test):
        predictions = []
        for cluster in fed_scenario.list_of_clusters:
            predictions.append(cluster.get_model().predict(x_test))
        predictions = array(predictions)
        predictions = swapaxes(predictions, 0, 1)
    
        server_weights = self.model.predict(x_test, verbose=0)
        acc = 0
        weighted_avg = [sum([server_weights[j][i]*predictions[j][i] for i in range(len(fed_scenario.list_of_clusters))]) for j in range(len(x_test))]
        for i in range(len(weighted_avg)):
            if argmax(weighted_avg[i]) == argmax(y_test[i]):
                acc += 1
        return acc / len(y_test)
    
    def local_evaluation(self, fed_scenario):
        acc = 0
        for cluster in fed_scenario.list_of_clusters:
            acc += self.evaluation(fed_scenario, cluster.test_data['images'], to_categorical(cluster.test_data['labels'], 10))
        return acc / len(fed_scenario.list_of_clusters)
    
    def custom_y(self, fed_scenario, train=True):
        if train == True:
            server_x = fed_scenario.server.x_train
            server_y = fed_scenario.server.y_train
        else:
            server_x = fed_scenario.server.x_test
            server_y = fed_scenario.server.y_test
        predictions = []
        for cluster in fed_scenario.list_of_clusters:
            predictions.append(cluster.get_model().predict(server_x))
        predictions = array(predictions)
        predictions = swapaxes(predictions, 0, 1)
        custom_y = []
        for i in range(len(server_x)):
            for j in range(len(predictions[i])):
                if argmax(predictions[i][j]) == argmax(server_y[i]):
                    custom_y.append(to_categorical(j, len(fed_scenario.list_of_clusters)))
                    break
            if not len(custom_y) == i+1: # FIX THIS CHOICE!
                custom_y.append(to_categorical(randint(0, len(fed_scenario.list_of_clusters)-1), len(fed_scenario.list_of_clusters)))
        return array(custom_y)