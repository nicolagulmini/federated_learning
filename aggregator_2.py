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
        y = Dense(100, activation='relu')(flatten_image)
        y = Dense(100, activation='relu')(y)
        y = Dense(100, activation='relu')(y)
        y = Dense(100, activation='relu')(y)
        y = Dense(100, activation='relu')(y)
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
    
    def custom_y(self, fed_scenario, x_dataset, y_dataset):
        predictions = []
        for cluster in fed_scenario.list_of_clusters:
            predictions.append(cluster.get_model().predict(x_dataset))
        predictions = array(predictions)
        predictions = swapaxes(predictions, 0, 1)
        custom_y = []
        for i in range(len(x_dataset)):
            conf = 0
            max_conf = 0
            right_cand = -1
            nearest_cand = 0
            right_pred = argmax(y_dataset[i])
            for j in range(len(predictions[i])):
                if argmax(predictions[i][j]) == right_pred:
                    if predictions[i][j][right_pred] > conf:
                        conf = predictions[i][j][right_pred]
                        right_cand = j
                if predictions[i][j][right_pred] > max_conf:
                    max_conf = predictions[i][j][right_pred]
                    nearest_cand = j
            if not right_cand == -1: # chosen candidates
                custom_y.append(to_categorical(right_cand, len(fed_scenario.list_of_clusters)))
            else: # there is no correct cluster prediction
                custom_y.append(to_categorical(nearest_cand, len(fed_scenario.list_of_clusters)))
        return array(custom_y)
