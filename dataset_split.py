import tensorflow
import random
import matplotlib.pyplot as plt
import pandas
import os

number_of_clusters = 9
bias = 0.5
number_of_images_per_dataset = 400

def save_federated_mnist(number_of_clusters, bias, number_of_images_per_dataset):
    
    (X_train, Y_train), (X_test, Y_test) = tensorflow.keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # single digit datasets
    single_digit_training_sets = [[] for _ in range(10)]
    for i in range(len(X_train)):
        single_digit_training_sets[Y_train[i]].append(X_train[i])
        
    single_digit_test_sets = [[] for _ in range(10)]
    for i in range(len(X_test)):
        single_digit_test_sets[Y_test[i]].append(X_test[i])
    
    most_frequent_class_for_client = [random.randint(0, 9) for _ in range(number_of_clusters)]
    client_x_train, client_y_train, client_x_test, client_y_test = [[[] for _ in range(number_of_clusters)] for i in range(4)]
    
    for c in range(number_of_clusters):
        for _ in range(int(number_of_images_per_dataset*bias)):
            client_x_train[c].append(random.choice(single_digit_training_sets[most_frequent_class_for_client[c]]))
            client_x_test[c].append(random.choice(single_digit_test_sets[most_frequent_class_for_client[c]]))
            client_y_train[c].append(most_frequent_class_for_client[c])
            client_y_test[c].append(most_frequent_class_for_client[c])
        for _ in range(int(number_of_images_per_dataset*(1-bias))):
            random_index = random.randint(0, len(X_train)-1)
            client_x_train[c].append(X_train[random_index])
            client_y_train[c].append(Y_train[random_index])
            random_index = random.randint(0, len(X_test)-1)
            client_x_test[c].append(X_test[random_index])
            client_y_test[c].append(Y_test[random_index])
        
    os.makedirs('./dfataset')
    root_path = "./dfataset"
    for c in range(number_of_clusters):
        client_path = root_path + "/" + str(c)
        os.makedirs(client_path)
        os.makedirs(client_path + "/training_images")
        os.makedirs(client_path + "/test_images")
        for i in range(len(client_x_train[c])):
            plt.imsave(client_path + "/training_images/" + str(i) + ".png", client_x_train[c][i], cmap='gray')
        for i in range(len(client_x_test[c])):
            plt.imsave(client_path + "/test_images/" + str(i) + ".png", client_x_test[c][i], cmap='gray')
        
        df = pandas.DataFrame({'name': [i for i in range(len(client_x_train[c]))], 'label': client_y_train[c]})
        df.to_csv(client_path + "/training_labels.csv", header=False, index=False)
        df = pandas.DataFrame({'name': [i for i in range(len(client_x_test[c]))], 'label': client_y_test[c]})
        df.to_csv(client_path + "/test_labels.csv", header=False, index=False)

save_federated_mnist(number_of_clusters, bias, number_of_images_per_dataset)

        

    

