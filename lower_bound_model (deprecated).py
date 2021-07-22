"""
Created on Wed Oct 14 23:43:24 2020

@author: Nicola Gulmini
@mail: nicolagulmini@gmail.com or nicola.gulmini@studenti.unipd.it
"""
# tensorflow 2.0 and keras 2.3 
# programma per il modello centralizzato federato: server - users 
# stimato lower bound per le prestazioni degli altri


from utils import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # for warnings...?

number_of_users = 20
number_of_clusters = 4
bias_factor = 0.2 # in interval (0,1)

epochs = 64
batch = 60

from utils import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # warnings

number_of_users = 20
number_of_clusters = 4
bias_factor = 0.2 # in interval (0,1)

epochs = 64
batch = 60

users_ids = list(range(number_of_users))
print('Users:', users_ids)
X_train, Y_train, X_test, Y_test = load_preprocessed_cifar10_ds()
division_train = create_dictionary_from_dataset(tf.data.Dataset.from_tensor_slices((X_train, Y_train)))
division_test = create_dictionary_from_dataset(tf.data.Dataset.from_tensor_slices((X_test, Y_test)))

clusters = assign_users_to_clusters_randomly(list(range(number_of_users)), number_of_clusters)
fav_classes = [np.random.randint(0, 10) for _ in range(number_of_clusters)] # to obtain the same heterogeneity for train and test datasets

clusters_train_datasets = ds_division(X_train, division_train, bias_factor, number_of_clusters, fav_classes)
clusters_test_datasets = ds_division(X_test, division_test, bias_factor, number_of_clusters, fav_classes)

# distribute the data among clusters
for i in range(len(clusters_train_datasets)):
    clusters[i].set_train_data(clusters_train_datasets[i])
    clusters[i].set_test_data(clusters_test_datasets[i])
  
for cluster in clusters:
    cluster_ds = cluster.train_data
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


percentage = 30

# the server has got a model, randomly initalized, of the same 'shape' of the users' single models
server = define_model()
server.model.set_weights(top_k_sparsificate_model_weights_tf(server.model.get_weights(), percentage/100))
# sbagliato sparsificare all'inizio ma lo faccio perche 
# e' come se gli utenti avessero fatto una iterazione a vuoto e uppata al server, sparsificata. 
# Alla fine a livello server devono starci dei pesi sparsificati.

def transfer_from_server_to_users(server, clusters):
    # trasferisco il modello dal server agli utenti
    w = server.model.get_weights()
    for cluster in clusters:
        for user in cluster.users:
            tmp_user_model = define_model()
            tmp_user_model.model.set_weights(w)
            user.set_model(tmp_user_model.model)
            del tmp_user_model

transfer_from_server_to_users(server, clusters)
already_done = False
iterations = 7
if not already_done:
    for _ in range(iterations):
        print('')
        print("************ Iteration " + str(_) + " ************")
        print('')
        server.model.save("LB_server_model_sparse"+str(percentage)+"_iter"+str(_)+".h5")
        transfer_from_server_to_users(server, clusters)
        for cluster in clusters:
            for user in cluster.users:
                train_user_model(user, cluster, epochs, batch)
                user.get_model().save("LB_user"+str(user.name)+"_sparse"+str(percentage)+"_"+str(_)+".h5")
        print('Start aggregating server...')
        # aggregate 
        # non metto i pesi perché tutti gli utenti hanno lo stesso ammontare di dati
        w = server.model.get_weights()
        sum_terms = []
        for cluster in clusters:
            for user in cluster.users:
                wu = user.get_model().get_weights()
                sum_terms.append([1/20*np.subtract(wu[i], w[i]) for i in range(len(wu))])
        update = sum_terms[0]
        for i in range(1, len(sum_terms)):
            tmp = sum_terms[i]
            update = [np.add(tmp[j], update[j]) for j in range(len(update))]
        new_weights = [np.add(w[i], update[i]) for i in range(len(w))]
        server.model.set_weights(top_k_sparsificate_model_weights_tf(new_weights, percentage/100))
        print('*** Updated server model ***')
        print('')

to_plot = []
for i in range(iterations):
    tmp_model = tf.keras.models.load_model("./centralized"+str(percentage)+"/LB_server_model_sparse"+str(percentage)+"_iter"+str(i)+".h5")
    _, accuracy = tmp_model.evaluate(X_test, Y_test)
    print('accuracy =', accuracy)
    to_plot.append(accuracy)
plt.plot(range(iterations), to_plot)
plt.xlabel('iterations')
plt.ylabel('server\' accuracy')
name = "centralized_" + str(percentage) + " sparse.png"
plt.savefig(name)
