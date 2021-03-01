"""
Created on Wed Oct 14 23:43:24 2020

@author: Nicola Gulmini
@mail: nicolagulmini@gmail.com or nicola.gulmini@studenti.unipd.it
"""
# tensorflow 2.0 and keras 2.3 

# attenzione ai cluster!!! A VOLTE CAMBIANO MA E' NECESSARIO CHE SIANO SEMPRE GLI STESSI CON LA STESSA DISTRIBUZIONE


from utils import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

number_of_users = 20
number_of_clusters = 4
bias_factor = 0.4 # in interval (0,1)

epochs = 60
estimation_epochs = 120
batch = 64
estimation_batch = 128
iterations = 7

already_done_CNN = False
already_done_estimation = True
trained_models = False
test_aggregation_among_clusters = False

classification_percentage = 100
estimation_percentage = 50

users_ids = list(range(number_of_users))
print('Users:', users_ids)

X_train, Y_train, X_test, Y_test = load_preprocessed_cifar10_ds()

division_train = create_dictionary_from_dataset(tf.data.Dataset.from_tensor_slices((X_train, Y_train)))
division_test = create_dictionary_from_dataset(tf.data.Dataset.from_tensor_slices((X_test, Y_test)))

'''
# plot a random image for each class, for train data
for i in range(10):
    image_list = division_train[i]
    random_image = image_list[np.random.randint(0, len(image_list))]
    plt.subplot(2, 5, i+1)
    plt.title(classes.get(i))
    plt.imshow(random_image.numpy())
    plt.axis('off')
'''

clusters = assign_users_to_clusters_randomly(list(range(number_of_users)), number_of_clusters)
fav_classes = [np.random.randint(0, 10) for _ in range(number_of_clusters)] # to obtain the same heterogeneity for train and test datasets

clusters_train_datasets = ds_division(X_train, division_train, bias_factor, number_of_clusters, fav_classes)
clusters_test_datasets = ds_division(X_test, division_test, bias_factor, number_of_clusters, fav_classes)

'''
# plot train dataset histograms for clusters
f = plt.figure(figsize=(15, 12))
i = 0 # to plot
for cluster in range(number_of_clusters):
    cluster_ds = clusters_train_datasets[cluster]
    plot_data = collections.defaultdict(list)
    for el in cluster_ds['labels']:
        label = int(np.argmax(el))
        plot_data[label].append(label)
    plt.subplot(4, 5, i+1)
    i += 1
    plt.title('cluster ' + str(cluster))
    for j in range(10):
        plt.hist(plot_data[j], bins=range(11))

# plot test dataset histograms for clusters
f = plt.figure(figsize=(15, 12))
i = 0 # to plot
for cluster in range(number_of_clusters):
    cluster_ds = clusters_test_datasets[cluster]
    plot_data = collections.defaultdict(list)
    for el in cluster_ds['labels']:
        label = int(np.argmax(el))
        plot_data[label].append(label)
    plt.subplot(4, 5, i+1)
    i += 1
    plt.title('cluster ' + str(cluster))
    for j in range(10):
        plt.hist(plot_data[j], bins=range(11))
'''

# distribute the data among clusters
for i in range(len(clusters_train_datasets)):
    clusters[i].set_train_data(clusters_train_datasets[i])
    clusters[i].set_test_data(clusters_test_datasets[i])
  
# divide the cluster dataset amoung users of the cluster, in a naive way...
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

# the server has got a model, randomly initalized, of the same 'shape' of the users' single models
server_model = define_model()

# in initialize models solo CNN ha pesi sparsificati, ma tanto non cambia nulla alla iterazione 0
initialize_models(clusters, server_model, classification_percentage/100) 

'''
# to test if there is a bug in the first part of the code
print("\nTest:")
for cluster in clusters:
    X_test = cluster.test_data['images']
    Y_test = cluster.test_data['labels']
    for user in cluster.users:
        user_model = user.get_model()
        accuracy = user_model.evaluate(X_test, Y_test)[1]
        print("acc of user " + str(user.name) + ": " + str(accuracy))
'''
# per ripristinare da dove si è interrotta la computazione delle CNN ************************************

for cluster in clusters:
    name = "cluster"+str(cluster.number)+"_"+str(classification_percentage)+"sparse_iter"+str(2)+".h5"
    tmp_model = tf.keras.models.load_model(name)
    cluster.model.set_weights(tmp_model.get_weights())


cluster_iterations = iterations
# training of the clusters' CNN

if not already_done_CNN:
    for _ in range(3, cluster_iterations):
        print('')
        print("************ Iteration " + str(_) + " ************")
        print("Sparsification = " + str(classification_percentage) + "%")
        print('')
        for cluster in clusters:
            cluster.model.save("cluster"+str(cluster.number)+"_"+str(classification_percentage)+"sparse_iter"+str(_)+".h5")
            transfer_cluster_model_to_users(cluster) # useless at first iteration but fundamental then # not sparsified!!!! Only in the uplink
            for user in cluster.users:
                train_user_model(user, cluster, epochs, batch)
                user.get_model().save("user"+str(user.name)+"_"+str(classification_percentage)+"sparse_iter"+str(_)+".h5")
            print('Start aggregating cluster', cluster.number, 'parameters...')
            # aggregate
            cluster_trainset_size = len(cluster.train_data['images'])
            wc = cluster.model.get_weights()
            sum_terms = []
            for user in cluster.users:
                wu = user.get_model().get_weights()
                nu = len(user.data['labels'])
                frac = nu/cluster_trainset_size
                sum_terms.append([frac*np.subtract(wu[i], wc[i]) for i in range(len(wu))])
            update = sum_terms[0]
            for i in range(1, len(sum_terms)): # could do better...
                tmp = sum_terms[i]
                update = [np.add(tmp[j], update[j]) for j in range(len(update))]
            new_cluster_weights = [np.add(wc[i], update[i]) for i in range(len(wc))]
            sparse_weights = top_k_sparsificate_model_weights_tf(new_cluster_weights, classification_percentage/100)
            cluster.model.set_weights(sparse_weights)
            print('Updated model of cluster', cluster.number)
    already_done_CNN = True
    trained_models = True

if already_done_CNN and trained_models:    
    plot_cluster_performances(clusters, cluster_iterations, classification_percentage)
    
    

# estimation *** SEPARATA PER NON FARE CONFUSIONE, MA E' DEL TUTTO EQUIVALENTE
est_iterations = iterations
if not already_done_estimation:
    for _ in range(est_iterations):
        print('')
        print("************ Iteration " + str(_) + " ************")
        print('')
        for cluster in clusters:
            torch.save(cluster.estimation.state_dict(), "estimation_cluster"+str(cluster.number)+"sparse_"+str(estimation_percentage)+"_iter"+str(_)+".pt")
            transfer_cluster_estimation_to_users(cluster)
            for user in cluster.users:
                # alleno il modello su X_train cosi si possono fare esperimenti con X_test
                dataloader = torch.utils.data.DataLoader(dataset=user.data['images'], batch_size=estimation_batch, shuffle=True, pin_memory=True)
                train_estimation(user.get_estimation(), estimation_epochs, dataloader)
                torch.save(user.get_estimation().state_dict(), "estimation_user"+str(user.name)+"sparse"+str(estimation_percentage)+"_iter"+str(_)+".pt")
            print('Start aggregating cluster', cluster.number, 'estimation parameters...')
            # aggregate
            cluster_trainset_size = len(cluster.train_data['images'])
            sum_terms = []
            for user in cluster.users:
                pu = user.get_estimation().state_dict()
                nu = len(user.data['labels'])
                frac = nu/cluster_trainset_size # perche anche la stima e' fatta su X_train
                for i in iter(pu):
                    pu[i] = pu[i]*frac # qui modifico i pesi che pero' non saranno caricati nel modello (e se anche fosse non avrebbe importanza perche poi lo sostituisco)
                sum_terms.append(pu)
            # uso il primo termine in sum_terms per sommarci tutti gli altri
            ref = sum_terms[0]
            for i in iter(ref):
                for j in sum_terms[1:]:
                    ref[i] += j[i]
            cluster.estimation.load_state_dict(topk_sparsification_torch_load(ref, estimation_percentage))
            print('Setted estimation of cluster', cluster.number)


if test_aggregation_among_clusters:
    print('')
    print("Test the " + str(classification_percentage) + "-sparse classification + " + str(estimation_percentage) + "-sparse estimation model")
    accuracies = []
    for _ in range(6, iterations):
        print('')
        print("************ Iteration " + str(_) + " ************")
        print('')
        s = test_aggregation(X_test, Y_test, clusters, classification_percentage, estimation_percentage, _, _)
        print('accuracy:', s)
        accuracies.append(s)
    plt.plot(range(iterations), accuracies)
    plt.xlabel('iterations')
    plt.ylabel('local acc')
    name = str(classification_percentage) + "-" + str(estimation_percentage) + "_model.png" 
    plt.savefig(name)

    

# PROVA PER X-0 model!!

true_predictions = 0
for t in range(len(X_test)):
    test_img = X_test[t]
    test_label = Y_test[t]
    true = np.argmax(test_label)
    prediction_vectors = []
    for cluster in clusters:
        tmp_model = tf.keras.models.load_model("./classification"+str(100)+"/cluster"+str(cluster.number)+"_"+str(100)+"sparse_iter"+str(6)+".h5")
        pred = tmp_model.predict(test_img.reshape((1, 32, 32, 3)))
        prediction_vectors.append(pred.reshape(-1))
    #print(prediction_vectors)
    rule = sum(prediction_vectors)
    predicted = np.argmax(rule)
    #print(predicted)
    if predicted==true:
        true_predictions +=1
print(true_predictions/len(X_test)*100) 

