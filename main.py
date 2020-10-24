"""
Created on Wed Oct 14 23:43:24 2020

@author: Nicola Gulmini
@mail: nicolagulmini@gmail.com or nicola.gulmini@studenti.unipd.it
"""
# tensorflow 2.0 and keras 2.3 
# lavora anche con l'altro ambiente pare... quindi posso prendere i codici gia pronti!

# FUTURE UPDATES:
# a more sophisticated train-validation split
# save in the clusters the different accuracies and models for each iteration...
# implement some if statements with load_model instead of traning every time!!
# aggrega a livello di server, con k sparsification, variando k_model e k_estimation e trovando il migliore
# at the end, try to change the bias factor and repeat the entire analysis
# attenzione ai cluster!!! A VOLTE CAMBIANO MA E' NECESSARIO CHE SIANO SEMPRE GLI STESSI CON LA STESSA DISTRIBUZIONE


from utils_tf import *
from utils_torch import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # mi da dei warning fastidiosi

number_of_users = 20
number_of_clusters = 4
bias_factor = 0.2 # in interval (0,1)

epochs = 64
estimation_epochs = 1000
batch = 60
estimation_batch = 128

already_done_CNN = True
already_done_estimation = True

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
'''
'''
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
initialize_models(clusters, server_model)

cluster_iterations = 7 # a good number

# training of the clusters' CNN
if not already_done_CNN:
    for _ in range(cluster_iterations):
        print('')
        print("************ Iteration " + str(_) + " ************")
        print('')
        for cluster in clusters:
            cluster.model.save("cluster"+str(cluster.number)+"_iter"+str(_)+".h5")
            transfer_cluster_model_to_users(cluster) # useless at first iteration but fundamental then
            for user in cluster.users:
                train_user_model(user, cluster, epochs, batch)
                user.get_model().save("user"+str(user.name)+"_iter"+str(_)+".h5")
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
            for i in range(1, len(sum_terms)): # avrei una soluzione migliore che usa np.reshape()...
                tmp = sum_terms[i]
                update = [np.add(tmp[j], update[j]) for j in range(len(update))]
            cluster.model.set_weights([np.add(wc[i], update[i]) for i in range(len(wc))])
            print('Updated model of cluster', cluster.number)
            
#if already_done_CNN:
    #plot_cluster_performances(clusters, iterations)
    
    
# ora carica gli ultimi modelli salvati per poterli aggregare a livello di cluster


# estimation *** SEPARATA PER NON FARE CONFUSIONE, MA E' DEL TUTTO EQUIVALENTE
est_iterations = 3
estimation_batch = 10
estimation_epochs = 3
if not already_done_estimation:
    for _ in range(est_iterations):
        print('')
        print("************ Iteration " + str(_) + " ************")
        print('')
        for cluster in clusters:
            torch.save(cluster.estimation.state_dict(), "estimation_cluster"+str(cluster.number)+"_iter"+str(_)+".pt")
            transfer_cluster_estimation_to_users(cluster)
            for user in cluster.users:
                # alleno il modello su X_train cosi si possono fare esperimenti con X_test
                dataloader = torch.utils.data.DataLoader(dataset=user.data['images'], batch_size=estimation_batch, shuffle=True, pin_memory=True)
                train_estimation(user.get_estimation(), estimation_epochs, dataloader)
                torch.save(user.get_estimation().state_dict(), "estimation_user"+str(user.name)+"_iter"+str(_)+".pt")
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
            cluster.estimation.load_state_dict(ref)
            print('Setted estimation of cluster', cluster.number)
            
#if already_done_estimation:
    #caricale

# per ora ho le stime solo a iter2, dovrei aumentarle, a meno che non abbia deciso di fermarmi perche l'accuracy non stava aumentando...

true_predictions = 0
for t in range(len(X_test)):
    iterazione_considerata = cluster_iterations-6 # considero questi modelli per ora
    test_img = X_test[t]
    test_label = Y_test[t]
    true = np.argmax(test_label)
    #print('true:', classes[true])
    prediction_vectors = []
    log_probs = []
    for cluster in clusters:
        tmp_model = tf.keras.models.load_model("cluster"+str(cluster.number)+"_iter"+str(iterazione_considerata)+".h5")
        tmp_est = NICE(data_dim=3072, num_coupling_layers=4)
        tmp_est.load_state_dict(torch.load("estimation_cluster"+str(cluster.number)+"_iter"+str(iterazione_considerata)+".pt"))
        log_prob= tmp_est.forward(torch.from_numpy(test_img.reshape((3072))))[1]-tmp_est.f(torch.from_numpy(test_img.reshape((3072))))[1]
        log_probs.append(log_prob.detach().numpy().reshape((-1)))
        pred = tmp_model.predict(test_img.reshape((1, 32, 32, 3)))
        prediction_vectors.append(pred.reshape(-1))
      
    a = float(max(log_probs))
    log_meno_a = [float(el-a) for el in log_probs]
    sum_exp = 0
    for el in log_meno_a:
        sum_exp += np.exp(el)
    comp = a+np.log(sum_exp)
    alpha = [np.exp(el-comp) for el in log_probs]
    
    final_vector = [alpha[i]*prediction_vectors[i] for i in range(len(clusters))]
    rule = final_vector[0]
    
    for i in range(1, len(final_vector)):
        rule += final_vector[i]
    predicted = np.argmax(rule)
    #print('predicted:', classes[predicted])
    if predicted == true:
        true_predictions += 1
print('fraction of true predictions', true_predictions/len(X_test))

# trasforma in metodo
# implementa la topk
# chiedi della topk
# ripeti in funzione dell'eterogeneita
# 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

