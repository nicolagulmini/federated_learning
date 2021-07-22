class cluster:
    def __init__(self, number):
        self.users = []
        self.number = number
    def number_of_users(self):
        return len(self.users)
    def add_user(self, user):
        self.users.append(user)
    def set_train_data(self, train_data):
        self.train_data = train_data
    def set_test_data(self, test_data):
        self.test_data = test_data
    def set_model(self, model):
        self.model = model
    def set_estimation(self, estimation):
        self.estimation = estimation
    def get_model(self):
        return self.model
    def get_estimation(self):
        return self.estimation
    def print_information(self):
        print("Cluster number " + str(self.number) + ". User ids: " + str([user.name for user in self.users]))
        
class user_information:
    def __init__(self, name, cluster):
        self.name = name
        self.cluster = cluster
    def set_data(self, data):
        self.data = data
    def set_model(self, model):
        self.model = model
    def get_model(self):
        return self.model
    def set_estimation(self, estimation):
        self.estimation = estimation
    def get_estimation(self):
        return self.estimation
        
    
class federated_setup:
    
    def assign_users_to_clusters(number_of_users, number_of_clusters):
        # assigns uniformly and in a sorted way, the given users to the given clusters. Returns a list of clusters. This step has to be done before the data assignment.
        if not number_of_users % number_of_clusters == 0:
            print("It is better to have the same number of users for each cluster, to make the computation easier. This issue will be solved.")
            return
        users_per_cluster = int(number_of_users/number_of_clusters)
        clusters_list = []
        user_id = 0
        for _ in range(number_of_clusters):
            tmp_cluster = cluster(_)
            for __ in range(users_per_cluster):
                tmp_user = user_information(user_id, tmp_cluster)
                tmp_cluster.add_user(tmp_user)
                user_id += 1
            clusters_list.append(tmp_cluster)
        # to recap
        for c in clusters_list:
            c.print_information()
        return clusters_list