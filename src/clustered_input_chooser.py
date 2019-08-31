
from src.input_chooser import InputChooser
from sklearn.cluster import KMeans

import numpy as np


class ClusteredInputChooser:
    def __init__(self, test_inputs, test_outputs, input_transformer=lambda i: i, n_clusters=20):
        self.test_inputs = test_inputs.copy()
        self.test_outputs = test_outputs.copy()
        self.input_transformer = input_transformer
        self.n_clusters = n_clusters
        transormed_test_inputs = self.input_transformer(test_inputs.copy())
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        transormed_test_inputs = transormed_test_inputs.reshape(len(transormed_test_inputs), -1)
        self.clusters_indexes = self.kmeans.fit_predict(transormed_test_inputs)
        self.cluster_inputs = [[] for i in range(self.n_clusters)]
        self.cluster_ouputs = [[] for i in range(self.n_clusters)]
        for i in range(len(self.test_inputs)):
            cluster_index = self.clusters_indexes[i]
            test_input = self.test_inputs[i]
            test_output = self.test_outputs[i]
            self.cluster_inputs[cluster_index].append(test_input)
            self.cluster_ouputs[cluster_index].append(test_output)

        self.cluster_input_choosers = []
        for i in range(self.n_clusters):
            input_chooser = InputChooser(np.array(self.cluster_inputs[i]), np.array(self.cluster_ouputs[i]))
            self.cluster_input_choosers.append(input_chooser)


        self.cluster_weights = np.ones(self.n_clusters)

    def __len__(self):
        return self.n_clusters

    def sample(self, batch_size, cluster_index=None):
        if cluster_index == None:
            cluster_index = np.random.choice(self.n_clusters, p=self.cluster_weights/np.sum(self.cluster_weights))
        return cluster_index, self.cluster_input_choosers[cluster_index].sample(batch_size)

    def __call__(self, batch_size=1, cluster_index=None):
        return self.sample(batch_size, cluster_index=cluster_index)

    def append(self, new_inputs, new_outputs):
        new_inputs_transformed = self.input_transformer(new_inputs)
        new_inputs_transformed = new_inputs_transformed.reshape(len(new_inputs), -1)
        new_cluster_indexes = self.kmeans.predict(new_inputs_transformed)
        self.clusters_indexes += new_cluster_indexes
        for i in range(len(new_inputs)):
            cluster_index = new_cluster_indexes[i]
            test_input = self.new_inputs[i]
            test_output = self.new_outputs[i]
            self.cluster_inputs[cluster_index].append(test_input)
            self.cluster_ouputs[cluster_index].append(test_output)
            self.cluster_input_choosers[cluster_index].append(test_input, test_output)

    def increase_cluster_weights(self, indices, increase=1):
        self.cluster_weights[indices] += increase
    
    def set_cluster_weights(self, indices, weights):
        self.cluster_weights[indices] = weights

    def increase_input_weights(self, cluster_index, indices, increase=1):
        self.cluster_input_choosers[cluster_index].increase_weights(indices, increase=increase)
    
    def set_input_weights(self, cluster_index, indices, weights):
        self.cluster_input_choosers[cluster_index].set_weights(indices, weights)
