from os import makedirs
from sklearn.cluster import KMeans
import pickle
import numpy as np


class Anomaly_KMeans():
    def __init__(self, clusters = 1, model_path = None):
        super(Anomaly_KMeans, self).__init__()
        self.model = KMeans(n_clusters = clusters, random_state=0)
        self.estimatior = None
        self.max_distances = None
        if model_path:
            self.load_model(model_path)

    def train(self, data_array):
        self.estimatior = self.model.fit(data_array)
        return self.estimatior.labels_
    def load_model(self, pickel_path):

        with open(pickel_path) as f:
            params = pickle.load(f)
        self.estimatior = self.model.set_params(params)

    def save_model(self, save_path):
        params = self.estimatior.get_params()
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)

    def transform(self, data):
        return self.estimatior.transform(data)

    def set_anomaly_metrics(self, metrics):
        self.max_distances = metrics

    def is_anomaly(self, data):
        assert self.max_distances is not None
        distances = self.transform(data)
        min_distance = 0
        for pred_val, max_val in (distances, self.max_distances):
            min_distance = abs(max_val - pred_val)
            if pred_val < max_val:
                return True, min_distance
            
        return False, min_distance

