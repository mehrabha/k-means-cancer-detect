import numpy as np

class KMeans:
    def __init__(self, n_clusters):
        self.X = None

    def fit(self, X):
        self.X = X
        # TODO find centroids

        return self
    
    def predict(self, X):
        result = np.zeros(X.shape)

        # TODO predict X
        
        return result
    