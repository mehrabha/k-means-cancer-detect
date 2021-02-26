import numpy as np
import math

class KMeans:
    def __init__(self, k_clusters):
        self.k = k_clusters
        self.centroids = None

    def fit(self, X):
        # Select k different rows as centroids
        indexes = np.random.choice(X.shape[0], self.k, False)
        self.centroids = X[indexes]

        while True:
            clusters = []
            for i in range(self.k):
                clusters.append([])

            for row in X:
                # Assign row to nearest cluster
                nearest = find_centroid(self.centroids, row)
                clusters[nearest].append(row)
            
            # Calculate new centroids
            new_centroids = []
            for cluster in clusters:
                mean = np.mean(cluster, axis=0)
                new_centroids.append(mean)
            
            # Break if new centroids are same as old
            # else move to new centroids
            if np.array_equal(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids
                
        return self
    
    def predict(self, X):
        result = np.full(X.shape[0], -1, dtype=np.int32)

        # Find nearest center for each row
        for i in range(X.shape[0]):
            nearest = find_centroid(self.centroids, X[i])
            result[i] = nearest

        return result
    

def dist_euclid(vector1, vector2):
    d = vector2 - vector1
    sum_squared = np.sum(np.square(d))
    return np.sqrt(sum_squared)

def find_centroid(centroids, vector):
    min_distance = math.inf
    min_indx = math.inf
    for i in range(len(centroids)):
        d = dist_euclid(centroids[i], vector)
        if d < min_distance:
            min_distance = d
            min_indx = i
    return min_indx