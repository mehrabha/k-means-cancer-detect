import numpy as np
import math

class KMeans:
    def __init__(self, k_clusters):
        self.k = k_clusters
        self.centroids = None

    def fit(self, X):
        indexes = np.random.choice(X.shape[0], self.k, False)
        self.centroids = X[indexes]

        while True:
            clusters = []
            for i in range(self.k):
                clusters.append([])

            for row in X:
                # Assign row to nearest cluster
                min_distance = math.inf
                min_indx = math.inf
                for i in range(self.k):
                    d = dist_euclid(self.centroids[i], row)
                    if d < min_distance:
                        min_distance = d
                        min_indx = i
                clusters[min_indx].append(row)
            
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
        result = np.zeros(X.shape)

        # TODO predict X
        
        return result

def dist_euclid(vector1, vector2):
    d = vector2 - vector1
    sum_squared = np.sum(np.square(d))
    return np.sqrt(sum_squared)