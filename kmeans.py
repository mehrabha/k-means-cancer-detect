import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('./data/data.csv')
dataset = dataset.to_numpy()

features = dataset[:, 2:]
labels = dataset[:, 1]

X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size=.2, shuffle=True)

kmeans = KMeans(n_clusters=2).fit(X_train)
predictions = kmeans.predict(X_test)


