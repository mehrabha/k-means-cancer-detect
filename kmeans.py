import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/data.csv')
dataset = dataset.to_numpy()

features = dataset[:, 2:]
labels = dataset[:, 1]
