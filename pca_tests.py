from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from main import PCA

data = load_iris()
data_X = data.data
data_Y = data.target

pca = PCA(2)
pca.fit(data_X)
X_projected = pca.transform(data_X)