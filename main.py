import numpy as np


class PCA:
    def __init__(self, number_components):
        self.number_components = number_components
        self.mean = None
        self.components = None


    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        data = data - self.mean
        covariance = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        idxs = np.argsort(eigenvectors.T)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]


        #store n
        self.components = eigenvectors[0:self.number_components]
    def transform(self, data):
        data = data - self.mean
        return np.dot(data, self.components.T)


