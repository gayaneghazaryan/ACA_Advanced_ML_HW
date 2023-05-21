import numpy as np
from sklearn.cluster import  KMeans
from sklearn.metrics import pairwise_kernels

from scipy.sparse import csgraph

class SpectralClustering:
  def __init__(self, n_clusters=8, gamma = 1.0, random_state = None):
    self.n_clusters = n_clusters
    self.gamma = gamma
    self.random_state = random_state
  
  def fit_predict(self, X):
    affinity_matrix = pairwise_kernels(X, metric='rbf', gamma=self.gamma)
    
    laplacian_matrix = csgraph.laplacian(affinity_matrix, normed = False)

    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

    indices = np.argsort(eigenvalues)[:self.n_clusters]
    eigenvectors = eigenvectors[:, indices]

    row_norms = np.linalg.norm(eigenvectors, axis=1)
    eigenvectors = eigenvectors / row_norms[:, np.newaxis]

    kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
    kmeans.fit(eigenvectors)

    return kmeans.labels_