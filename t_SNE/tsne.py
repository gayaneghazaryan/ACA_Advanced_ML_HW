from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class TSNE:

    def __init__(self, n_components = 2, perplexity= 30.0, momentum = 0.5, learning_rate = 200.0, n_iter= 1000.0, metric = 'euclidean'):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.metric = metric
        self.momentum = momentum

    def fit_transform(self, X):
        n_samples = X.shape[0]

        p_ij = self._calculate_joint_probabilities(X)
        y = np.random.normal(0,1e-4, size = (n_samples,2))

        for t in range(1, self.n_iter + 1):
            q_ij = self._calculate_low_dim_affinities(y)
            grad = self._compute_gradient(p_ij, q_ij, y)
            
            if t > 1:
                y[t] = y[t-1] + self.learning_rate * grad + self.momentum * (y[t-1] + y[t-2])
            else:
                y[t] = y[t-1] + self.learning_rate * grad

        return y


    
    def _calculate_joint_probabilities(self, X):

        distances = pairwise_distances(X, metric = self.metric)
        p_ij = self._binary_search(distances, self.perplexity)
        p_ij = (p_ij + p_ij.T) / (2.0 * X.shape[0])

        return p_ij
    
    def _binary_search(self, distances, perplexity, epsilon = 1e-5, max_iterations = 1000):
        n = distances.shape[0]
        p_ij = np.zeroes((n,n))

        for i in range(n):
            beta_min = - np.inf
            beta_max = np.inf
            beta = 1.0
            current_distances = distances[1, np.concatenate((np.arange(0, i), np.arange(i + 1, n)))]
            sum_Pi = 0
            iter_count = 0

            while iter_count < max_iterations:
                affinities = np.exp(-current_distances * beta)
                sum_affinities = np.sum(affinities)
                Pi = affinities / (sum_affinities + epsilon)
                sum_Pi = np.sum(Pi)

                entropy = np.sum(Pi * np.log2((Pi + epsilon) / (sum_Pi + epsilon)))
                perplexity_diff = entropy - np.log2(perplexity)

                if abs(perplexity_diff) < epsilon:
                    break

                if perplexity_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2.0

                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2.0
                    else:
                        beta = (beta + beta_min) / 2.0

            p_ij[i, np.concatenate((np.arange(0,i), np.arange(i+1, n)))] = Pi

    def _calculate_low_dim_affinities(self, y):
        distances = pairwise_distances(y, metric = self.metric)

        inv_distances = np.maximum(distances, 1e-12)
        q_ij = 1.0 / (1.0 + inv_distances**2)
        np.fill_diagonal(q_ij, 0)

        q_ij = q_ij/ np.sum(q_ij)

        return q_ij
    
    def _compute_gradient(self, p_ij, q_ij, y):
        grad = np.zeroes((len(p_ij),y.shape[1]))

        for i in range(len(p_ij)):

            grad[i] = 4.0 * np.sum((np.array([(p_ij[i,:] - q_ij[i,:])]) * np.array([1/(1+np.linalg.norm(y[i]-y, axis = 1))])).T*(y[i]-y), axis = 0)

        return grad




        




