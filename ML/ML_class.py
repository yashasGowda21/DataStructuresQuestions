# Linear Regression
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)  # Initialize weights
        self.b = 0  # Initialize bias

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            dw = (1/self.m) * np.dot(X.T, (y_pred - y))  # Gradient w.r.t weights
            db = (1/self.m) * np.sum(y_pred - y)  # Gradient w.r.t bias
            self.W -= self.lr * dw  # Update weights
            self.b -= self.lr * db  # Update bias

    def predict(self, X):
        return np.dot(X, self.W) + self.b  # Linear prediction

# Logistic Regression
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid activation

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)  # Initialize weights
        self.b = 0  # Initialize bias

        for _ in range(self.epochs):
            y_pred = self.sigmoid(np.dot(X, self.W) + self.b)
            dw = (1/self.m) * np.dot(X.T, (y_pred - y))  # Gradient
            db = (1/self.m) * np.sum(y_pred - y)
            self.W -= self.lr * dw  # Update weights
            self.b -= self.lr * db  # Update bias

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.W) + self.b) >= 0.5).astype('int')  # Class prediction

# K-Nearest Neighbors (KNN)
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X  # Memorize training data
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)  # Compute distance
            k_indices = distances.argsort()[:self.k]  # Find k nearest
            k_labels = self.y_train[k_indices]
            preds.append(Counter(k_labels).most_common(1)[0][0])  # Majority vote
        return np.array(preds)

# Decision Tree (ID3)
class DecisionTree:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, y):
        if len(set(y)) == 1 or self.depth == self.max_depth:
            self.prediction = Counter(y).most_common(1)[0][0]  # Leaf node prediction
            return

        best_feat, best_thresh = self.best_split(X, y)
        if best_feat is None:
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        self.feat = best_feat
        self.thresh = best_thresh
        self.left = DecisionTree(self.depth + 1, self.max_depth)
        self.right = DecisionTree(self.depth + 1, self.max_depth)
        self.left.fit(X[left_idx], y[left_idx])  # Left subtree
        self.right.fit(X[right_idx], y[right_idx])  # Right subtree

    def best_split(self, X, y):
        m, n = X.shape
        best_gain = 0
        best_feat, best_thresh = None, None
        for feat in range(n):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = X[:, feat] > thresh
                if len(set(y[left_idx])) == 0 or len(set(y[right_idx])) == 0:
                    continue
                gain = self.info_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feat, best_thresh = feat, thresh
        return best_feat, best_thresh

    def info_gain(self, parent, l, r):
        def entropy(y):
            probs = np.bincount(y) / len(y)
            return -np.sum([p*np.log2(p) for p in probs if p > 0])
        return entropy(parent) - (len(l)/len(parent))*entropy(l) - (len(r)/len(parent))*entropy(r)

    def predict(self, X):
        if hasattr(self, 'prediction'):
            return self.prediction  # Leaf node
        branch = self.left if X[self.feat] <= self.thresh else self.right
        return branch.predict(X)

# Naive Bayes
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)  # Mean per class
            self.var[c] = X_c.var(axis=0) + 1e-9  # Variance per class
            self.priors[c] = X_c.shape[0] / n_samples  # Prior probability

    def predict(self, X):
        posteriors = []
        for x in X:
            post_c = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c]))
                likelihood -= 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c])
                post_c.append(prior + likelihood)
            posteriors.append(self.classes[np.argmax(post_c)])
        return np.array(posteriors)

# KMeans Clustering
class KMeans:
    def __init__(self, K=3, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def fit(self, X):
        self.X = X
        self.m, self.n = X.shape

        # Step 1: Initialize centroids randomly
        random_indices = np.random.choice(self.m, self.K, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Step 2: Assign each point to the nearest centroid
            self.labels = self._assign_clusters(self.X)

            # Step 3: Update centroids
            new_centroids = np.array([self.X[self.labels == k].mean(axis=0) for k in range(self.K)])

            # Convergence check
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)  # Assign to nearest cluster

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)  # Assign test points to nearest centroid


class LinearSVM:
    def __init__(self, lr=0.001, C=1.0, epochs=1000):
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(m):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.w)
                else:
                    self.w -= self.lr * (2 * self.w - self.C * y[i] * X[i])
                    self.b -= self.lr * self.C * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

import numpy as np

def pca(X, k):
    # Step 1: Center the data by subtracting the mean of each feature (column-wise)
    X_centered = X - X.mean(axis=0)

    # Step 2: Compute the covariance matrix of the centered data
    # Transpose X_centered because np.cov expects features in rows
    cov = np.cov(X_centered.T)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eig_vals, eig_vecs = np.linalg.eig(cov)

    # Step 4: Select the top 'k' eigenvectors corresponding to the largest eigenvalues
    # np.argsort(eig_vals) returns indices that would sort the eigenvalues in ascending order
    # [-k:] selects the indices of the top k eigenvalues (largest ones)
    # The eigenvectors (columns of eig_vecs) corresponding to these eigenvalues form the principal components
    components = eig_vecs[:, np.argsort(eig_vals)[-k:]]

    # Step 5: Project the centered data onto the top 'k' principal components
    return np.dot(X_centered, components)

