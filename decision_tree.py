import numpy as np
from utils import best_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Fits the tree to the data."""
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (num_samples >= self.min_samples_split) and (depth < self.max_depth):
            feature, threshold = best_split(X, y)
            if feature is not None:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
                right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
                return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

        return Node(value=np.mean(y))

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)