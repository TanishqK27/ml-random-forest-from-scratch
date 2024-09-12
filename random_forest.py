from tqdm import tqdm  # Import tqdm for the loading bar
from decision_tree import DecisionTreeRegressor
import numpy as np

class RandomForestRegressor:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """Fits the random forest to the data with a progress bar."""
        self.trees = []
        for _ in tqdm(range(self.n_trees), desc="Building Trees"):
            self.trees.append(self._build_tree(X, y))

    def _build_tree(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample, y_sample = X[indices], y[indices]
        tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)