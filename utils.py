import numpy as np


def calculate_mse(y):
    """Calculates the mean squared error for a set of target values y."""
    if len(y) == 0:
        return 0
    return np.mean((y - np.mean(y)) ** 2)


def best_split(X, y):
    """Finds the best feature and threshold to split on to minimize MSE."""
    num_samples, num_features = X.shape
    if num_samples <= 1:
        return None, None

    best_mse = float('inf')
    best_feature = None
    best_threshold = None

    # Loop over each feature
    for feature in range(num_features):
        # Sort the data along the current feature
        sorted_indices = np.argsort(X[:, feature])
        X_sorted, y_sorted = X[sorted_indices], y[sorted_indices]

        # Try splitting at the midpoints between each consecutive pair of unique values
        for i in range(1, num_samples):
            if X_sorted[i, feature] == X_sorted[i - 1, feature]:
                continue  # Skip if the value is the same as the previous one

            threshold = (X_sorted[i, feature] + X_sorted[i - 1, feature]) / 2

            # Split the data
            left_indices = X[:, feature] <= threshold
            right_indices = X[:, feature] > threshold

            y_left, y_right = y[left_indices], y[right_indices]

            # Calculate MSE for the split
            mse_left, mse_right = calculate_mse(y_left), calculate_mse(y_right)
            mse_split = (len(y_left) * mse_left + len(y_right) * mse_right) / num_samples

            # Update the best split if this split is better
            if mse_split < best_mse:
                best_mse = mse_split
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold