
from random_forest import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the custom Random Forest
rf = RandomForestRegressor(n_trees=5, max_depth=15)
rf.fit(X_train, y_train)

# Predict and evaluate custom model
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Custom Random Forest Mean Squared Error: {mse:.4f}")

# Visualize custom model predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    lw=2
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Custom Model: Actual vs. Predicted Values')
plt.show()

# Compare with Scikit-Learn's RandomForestRegressor
sklearn_rf = SklearnRandomForestRegressor(
    n_estimators=100, max_depth=15, random_state=42
)
sklearn_rf.fit(X_train, y_train)

# Predict and evaluate Scikit-Learn model
y_pred_sklearn = sklearn_rf.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(f"Scikit-Learn Random Forest Mean Squared Error: {mse_sklearn:.4f}")

# Visualize Scikit-Learn model predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_sklearn, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',
    lw=2
)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Scikit-Learn Model: Actual vs. Predicted Values')
plt.show()