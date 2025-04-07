import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear', 'saga']
}

grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")

best_log_reg = grid_search.best_estimator_
y_pred_best = best_log_reg.predict(X_test)

print(f"Accuracy of Best Model: {accuracy_score(y_test, y_pred_best)}")
print("Classification Report of Best Model:")
print(classification_report(y_test, y_pred_best))
print("Confusion Matrix of Best Model:")
print(confusion_matrix(y_test, y_pred_best))
