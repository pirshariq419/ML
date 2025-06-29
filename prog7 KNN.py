import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

param_grid = {'n_neighbors': list(range(1, 21))}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best number of neighbors: {grid_search.best_params_['n_neighbors']}")

best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test)

print(f"Accuracy of Best Model: {accuracy_score(y_test, y_pred_best)}")
print("Classification Report of Best Model:")
print(classification_report(y_test, y_pred_best))
print("Confusion Matrix of Best Model:")
print(confusion_matrix(y_test, y_pred_best))
