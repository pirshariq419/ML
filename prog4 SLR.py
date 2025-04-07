import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  
y = np.array([1.2, 1.9, 3.1, 3.9, 5.2, 6.1, 6.9, 8.1, 9.0, 10.2]) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

plt.scatter(X, y, color='blue', label='Data Points')  
plt.plot(X, model.predict(X), color='red', label='Regression Line') 
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


print(f"Model accuracy (R^2 score) on test data: {model.score(X_test, y_test)}")
