import numpy as np
import matplotlib.pyplot as plt

X_input = input("Enter X values separated by commas: ")
y_input = input("Enter y values separated by commas: ")

X = np.array([float(i) for i in X_input.split(',')])
y = np.array([float(i) for i in y_input.split(',')])

if len(X) != len(y):
    print("Error")
    exit()

x_mean = np.mean(X)
y_mean = np.mean(y)

numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
m = numerator / denominator
c = y_mean - m * x_mean

y_pred = m * X + c

print(f"Slope : {m}")
print(f"Intercept: {c}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
