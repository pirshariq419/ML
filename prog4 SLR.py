import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

x_mean = np.mean(X)
y_mean = np.mean(y)

numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
m = numerator / denominator
c = y_mean - m * x_mean

y_pred = m * X + c

print(f"Slope : {m}")
print(f"Intercept: {c}")

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
