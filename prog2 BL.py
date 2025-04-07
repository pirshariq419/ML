import statistics
import math
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import scipy.integrate as integrate

data = [12, 15, 14, 10, 18, 21, 22, 13, 17, 20]

mean_data = statistics.mean(data)
median_data = statistics.median(data)
stdev_data = statistics.stdev(data)

print("Statistics Library Results:")
print(f"Mean: {mean_data}")
print(f"Median: {median_data}")
print(f"Standard Deviation: {stdev_data}")
print()

x = 25
sqrt_x = math.sqrt(x)
factorial_5 = math.factorial(5)
pi_value = math.pi

print("Math Library Results:")
print(f"Square Root of {x}: {sqrt_x}")
print(f"Factorial of 5: {factorial_5}")
print(f"Value of Pi: {pi_value}")
print()

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

mean_arr = np.mean(arr)
std_arr = np.std(arr)
sum_arr = np.sum(arr)
random_numbers = np.random.rand(5)

print("NumPy Library Results:")
print(f"Mean of array: {mean_arr}")
print(f"Standard Deviation of array: {std_arr}")
print(f"Sum of array elements: {sum_arr}")
print(f"Random Numbers: {random_numbers}")
print()

data_normal = stats.norm.rvs(loc=0, scale=1, size=1000)

mean_normal = np.mean(data_normal)
std_normal = np.std(data_normal)

def quadratic(x):
    return x**2 - 4*x + 4

opt_result = optimize.minimize(quadratic, 0)

result_integral, error = integrate.quad(lambda x: x**2, 0, 1)

print("SciPy Library Results:")
print(f"Generated Normal Distribution Data - Mean: {mean_normal}, Std Dev: {std_normal}")
print(f"Optimized result of quadratic function: {opt_result.x}")
print(f"Numerical integration result of x^2 from 0 to 1: {result_integral} (Error estimate: {error})")
