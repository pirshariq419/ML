import statistics
import math
import numpy as np
import scipy.stats as stats

data = [float(x) for x in input("Enter numbers separated by spaces: ").split()]

mean=statistics.mean(data)
median=statistics.median(data)
stdev=statistics.stdev(data)

print("Statistics Library Results : ")
print(f"Mean : {mean}")
print(f"Median : {median}")
print(f"Standard Deviation : {stdev}")     
print()

X=25
sqrt_x=math.sqrt(X)
factorial_x=math.factorial(X)
pi_value=math.pi

print("Math Library Results : ")
print(f"Square Root of {X} is : {sqrt_x}")
print(f"Factorial of {X} is : {factorial_x}")
print(f"Value of Pi is : {pi_value}")
print()

arr =np.array([float(x) for x in input("Enter numbers seperated by spaces : ").split()])
mean_arr=np.mean(arr)
std_arr=np.std(arr)
sum_arr=np.sum(arr)

print("NumPy Library Results :")
print(f"Mean : {mean_arr}")
print(f"Standard Deviation : {std_arr}")
print(f"Sum : {sum_arr}")
print()

data_normal= stats.norm.rvs(loc=0,scale=1,size=1000)
mean_nor=np.mean(data_normal)
std_nor=np.std(data_normal)

print("SciPy Library Results : ")
print(f"Mean of Normal Distribution Data :  {mean_nor}")
print(f"Standard Deviation of Normal Distribution Data :  {std_nor}")

