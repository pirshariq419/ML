import statistics

def central_tendency_measures(data):
    mean = statistics.mean(data)
    median = statistics.median(data)
    try:
        mode = statistics.mode(data)
    except statistics.StatisticsError:
        mode = "No unique mode"
    return mean, median, mode

def measures_of_dispersion(data):
    variance = statistics.variance(data)
    std_deviation = statistics.stdev(data)
    return variance, std_deviation

data = [float(x) for x in input("Enter numbers separated by spaces: ").split()]

mean, median, mode = central_tendency_measures(data)
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")

variance, std_deviation = measures_of_dispersion(data)
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_deviation}")
