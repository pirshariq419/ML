import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

print(df.head())
print(df.describe())
print(df['species'].value_counts())

df['sepal_length'].hist(bins=20, color='skyblue', edgecolor='black')
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

colors = {'setosa':'red', 'versicolor':'green', 'virginica':'blue'}
plt.scatter(df['sepal_length'], df['sepal_width'], c=df['species'].map(colors))
plt.title('Sepal Length vs Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

df.boxplot(column='petal_length', by='species', grid=False)
plt.title('Petal Length by Species')
plt.suptitle('')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()
