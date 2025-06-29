import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Name': ['Aman', 'Bina', 'Chetan'],
    'Marks': [80, 90, 85]
}

df = pd.DataFrame(data)
print(df)

plt.bar(df['Name'], df['Marks'], color='skyblue')
plt.title('Student Marks')
plt.xlabel('Name')
plt.ylabel('Marks')
plt.show()
