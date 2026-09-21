import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Create dataset
data = {
    "Age": [22, 25, 30, 35, 40],
    "Salary": [25000, 30000, 45000, 55000, 70000],
    "Years_of_Experience": [1, 2, 5, 8, 12]
}

df = pd.DataFrame(data)

# StandardScaler
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(df)

# MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(df)

print("Original Data:")
print(df)

print("\nStandardScaler:")
print(standard_data)

print("\nMinMaxScaler:")
print(minmax_data)

# Compare ranges
print("\nStandardScaler Range:")
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())

print("\nMinMaxScaler Range:")
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())
