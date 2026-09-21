import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Create dataset
data = {
    "Age": [22, 25, 30, 35, 40],
    "Salary": [25000, 30000, 45000, 55000, 70000],
    "Years_of_Experience": [1, 2, 5, 8, 12]
}

df = pd.DataFrame(data)

# Apply MinMaxScaler
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Original Data:")
print(df)

print("\nMinMaxScaler Data:")
print(scaled_data)

print("\nMinimum value:", scaled_data.min())
print("Maximum value:", scaled_data.max())
