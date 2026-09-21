import pandas as pd
import numpy as np

# Create dataset
data = {
    "Age": [22, 25, 30, 35, 40],
    "Salary": [25000, 30000, 45000, 55000, 70000],
    "Department": ["IT", "HR", "IT", "Finance", "HR"],
    "Years_of_Experience": [1, 2, 5, 8, 12]
}

df = pd.DataFrame(data)

# Add missing values
df.loc[1, "Age"] = np.nan
df.loc[2, "Salary"] = np.nan
df.loc[3, "Department"] = np.nan
df.loc[4, "Years_of_Experience"] = np.nan

print("Dataset with missing values:")
print(df)

# Fill missing numerical values with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())
df["Years_of_Experience"] = df["Years_of_Experience"].fillna(
    df["Years_of_Experience"].mean()
)

# Fill missing Department with mode
df["Department"] = df["Department"].fillna(
    df["Department"].mode()[0]
)

print("\nAfter preprocessing:")
print(df)
