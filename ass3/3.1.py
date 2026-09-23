
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
df['target'] = wine_data.target
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print("\nTarget Class Distribution:")
print(df['target'].value_counts(normalize=True))
print("\nFeature Summary:")
print(df.describe().T[['mean', 'std', 'min', 'max']])
