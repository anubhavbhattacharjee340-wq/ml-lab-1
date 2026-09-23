import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# 1. Load the classic wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 2. Standardize features to a uniform scale for visual comparison
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=wine.feature_names)

# 3. Plot horizontal boxplots for all features
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_scaled, orient='h', palette='vlag', flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 6})

plt.title('Standardized Boxplot of Wine Dataset Attributes (Outlier Identification)', fontsize=14, pad=15)
plt.xlabel('Standardized Score (Z-Score)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5) # Reference line at center
plt.tight_layout()
plt.show()
