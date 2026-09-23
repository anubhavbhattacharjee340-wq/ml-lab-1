import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
data = {
    "Feature_A": np.random.rand(100),
    "Feature_B": np.random.rand(100),
    "Feature_C": np.random.rand(100),
    "Feature_D": np.random.rand(100),
}
data["Feature_C"] = data["Feature_A"] * 0.85 + np.random.rand(100) * 0.15

df = pd.DataFrame(data)


corr_matrix = df.corr()


plt.figure(figsize=(8, 6))
# Using a diverging palette ('coolwarm') where dark red represents high positive correlation
sns.heatmap(
    corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".3f"
)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()
np.fill_diagonal(corr_matrix.values, np.nan)
unstacked_corr = corr_matrix.unstack()
strongest_value = unstacked_corr.max()
strongest_pair = unstacked_corr.idxmax()

print(
    f"Strongest Positive Correlation: {strongest_value:.3f} between {strongest_pair}"
)
