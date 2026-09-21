import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


# Create dataset
data = {
    "Study_Hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                    6, 7, 7, 8, 8, 9, 9, 10, 10, 11],

    "Attendance": [50, 55, 60, 62, 65, 68, 70, 72, 75, 78,
                   80, 82, 84, 86, 88, 90, 92, 94, 95, 98],

    "Pass": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)


# Features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = LogisticRegression()

model.fit(X_train, y_train)


# Probability of passing
y_probability = model.predict_proba(X_test)[:, 1]


# ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


# AUC Score
auc = roc_auc_score(
    y_test,
    y_probability
)

print("AUC Score:", auc)


# Plot ROC curve
plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.2f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()
