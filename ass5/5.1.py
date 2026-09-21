import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Create student dataset
data = {
    "Study_Hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                    6, 7, 7, 8, 8, 9, 9, 10, 10, 11],

    "Attendance": [50, 55, 60, 62, 65, 68, 70, 72, 75, 78,
                   80, 82, 84, 86, 88, 90, 92, 94, 95, 98],

    "Pass": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df)


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


# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))


# Probability of passing
probability = model.predict_proba(X_test)[:, 1]

print("\nPass Probability:")
print(probability)
