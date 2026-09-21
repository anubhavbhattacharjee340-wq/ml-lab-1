import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Price"] = housing.target

# Feature and target
X = df[["MedInc"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, y_pred_linear)

print("Linear Regression R2 Score:", linear_r2)




poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

poly_r2 = r2_score(y_test, y_pred_poly)

print("Polynomial Regression R2 Score:", poly_r2)



print("\nComparison:")
print("Linear Regression R2:", linear_r2)
print("Polynomial Regression R2:", poly_r2)


plt.scatter(X_test, y_test, label="Actual")

plt.scatter(
    X_test,
    y_pred_linear,
    label="Linear Prediction"
)

plt.scatter(
    X_test,
    y_pred_poly,
    label="Polynomial Prediction"
)

plt.xlabel("Median Income")
plt.ylabel("House Price")

plt.title("Linear vs Polynomial Regression")

plt.legend()

plt.show()
