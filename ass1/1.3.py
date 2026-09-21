import pandas as pd
import numpy as np

marks = np.array([72, 85, 91, 68, 77])

print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))


data = {
    "Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
    "Attendance": [88, 92, 76, 95, 81],
    "Marks": [72, 85, 68, 91, 77]
}

df = pd.DataFrame(data)


# Calculate Grade based on Marks
def calculate_grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    else:
        return "F"


# Add Grade column
df["Grade"] = df["Marks"].apply(calculate_grade)


print("\n--- DataFrame with Grade ---")
print(df)
