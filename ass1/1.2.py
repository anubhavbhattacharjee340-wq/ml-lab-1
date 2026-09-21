import pandas as pd
import numpy as np

marks = np.array([72, 85, 91, 68, 77])

print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Maximum:", np.max(marks))
print("Minimum:", np.min(marks))


data = {
    "Student Name": ["Amit", "Riya", "Sourav", "Neha", "Rahul"],
    "Roll Number": [101, 102, 103, 104, 105],
    "Marks": [72, 85, 68, 91, 77],
    "Attendance": [88, 92, 76, 95, 81]
}

df = pd.DataFrame(data)

print("\n--- Complete DataFrame ---")
print(df)


print("\n--- Students Scoring Above 80 Marks ---")

filtered_df = df[df["Marks"] > 80]

print(filtered_df)
