
import numpy as np

# Internal marks of 10 students
marks = np.array([75, 82, 68, 90, 55, 72, 88, 64, 79, 85])

# Calculate statistics
mean = np.mean(marks)
median = np.median(marks)
std_dev = np.std(marks)
maximum = np.max(marks)
minimum = np.min(marks)

# Print results
print("Internal Marks:", marks)
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Maximum:", maximum)
print("Minimum:", minimum)
