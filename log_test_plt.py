import pandas as pd
import matplotlib.pyplot as plt

slate_size = 3
sample_size = 5000

fileName = f'intervals_window_additive_0.5_{slate_size}_10_{sample_size}_total_error.xlsx'

# Specify the ranges of rows you want to select
row_ranges = [(1, 4), (5, 8), (9, 13)]

# Read data for the first range to get labels
start_row, end_row = row_ranges[0]
df_first_range = pd.read_excel(fileName, skiprows=start_row - 1, nrows=end_row - start_row + 1)
labels = df_first_range.iloc[:, 0]

# Initialize lists to store values for all ranges
all_values = []

# Read data for each range and append to the list of values
for start_row, end_row in row_ranges:
    df = pd.read_excel(fileName, skiprows=start_row - 1, nrows=end_row - start_row + 1)
    all_values.append(df.iloc[:, 1])

# Plotting the bar graph for all groups
for values in all_values:
    plt.bar(labels, values)

plt.xlabel('OPE Method')
plt.ylabel('Policy Value')
plt.title(f'Random Policy \n size={slate_size} // sample size={sample_size}')

# Add legend for each group
plt.legend([f'Rows {start}-{end}' for start, end in row_ranges])

# Rotating x-axis labels for better visibility if needed
plt.xticks(rotation=45)

# Show the plot
plt.show()