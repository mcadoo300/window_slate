import pandas as pd
import matplotlib.pyplot as plt
import pdb
slate_size=4
sample_size=5000

fileName =f'intervals_window_additive_0.5_{slate_size}_10_{sample_size}_total_error.xlsx'

# Specify the range of rows you want to select
start_row = 9  # Start row index (inclusive)
end_row = 12   # End row index (exclusive)

# Read the specific range of rows from the Excel file into a pandas DataFrame
df = pd.read_excel(fileName, skiprows=start_row - 1, nrows=end_row - start_row + 1)

# Assuming the first column contains labels and the second column contains values
labels = df.iloc[:, 0]  # Selecting the first column as labels
values = df.iloc[:, 1]  # Selecting the second column as values
# Plotting the bar graph
bars = plt.bar(labels, values)
# Plotting the bar graph
#plt.bar(labels, values)

horizontal_line_value = df.iloc[0,-1]  # Adjust
print(horizontal_line_value)
plt.axhline(y=horizontal_line_value, color='r', linestyle='--')  # Add a horizontal line at y=10
# Add labels and title

for value, bar in zip(values, bars):
    #pdb.set_trace()
    print(value)
    print(horizontal_line_value)
    if value < horizontal_line_value:
        bar.set_color('blue')
    else:
        bar.set_color('green')

plt.xlabel('OPE Method')
plt.ylabel('Policy Value')
plt.title(f'Dissimilar Policy \n size={slate_size} // sample size={sample_size}')

# Rotating x-axis labels for better visibility if needed
plt.xticks(rotation=45)

# Show the plot
plt.show()