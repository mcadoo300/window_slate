import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

random_data = np.empty((0, 2), dtype=object)
similar_data = np.empty((0, 2), dtype=object)
dissimilar_data = np.empty((0, 2), dtype=object)

samples = [1, 2]
ope = ["wipss","wiips","sips", "iips", "rips",   "wiipsfull"]

for o in ope:
    row_random = np.empty((1, 0))
    row_similar = np.empty((1, 0))
    row_dissimilar = np.empty((1, 0))
    
    for s in samples:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(f"slate_5/{o}_window_additive_0.5_5_{s}000_error.xlsx")
        
        # Calculate the mean of each column
        column_means = df.mean()
        #pdb.set_trace()
        #print(f"slate_3/{o}_window_additive_0.5_3_{s}000_error.xlsx")
        #print(df)
        #print(column_means)
        #print([column_means["dissimilar"]])
        # Append means to respective rows
        row_random = np.column_stack((row_random, [column_means["random"]]))
        row_similar = np.column_stack((row_similar, [column_means["similar"]]))
        row_dissimilar = np.column_stack((row_dissimilar, [column_means["dissimilar"]]))
    
    # Append rows to respective data arrays
    random_data = np.append(random_data, row_random, axis=0)
    similar_data = np.append(similar_data, row_similar, axis=0)
    dissimilar_data = np.append(dissimilar_data, row_dissimilar, axis=0)
labels_column = np.array(ope)[:, np.newaxis]

# Stack labels_column and array horizontally
random_data = np.hstack((labels_column, random_data))
similar_data = np.hstack((labels_column, similar_data))
dissimilar_data = np.hstack((labels_column, dissimilar_data))
print(random_data)
# Plotting
for i in range(random_data.shape[0]):
    label = random_data[i, 0]
    data = random_data[i, 1:].astype('float64')
    plt.plot(data, label=label)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph with Labels')
plt.legend()
plt.show()
plt.clf()
# Plotting
for i in range(similar_data.shape[0]):
    label = similar_data[i, 0]
    data = similar_data[i, 1:].astype(float)
    plt.plot(data, label=label)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph with Labels')
plt.legend()
plt.show()
plt.clf()
# Plotting
for i in range(dissimilar_data.shape[0]):
    label = dissimilar_data[i, 0]
    data = dissimilar_data[i, 1:].astype(float)
    plt.plot(data, label=label)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph with Labels')
plt.legend()
plt.show()
