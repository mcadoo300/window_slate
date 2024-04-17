import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
random_data = np.empty((0, 5), dtype=object)
similar_data = np.empty((0, 5), dtype=object)
dissimilar_data = np.empty((0, 5), dtype=object)

samples = [1, 2, 3, 4, 5]
ope = ["sips", "iips", "rips", "wiips", "wipss", "wiipsfull"]

for o in ope:
    row_random = np.empty((1, 0))
    row_similar = np.empty((1, 0))
    row_dissimilar = np.empty((1, 0))
    
    for s in samples:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(f"slate_3/{o}_window_additive_0.5_3_{s}000_error.xlsx")

        # Calculate the mean of each column
        column_means = df.mean()

        # Append means to respective rows
        row_random = np.column_stack((row_random, [column_means["random"]]))
        row_similar = np.column_stack((row_similar, [column_means["similar"]]))
        row_dissimilar = np.column_stack((row_dissimilar, [column_means["dissimilar"]]))
    
    # Append rows to respective data arrays
    random_data = np.append(random_data, row_random, axis=0)
    similar_data = np.append(similar_data, row_similar, axis=0)
    dissimilar_data = np.append(dissimilar_data, row_dissimilar, axis=0)


# Create subplots with 3 rows and 1 column
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot random data
axs[0].plot(random_data, label='Random Data')
axs[0].set_title('Random Data')
axs[0].set_xlabel('Sample')
axs[0].set_ylabel('Mean')
axs[0].legend()

# Plot similar data
axs[1].plot(similar_data, label='Similar Data')
axs[1].set_title('Similar Data')
axs[1].set_xlabel('Sample')
axs[1].set_ylabel('Mean')
axs[1].legend()

# Plot dissimilar data
axs[2].plot(dissimilar_data, label='Dissimilar Data')
axs[2].set_title('Dissimilar Data')
axs[2].set_xlabel('Sample')
axs[2].set_ylabel('Mean')
axs[2].legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()