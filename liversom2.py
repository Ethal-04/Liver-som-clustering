import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Patch

# Load the liver dataset (ensure the file is correctly placed)
blood_data = np.loadtxt(r"C:\Users\Sheethal\Downloads\Project\liverdata.txt", delimiter="\t")

# Convert to DataFrame for easier handling and column naming
df = pd.DataFrame(blood_data, columns=['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                                       'Aspartate_Aminotransferase', 'Albumin'])

# Apply a simple rule to classify 'Disease_Status'
# For example: if Total_Bilirubin > 2.0 or AST > 50, classify as having liver disease (1), else healthy (2)
df['Disease_Status'] = np.where((df['Total_Bilirubin'] > 2.0) | (df['Aspartate_Aminotransferase'] > 50), 1, 2)

# Normalize the data (important for SOM)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df.drop(columns=['Disease_Status']))

# Initialize and train the SOM
som = MiniSom(x=7, y=7, input_len=normalized_data.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(normalized_data)
som.train_random(normalized_data, 200)  # Increase iterations if needed

# Map each sample to its best matching unit (BMU)
mapped_data = np.array([som.winner(x) for x in normalized_data])

# Create a heatmap of how many samples mapped to each neuron
heatmap = np.zeros((7, 7))
for (x, y) in mapped_data:
    heatmap[x, y] += 1

# Visualize the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap, cmap='viridis')

# Add a color bar for the number of samples in each cluster
plt.colorbar(im, label="Sample Count in Cluster")
plt.title("SOM Clustering of Liver-Related Blood Parameters")

# Step 2: Color-code the clusters based on disease status
cluster_labels = np.array([df['Disease_Status'].iloc[i] for i in range(len(df))])

# Map the disease status to colors: (e.g., 1 = red for disease, 2 = green for healthy)
disease_colors = np.where(cluster_labels == 1, 'red', 'green')

# Step 3: Scatter the clusters with color-coding based on disease status
for i in range(len(mapped_data)):
    ax.scatter(mapped_data[i][1], mapped_data[i][0], color=disease_colors[i], alpha=0.6, edgecolors="w", s=80)

# Add a legend to show disease status
legend_labels = [Patch(color='red', label='Liver Disease'), Patch(color='green', label='No Disease'),
                 Patch(color='brown', label='Empty/Sparse Regions')]

plt.legend(handles=legend_labels, loc='upper right')

plt.xlabel("SOM X-axis")
plt.ylabel("SOM Y-axis")

# Show the plot before entering patient data
plt.show()

# Now, let's classify a single patient's data

# Example: Prompt for single patient data (adjust these values as necessary)
single_patient_input = input("Enter values separated by space (Total_Bilirubin Direct_Bilirubin Alkaline_Phosphotase Aspartate_Aminotransferase Albumin): ")
single_patient = [float(x) for x in single_patient_input.split()]

# Normalize the patient's data using the same scaler
normalized_patient_data = scaler.transform([single_patient])

# Map the patient's data to the SOM grid
bmu = som.winner(normalized_patient_data[0])

# Plot the SOM grid again, but now mark the patient's position
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap, cmap='viridis')
plt.colorbar(im, label="Sample Count in Cluster")
plt.title("SOM Clustering of Liver-Related Blood Parameters (with Patient's Position)")

# Plot the clusters
for i in range(len(mapped_data)):
    ax.scatter(mapped_data[i][1], mapped_data[i][0], color=disease_colors[i], alpha=0.6, edgecolors="w", s=80)

# Mark the patient's position on the SOM grid (using a distinct color like 'blue')
ax.scatter(bmu[1], bmu[0], color='blue', edgecolors="w", s=100, label='Patient\'s BMU')

# Add a legend for the patient's position
plt.legend(handles=legend_labels + [Patch(color='blue', label="Patient's Position")], loc='upper right')

# Show the plot with the patient's data marked
plt.xlabel("SOM X-axis")
plt.ylabel("SOM Y-axis")
plt.show()

# Output the Best Matching Unit (BMU) for the patient's data
print(f"The Best Matching Unit (BMU) for this patient is at: {bmu}")
