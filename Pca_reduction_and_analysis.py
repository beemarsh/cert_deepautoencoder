import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the email_final.csv file
df = pd.read_csv("http_final.csv")

# Select the 50 GloVe embedding columns
glove_columns = [f'c{i}' for i in range(50)]  # Adjust column names to match your data
embeddings = df[glove_columns]

# Perform PCA
pca = PCA(n_components=50)  # Preserve all components initially
pca.fit(embeddings)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Calculate variance retained by the first 20 PCs
variance_retained_20 = cumulative_variance[19]  # Index 19 corresponds to the 20th component (Python indexing starts at 0)

# Save results to a text file
with open("pca_http_analysis_results.txt", "w") as f:
    f.write("Variance explained by each component:\n")
    f.write(", ".join([f"{x:.4f}" for x in explained_variance_ratio]) + "\n\n")
    f.write("Cumulative variance explained:\n")
    f.write(", ".join([f"{x:.4f}" for x in cumulative_variance]) + "\n\n")
    f.write(f"Variance retained by the first 20 components: {variance_retained_20 * 100:.2f}%\n")

# Plot cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), cumulative_variance * 100, marker='o', linestyle='--')
plt.axvline(x=20, color='r', linestyle='--', label='20 Principal Components')
plt.title('Cumulative Variance Explained by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.grid()
plt.legend()
plt.savefig("pca_http_variance_plot.png")  # Save plot as an image
plt.close()

# Transform data to the first 20 principal components
pca_20 = PCA(n_components=20)
reduced_embeddings = pca_20.fit_transform(embeddings)

# Save the reduced embeddings
reduced_df = pd.DataFrame(reduced_embeddings, columns=[f'pc_{i+1}' for i in range(20)])
reduced_df.to_csv("http_final_reduced.csv", index=False)

# Confirmation message in the text file
with open("pca_http_analysis_results.txt", "a") as f:
    f.write("\nPCA reduced embeddings saved to http_final_reduced.csv\n")
    f.write("Cumulative variance plot saved to pca_http_variance_plot.png\n")

print("PCA analysis and results saved successfully.")
