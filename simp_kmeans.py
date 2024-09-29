import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV file
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data.to_numpy()

# Function to initialize centroids
def initialize_centroids(data, k):
    np.random.seed(42)  # For reproducibility
    random_idx = np.random.choice(len(data), k, replace=False)
    centroids = data[random_idx]
    return centroids

# Function to assign points to nearest centroid
def assign_clusters(data, centroids):
    distances = np.zeros((len(data), len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(data - centroid, axis=1)  # Calculate distance from each point to each centroid
    return np.argmin(distances, axis=1)  # Return index of the closest centroid

# Function to update centroids
def update_centroids(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        new_centroids[i] = data[labels == i].mean(axis=0)  # Compute the mean of points in each cluster
    return new_centroids

# K-Means algorithm
def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)  # Assign points to the closest centroid
        new_centroids = update_centroids(data, labels, k)  # Update centroids based on cluster assignment
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Plot the clusters
def plot_clusters(data, labels, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
    plt.title('K-Means Clustering')
    plt.show()

# Main function to run the algorithm
def main():
    # Load dataset from CSV file
    file_path = './kmeans_dataset.csv'  # Replace with your file path
    data = load_dataset(file_path)
    
    # Run K-Means
    k = 2  # Number of clusters
    labels, centroids = kmeans(data, k)
    
    # Plot the results
    plot_clusters(data, labels, centroids)

if __name__ == "__main__":
    main()
